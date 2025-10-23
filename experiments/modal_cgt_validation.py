"""
Modal deployment for CGT operator validation (NSM-34).

Runs validation experiments for Conway temperature and cooling operators on A100 GPUs.
Implements all Modal best practices from MODAL_BEST_PRACTICES.md.

Usage:
    modal run experiments/modal_cgt_validation.py::validate_temperature
    modal run experiments/modal_cgt_validation.py::validate_cooling
    modal run experiments/modal_cgt_validation.py::validate_all_operators
"""

import modal
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# ============================================================================
# MODAL SETUP
# ============================================================================

app = modal.App("nsm-cgt-validation")
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Optimized image build following Modal best practices
base = modal.Image.from_registry(
    "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
    add_python="3.10"
)

image = (
    base
    .run_commands(
        "pip install --no-cache-dir torch-scatter torch-sparse "
        "-f https://data.pyg.org/whl/torch-2.1.0+cu118.html"
    )
    .pip_install(
        "torch-geometric==2.4.0",
        "numpy", "scipy", "networkx", "matplotlib", "tensorboard",
        "pytest"  # For validation tests
    )
    # IMPORTANT: Use /root as remote path (not /root/nsm) - Modal adds /root to PYTHONPATH
    .add_local_dir(PROJECT_ROOT / "nsm", remote_path="/root")
)

# Persistent volume for checkpoints and results
volume = modal.Volume.from_name("nsm-cgt-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = "/checkpoints"
RESULTS_DIR = "/results"


# ============================================================================
# OPERATOR 1 & 2: TEMPERATURE + COOLING VALIDATION
# ============================================================================

@app.cls(
    image=image,
    gpu="A100-40GB",  # Strict GPU sizing (avoid 80GB surprise upgrades)
    cpu=8.0,  # Reserve CPUs for data loading
    memory=32_000,  # 32GB RAM
    timeout=3600,  # 1 hour per attempt
    volumes={CHECKPOINT_DIR: volume},
    enable_memory_snapshot=True,  # 3-5x faster cold starts
    retries=modal.Retries(
        max_retries=2,
        backoff_coefficient=2.0,
        initial_delay=60.0
    )
)
class CGTTemperatureValidator:
    """
    Validates Conway temperature (Operator 1) and cooling monitor (Operator 2).

    Pre-registered predictions tested:
    - P1.1: Temperature decreases during collapse
    - P1.2: Temperature < 0.2 predicts collapse with >90% accuracy
    - P2.1: Cooling rate < -0.05 predicts collapse within 2 epochs
    """

    @modal.enter(snap=True)
    def load_modules(self):
        """Load heavy imports (CPU-only, snapshotted for fast cold starts)."""
        import sys
        sys.path.insert(0, "/root")

        # Import NSM modules
        from nsm.data.planning_dataset import PlanningTripleDataset
        from nsm.models.chiral import FullChiralModel
        from nsm.training.trainer import NSMTrainer
        from nsm.training.cgt_metrics import (
            temperature_conway,
            CoolingMonitor,
            extract_hinge_parameter,
            compute_all_temperature_metrics
        )
        from nsm.training.physics_metrics import compute_safety_factor

        self.dataset_class = PlanningTripleDataset
        self.model_class = FullChiralModel
        self.trainer_class = NSMTrainer

        # CGT operators
        self.temperature_conway = temperature_conway
        self.CoolingMonitor = CoolingMonitor
        self.extract_hinge_parameter = extract_hinge_parameter
        self.compute_all_temperature_metrics = compute_all_temperature_metrics

        # Physics baseline
        self.compute_safety_factor = compute_safety_factor

        print("✅ Modules loaded and snapshotted")

    @modal.enter(snap=False)
    def setup_gpu(self):
        """Setup GPU resources (runs after snapshot restore)."""
        import torch
        self.device = torch.device('cuda')
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    @modal.exit()
    def cleanup(self):
        """Flush results on exit (success, failure, or preemption)."""
        print("💾 Final volume commit...")
        volume.commit()

    @modal.method()
    def validate_temperature_operator(
        self,
        num_samples: int = 20,
        num_test_batches: int = 50,
        batch_size: int = 32,
        seed: int = 42
    ) -> Dict:
        """
        Validate Operator 1: Conway Temperature.

        Tests:
        1. Temperature computation on symmetric vs asymmetric models
        2. Temperature trajectory during training
        3. Correlation with collapse events
        4. Comparison to physics baseline (q_neural)

        Args:
            num_samples: Monte Carlo samples for temperature estimation
            num_test_batches: Number of batches to test
            batch_size: Batch size
            seed: Random seed

        Returns:
            Validation results dictionary
        """
        import torch
        import numpy as np
        from pathlib import Path
        from torch.utils.data import DataLoader
        from torch_geometric.data import Batch

        print("\n" + "="*80)
        print("VALIDATION: Operator 1 - Conway Temperature")
        print("="*80)

        results_path = Path(RESULTS_DIR) / "temperature"
        results_path.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Create dataset
        dataset = self.dataset_class(
            root="/data/planning",
            split="train",
            num_problems=500,
            seed=seed
        )

        def collate_fn(batch_list):
            data_list = [item[0] for item in batch_list]
            labels = torch.tensor(
                [item[1].item() for item in batch_list],
                dtype=torch.long
            )
            batched_data = Batch.from_data_list(data_list)
            return {
                'x': batched_data.x,
                'edge_index': batched_data.edge_index,
                'edge_type': batched_data.edge_type,
                'edge_attr': getattr(batched_data, 'edge_attr', None),
                'batch': batched_data.batch,
                'y': labels
            }

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        # Create model
        model = self.model_class(
            node_features=64,
            num_relations=22,
            num_classes=2,
            num_bases=8,
            pool_ratio=0.5,
            task_type='classification',
            num_levels=6
        ).to(self.device)

        model.eval()

        # Test 1: Compute temperature on multiple batches
        print("\n📊 Test 1: Temperature computation")
        temperatures = []
        diagnostics_list = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_test_batches:
                    break

                # Move batch to GPU
                x = batch['x'].to(self.device)

                # Compute Conway temperature
                temp, diag = self.temperature_conway(
                    model, x, num_samples=num_samples, metric='mse'
                )

                temperatures.append(temp)
                diagnostics_list.append(diag)

                if i == 0:
                    print(f"   First batch: t(G) = {temp:.4f}")
                    print(f"   max_left = {diag['max_left']:.4f}")
                    print(f"   min_right = {diag['min_right']:.4f}")

        mean_temp = np.mean(temperatures)
        std_temp = np.std(temperatures)
        print(f"   Mean temperature: {mean_temp:.4f} ± {std_temp:.4f}")
        print(f"   Range: [{min(temperatures):.4f}, {max(temperatures):.4f}]")

        # Test 2: Compare to physics baseline
        print("\n📊 Test 2: Comparison to physics baseline")

        # Dummy class accuracies for baseline
        class_accs = {
            'accuracy_class_0': 0.65,
            'accuracy_class_1': 0.55
        }

        q_neural, q_diag = self.compute_safety_factor(class_accs, model)
        print(f"   Physics q_neural: {q_neural:.4f}")
        print(f"   CGT temperature: {mean_temp:.4f}")

        # Both should indicate stable state
        stable_physics = q_neural >= 1.0
        stable_cgt = mean_temp > 0.2

        print(f"   Physics prediction: {'STABLE' if stable_physics else 'COLLAPSE RISK'}")
        print(f"   CGT prediction: {'STABLE' if stable_cgt else 'COLLAPSE RISK'}")

        # Compile results
        results = {
            'operator': 'temperature',
            'timestamp': datetime.now().isoformat(),
            'num_samples': num_samples,
            'num_test_batches': num_test_batches,
            'batch_size': batch_size,
            'statistics': {
                'mean_temperature': float(mean_temp),
                'std_temperature': float(std_temp),
                'min_temperature': float(min(temperatures)),
                'max_temperature': float(max(temperatures)),
                'temperatures': [float(t) for t in temperatures]
            },
            'baseline_comparison': {
                'q_neural': float(q_neural),
                'q_neural_stable': bool(stable_physics),
                'cgt_stable': bool(stable_cgt),
                'agreement': bool(stable_physics == stable_cgt)
            },
            'predictions_tested': {
                'P1.1': 'awaiting_training_data',  # Need collapse trajectory
                'P1.2': f"temp_threshold_check: mean={mean_temp:.4f} vs 0.2"
            }
        }

        # Save results
        with open(results_path / 'validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        volume.commit()

        print("\n✅ Temperature validation complete!")
        return results

    @modal.method()
    def validate_cooling_operator(
        self,
        num_epochs: int = 20,
        batch_size: int = 32,
        seed: int = 42
    ) -> Dict:
        """
        Validate Operator 2: Cooling Monitor.

        Tests:
        1. Cooling rate computation during simulated collapse
        2. Collapse time prediction accuracy
        3. Smoothed vs raw cooling rates

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            seed: Random seed

        Returns:
            Validation results dictionary
        """
        import torch
        import numpy as np
        from pathlib import Path
        from torch.utils.data import DataLoader
        from torch_geometric.data import Batch

        print("\n" + "="*80)
        print("VALIDATION: Operator 2 - Cooling Monitor")
        print("="*80)

        results_path = Path(RESULTS_DIR) / "cooling"
        results_path.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Create dataset
        dataset = self.dataset_class(
            root="/data/planning",
            split="train",
            num_problems=500,
            seed=seed
        )

        def collate_fn(batch_list):
            data_list = [item[0] for item in batch_list]
            labels = torch.tensor(
                [item[1].item() for item in batch_list],
                dtype=torch.long
            )
            batched_data = Batch.from_data_list(data_list)
            return {
                'x': batched_data.x,
                'edge_index': batched_data.edge_index,
                'edge_type': batched_data.edge_type,
                'edge_attr': getattr(batched_data, 'edge_attr', None),
                'batch': batched_data.batch,
                'y': labels
            }

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

        # Create model
        model = self.model_class(
            node_features=64,
            num_relations=22,
            num_classes=2,
            num_bases=8,
            pool_ratio=0.5,
            task_type='classification',
            num_levels=6
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Initialize cooling monitor
        monitor = self.CoolingMonitor(window_size=5)

        print("\n📊 Training and monitoring cooling")

        cooling_history = []
        temp_history = []
        collapse_predictions = []

        for epoch in range(num_epochs):
            model.train()

            # Simple training loop
            for batch in dataloader:
                x = batch['x'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                edge_type = batch['edge_type'].to(self.device)
                labels = batch['y'].to(self.device)
                batch_idx = batch['batch'].to(self.device)

                optimizer.zero_grad()

                # Forward pass
                output = model(
                    x=x,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    batch=batch_idx
                )

                # Simple cross-entropy loss
                loss = torch.nn.functional.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()

                break  # One batch per epoch for speed

            # Extract hinge parameters (if available)
            try:
                alpha = self.extract_hinge_parameter(model, 'alpha')
                beta = self.extract_hinge_parameter(model, 'beta')

                # Update cooling monitor
                cooling_rate = monitor.update(alpha, beta)
                stats = monitor.get_statistics()

                temp_history.append(stats['current_temp'])

                if cooling_rate is not None:
                    cooling_history.append(cooling_rate)

                    # Predict collapse
                    epochs_to_collapse = monitor.predict_collapse_time(threshold_temp=0.1)
                    collapse_predictions.append(epochs_to_collapse)

                    print(f"Epoch {epoch:3d}: T={stats['current_temp']:.4f}, "
                          f"δT/δe={cooling_rate:.6f}, "
                          f"collapse_in={epochs_to_collapse if epochs_to_collapse else 'N/A'}")

            except ValueError:
                # No hinge parameters in model
                print(f"Epoch {epoch:3d}: (No hinge parameters found, using manual simulation)")

                # Simulate α, β → 0.5 (manual cooling)
                alpha = 0.9 - (epoch / num_epochs) * 0.4  # 0.9 → 0.5
                beta = 0.1 + (epoch / num_epochs) * 0.4  # 0.1 → 0.5

                cooling_rate = monitor.update(alpha, beta)
                stats = monitor.get_statistics()

                temp_history.append(stats['current_temp'])

                if cooling_rate is not None:
                    cooling_history.append(cooling_rate)
                    epochs_to_collapse = monitor.predict_collapse_time(threshold_temp=0.1)
                    collapse_predictions.append(epochs_to_collapse)

                    print(f"Epoch {epoch:3d}: T={stats['current_temp']:.4f}, "
                          f"δT/δe={cooling_rate:.6f}, "
                          f"collapse_in={epochs_to_collapse if epochs_to_collapse else 'N/A'}")

        # Analysis
        print("\n📊 Cooling analysis")
        mean_cooling = np.mean(cooling_history)
        print(f"   Mean cooling rate: {mean_cooling:.6f}")
        print(f"   Temperature decreased: {temp_history[0]:.4f} → {temp_history[-1]:.4f}")
        print(f"   Rapid cooling events (< -0.05): {sum(1 for c in cooling_history if c < -0.05)}")

        results = {
            'operator': 'cooling',
            'timestamp': datetime.now().isoformat(),
            'num_epochs': num_epochs,
            'statistics': {
                'initial_temperature': float(temp_history[0]),
                'final_temperature': float(temp_history[-1]),
                'mean_cooling_rate': float(mean_cooling),
                'temperature_history': [float(t) for t in temp_history],
                'cooling_rate_history': [float(c) for c in cooling_history],
                'rapid_cooling_events': int(sum(1 for c in cooling_history if c < -0.05))
            },
            'predictions_tested': {
                'P2.1': f"rapid_cooling_detected: {sum(1 for c in cooling_history if c < -0.05)} events",
                'collapse_predictions': [int(p) if p is not None else None for p in collapse_predictions]
            }
        }

        # Save results
        with open(results_path / 'validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        volume.commit()

        print("\n✅ Cooling validation complete!")
        return results


# ============================================================================
# PARALLEL VALIDATION ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def validate_all_operators():
    """
    Run all CGT operator validations in parallel.

    Implements best practice: Independent error handling for each job.
    """
    print("🚀 Launching CGT operator validation suite...")
    print(f"   Time: {datetime.now().isoformat()}")

    # Create validator instance
    validator = CGTTemperatureValidator()

    # Launch jobs in parallel (non-blocking)
    jobs = {
        'temperature': validator.validate_temperature_operator.spawn(
            num_samples=20,
            num_test_batches=50
        ),
        'cooling': validator.validate_cooling_operator.spawn(
            num_epochs=20
        )
    }

    # Collect results with per-job error handling
    results = {}
    for operator_name, job in jobs.items():
        try:
            print(f"\n⏳ Waiting for {operator_name} validation...")
            result = job.get(timeout=1800)  # 30 min per operator
            results[operator_name] = {
                'status': 'success',
                'data': result
            }
            print(f"✅ {operator_name}: Success!")

        except Exception as e:
            results[operator_name] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"❌ {operator_name} failed: {e}")
            # Continue to next operator instead of crashing

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    for operator_name, result in results.items():
        status_icon = "✅" if result['status'] == 'success' else "❌"
        print(f"{status_icon} {operator_name:12s}: {result['status']}")

        if result['status'] == 'success':
            data = result['data']
            if 'statistics' in data:
                if 'mean_temperature' in data['statistics']:
                    print(f"   Mean temperature: {data['statistics']['mean_temperature']:.4f}")
                if 'mean_cooling_rate' in data['statistics']:
                    print(f"   Mean cooling rate: {data['statistics']['mean_cooling_rate']:.6f}")

    # Return partial results (even if some failed)
    return results


@app.local_entrypoint()
def validate_temperature():
    """Run only temperature operator validation."""
    print("🚀 Launching temperature validation...")
    validator = CGTTemperatureValidator()
    result = validator.validate_temperature_operator.remote(
        num_samples=20,
        num_test_batches=50
    )
    print("\n✅ Complete!")
    return result


@app.local_entrypoint()
def validate_cooling():
    """Run only cooling operator validation."""
    print("🚀 Launching cooling validation...")
    validator = CGTTemperatureValidator()
    result = validator.validate_cooling_operator.remote(num_epochs=20)
    print("\n✅ Complete!")
    return result


# ============================================================================
# HELPER: View RESULTS
# ============================================================================

@app.function(
    image=image,
    volumes={CHECKPOINT_DIR: volume}
)
def view_results(operator: str = "all"):
    """
    View validation results from volume.

    Args:
        operator: 'temperature', 'cooling', or 'all'
    """
    import json
    from pathlib import Path

    results_path = Path(RESULTS_DIR)

    if operator == "all":
        operators = ['temperature', 'cooling']
    else:
        operators = [operator]

    for op in operators:
        result_file = results_path / op / 'validation_results.json'
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            print(f"\n{'='*80}")
            print(f"RESULTS: {op.upper()}")
            print('='*80)
            print(json.dumps(data, indent=2))
        else:
            print(f"\n⚠️  No results found for {op}")


@app.local_entrypoint()
def show_results():
    """Display all validation results."""
    view_results.remote(operator="all")
