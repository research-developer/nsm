"""
Modal GPU validation script with COMBINED fix: L3 diversity regularization + adaptive training control.

This implements BOTH approaches together:
1. ARCHITECTURAL FIX: L3 diversity regularization in FullChiralModel
2. RUNTIME ADAPTATION: Adaptive hyperparameter adjustment based on physics metrics

The hypothesis is that these fixes are synergistic:
- Diversity regularization maintains "temperature" (representation spread)
- Adaptive control dynamically adjusts loss weights when physics metrics warn of collapse

Usage:
    modal run experiments/modal_combined_validation.py::validate_combined_fix
"""

import modal
import sys
from pathlib import Path

# Modal app configuration
app = modal.App("nsm-combined-fix")

# Project root for local imports
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy<2",  # Pin to NumPy 1.x for torch-scatter compatibility
        "torch==2.1.0",
        "torch-geometric==2.4.0",
        "tqdm",
    )
    .run_commands(
        "pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html"
    )
    .add_local_dir(PROJECT_ROOT, "/root/NSM", copy=True, ignore=["*.pyc", "__pycache__", ".git", "logs", "checkpoints", "data", ".pytest_cache"])
)

# Modal volume for checkpoints
volume = modal.Volume.from_name("nsm-checkpoints", create_if_missing=True)


class AdaptiveTrainingController:
    """
    Runtime adaptation based on physics metrics.

    Dynamically adjusts loss weights when collapse risk is detected:
    - If q_neural < 1.0: Increase diversity_weight
    - If temperature inverted: Increase cycle_weight
    - If Lawson Q < 1.0: Reduce learning rate
    """

    def __init__(
        self,
        initial_diversity_weight: float = 0.0,
        initial_cycle_weight: float = 0.01,
        initial_lr: float = 1e-4,
        min_diversity_weight: float = 0.0,
        max_diversity_weight: float = 0.5,
        min_cycle_weight: float = 0.01,
        max_cycle_weight: float = 0.1,
        min_lr: float = 1e-5,
        max_lr: float = 1e-3
    ):
        self.diversity_weight = initial_diversity_weight
        self.cycle_weight = initial_cycle_weight
        self.lr = initial_lr

        self.min_diversity_weight = min_diversity_weight
        self.max_diversity_weight = max_diversity_weight
        self.min_cycle_weight = min_cycle_weight
        self.max_cycle_weight = max_cycle_weight
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.adjustment_history = []

    def update(
        self,
        physics_metrics: dict,
        epoch: int,
        optimizer: any
    ) -> dict:
        """
        Update hyperparameters based on physics metrics.

        Args:
            physics_metrics: Dict from compute_all_physics_metrics
            epoch: Current epoch
            optimizer: PyTorch optimizer (for LR adjustment)

        Returns:
            Dict of adjustments made
        """
        adjustments = {
            'epoch': epoch,
            'diversity_weight_old': self.diversity_weight,
            'cycle_weight_old': self.cycle_weight,
            'lr_old': self.lr,
            'actions': []
        }

        # Action 1: Increase diversity weight if q_neural < 1.0
        if physics_metrics['q_neural'] < 1.0:
            old_weight = self.diversity_weight
            # Increase by 50% (multiplicative), capped at max
            self.diversity_weight = min(
                self.diversity_weight * 1.5 + 0.05,  # Add 0.05 if starting at 0
                self.max_diversity_weight
            )
            adjustments['actions'].append(
                f"⚡ Increased diversity_weight: {old_weight:.4f} → {self.diversity_weight:.4f} (q={physics_metrics['q_neural']:.3f} < 1)"
            )

        # Action 2: Increase cycle weight if temperature inverted
        if physics_metrics.get('profile_type') == 'inverted':
            old_weight = self.cycle_weight
            # Increase by 30%
            self.cycle_weight = min(
                self.cycle_weight * 1.3,
                self.max_cycle_weight
            )
            adjustments['actions'].append(
                f"⚡ Increased cycle_weight: {old_weight:.4f} → {self.cycle_weight:.4f} (temperature inverted)"
            )

        # Action 3: Reduce LR if Lawson Q < 0.5 (deep subignition)
        if physics_metrics['Q_factor'] < 0.5:
            old_lr = self.lr
            # Reduce by 20%
            self.lr = max(
                self.lr * 0.8,
                self.min_lr
            )
            # Apply to optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
            adjustments['actions'].append(
                f"⚡ Reduced learning_rate: {old_lr:.6f} → {self.lr:.6f} (Q={physics_metrics['Q_factor']:.3f} < 0.5)"
            )

        # Action 4: Restore diversity weight if system stable
        if physics_metrics['q_neural'] > 1.5 and self.diversity_weight > self.min_diversity_weight:
            old_weight = self.diversity_weight
            # Gradually reduce (don't eliminate entirely)
            self.diversity_weight = max(
                self.diversity_weight * 0.9,
                self.min_diversity_weight
            )
            adjustments['actions'].append(
                f"⚡ Reduced diversity_weight: {old_weight:.4f} → {self.diversity_weight:.4f} (q={physics_metrics['q_neural']:.3f} > 1.5, stable)"
            )

        adjustments['diversity_weight_new'] = self.diversity_weight
        adjustments['cycle_weight_new'] = self.cycle_weight
        adjustments['lr_new'] = self.lr

        self.adjustment_history.append(adjustments)

        return adjustments


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/checkpoints": volume}
)
def validate_combined_fix():
    """
    Validate 6-level chiral architecture with COMBINED fix:
    1. L3 diversity regularization (architectural)
    2. Adaptive training control (runtime)
    """
    import json
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch
    from datetime import datetime
    from tqdm import tqdm

    # Add NSM to path
    sys.path.insert(0, "/root/NSM")

    from nsm.models.chiral import FullChiralModel
    from nsm.training.chiral_loss import ChiralCompositeLoss, compute_class_balance_metrics
    from nsm.training.physics_metrics import compute_all_physics_metrics
    from nsm.data.planning_dataset import PlanningTripleDataset

    print("="*70)
    print("COMBINED FIX VALIDATION - NSM-33")
    print("="*70)
    print("\nTesting synergistic approach:")
    print("  1. ARCHITECTURAL: L3 diversity regularization")
    print("  2. RUNTIME: Adaptive hyperparameter control")
    print("="*70)

    # Configuration
    config = {
        "variant": "6level_combined_fix",
        "epochs": 20,  # More epochs to test adaptation
        "batch_size": 64,
        "learning_rate": 1e-4,
        "seed": 42,
        "pool_ratio": 0.5,
        "dropout": 0.1,
        "patience": 20,

        # Initial loss weights (will be adapted)
        "task_weight": 1.0,
        "aux_weight": 0.3,
        "cycle_weight": 0.01,  # Will increase if needed
        "diversity_weight": 0.0,  # Will increase if needed (starts at 0)

        # Adaptive control ranges
        "max_diversity_weight": 0.3,
        "max_cycle_weight": 0.1,
        "min_lr": 1e-5,

        # Optional focal loss
        "use_focal_loss": False,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,

        # Physics metrics
        "track_physics_metrics": True,
        "task_complexity": 1.0,

        # Enable adaptive control
        "use_adaptive_control": True
    }

    torch.manual_seed(config["seed"])

    # Load dataset
    print("\nLoading Planning dataset...")
    full_dataset = PlanningTripleDataset(root="/tmp/planning", split="train", num_problems=4100)

    # Materialize all graphs into a list
    print(f"Total dataset size: {len(full_dataset)}")
    all_graphs = [full_dataset[i] for i in range(len(full_dataset))]
    print(f"Materialized {len(all_graphs)} graphs")

    # Split into train/val
    train_size = 2000
    train_graphs = all_graphs[:train_size]
    val_graphs = all_graphs[train_size:]

    # Create DataLoaders with explicit collate function
    def pyg_collate(data_list):
        graphs = [item[0] for item in data_list]
        labels = torch.tensor([item[1] for item in data_list])
        batch = Batch.from_data_list(graphs)
        batch.y = labels
        return batch

    print(f"Train samples: {len(train_graphs)}")
    print(f"Val samples: {len(val_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=config["batch_size"], shuffle=True, collate_fn=pyg_collate)
    val_loader = DataLoader(val_graphs, batch_size=config["batch_size"], shuffle=False, collate_fn=pyg_collate)

    # Get data properties from first batch
    print("Fetching first batch...")
    sample = next(iter(train_loader))
    node_features = sample.x.size(1)
    num_relations = int(sample.edge_type.max().item()) + 1
    num_classes = 2

    print(f"\nDataset properties:")
    print(f"  Node features: {node_features}")
    print(f"  Num relations: {num_relations}")
    print(f"  Num classes: {num_classes}")

    # Initialize model (with L3 diversity regularization)
    print("\nInitializing FullChiralModel with L3 diversity regularization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FullChiralModel(
        node_features=node_features,
        num_relations=num_relations,
        num_classes=num_classes,
        pool_ratio=config["pool_ratio"],
        task_type='classification',
        dropout=config["dropout"]
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize adaptive controller
    controller = None
    if config["use_adaptive_control"]:
        print("\nInitializing AdaptiveTrainingController...")
        controller = AdaptiveTrainingController(
            initial_diversity_weight=config["diversity_weight"],
            initial_cycle_weight=config["cycle_weight"],
            initial_lr=config["learning_rate"],
            max_diversity_weight=config["max_diversity_weight"],
            max_cycle_weight=config["max_cycle_weight"],
            min_lr=config["min_lr"]
        )

    # Initialize loss function
    criterion = ChiralCompositeLoss(
        task_weight=config["task_weight"],
        aux_weight=config["aux_weight"],
        cycle_weight=config["cycle_weight"],
        diversity_weight=config["diversity_weight"],
        use_focal_loss=config["use_focal_loss"],
        focal_alpha=config["focal_alpha"],
        focal_gamma=config["focal_gamma"]
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training loop
    print("\n" + "="*70)
    print("TRAINING WITH COMBINED FIX")
    print("="*70)

    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    history = []

    for epoch in range(config["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        train_loss_task = 0.0
        train_loss_aux = 0.0
        train_loss_cycle = 0.0
        train_loss_diversity = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"):
            batch = batch.to(device)

            # Forward pass
            output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)

            # Update loss weights from controller
            if controller is not None:
                criterion.diversity_weight = controller.diversity_weight
                criterion.cycle_weight = controller.cycle_weight

            # Compute loss
            loss_dict = criterion(output, batch.y, task_type='classification')

            # Backward
            optimizer.zero_grad()
            loss_dict['loss'].backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss_dict['loss'].item()
            train_loss_task += loss_dict['loss_task'].item()
            train_loss_aux += loss_dict['loss_task_aux'].item()
            train_loss_cycle += loss_dict['loss_cycle'].item()
            train_loss_diversity += loss_dict.get('loss_diversity', 0.0)

        train_loss /= len(train_loader)
        train_loss_task /= len(train_loader)
        train_loss_aux /= len(train_loader)
        train_loss_cycle /= len(train_loader)
        train_loss_diversity /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        val_loss_task = 0.0
        val_loss_aux = 0.0
        val_loss_cycle = 0.0
        val_loss_diversity = 0.0
        correct_total = 0
        correct_class_0 = 0
        correct_class_1 = 0
        total_class_0 = 0
        total_class_1 = 0
        total = 0

        # For physics metrics: collect level representations
        all_level_reps_l1 = []
        all_level_reps_l2 = []
        all_level_reps_l3 = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]"):
                batch = batch.to(device)

                # Forward pass
                output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)

                # Collect level representations for physics metrics
                if 'x_l1' in output:
                    all_level_reps_l1.append(output['x_l1'].cpu())
                if 'x_l2' in output:
                    all_level_reps_l2.append(output['x_l2'].cpu())
                if 'x_l3' in output:
                    all_level_reps_l3.append(output['x_l3'].cpu())

                # Compute loss
                loss_dict = criterion(output, batch.y, task_type='classification')

                val_loss += loss_dict['loss'].item()
                val_loss_task += loss_dict['loss_task'].item()
                val_loss_aux += loss_dict['loss_task_aux'].item()
                val_loss_cycle += loss_dict['loss_cycle'].item()
                val_loss_diversity += loss_dict.get('loss_diversity', 0.0)

                # Accuracy
                pred = output['logits'].argmax(dim=1)
                correct_total += (pred == batch.y).sum().item()
                total += batch.y.size(0)

                # Per-class accuracy
                for cls in [0, 1]:
                    mask = (batch.y == cls)
                    if mask.sum() > 0:
                        if cls == 0:
                            correct_class_0 += (pred[mask] == cls).sum().item()
                            total_class_0 += mask.sum().item()
                        else:
                            correct_class_1 += (pred[mask] == cls).sum().item()
                            total_class_1 += mask.sum().item()

        val_loss /= len(val_loader)
        val_loss_task /= len(val_loader)
        val_loss_aux /= len(val_loader)
        val_loss_cycle /= len(val_loader)
        val_loss_diversity /= len(val_loader)
        val_accuracy = correct_total / total
        val_accuracy_class_0 = correct_class_0 / total_class_0 if total_class_0 > 0 else 0
        val_accuracy_class_1 = correct_class_1 / total_class_1 if total_class_1 > 0 else 0
        class_balance_delta = abs(val_accuracy_class_0 - val_accuracy_class_1)

        # ===== PHYSICS METRICS =====
        physics_metrics = {}
        if config["track_physics_metrics"]:
            # Prepare class accuracies
            class_accs = {
                'accuracy_class_0': val_accuracy_class_0,
                'accuracy_class_1': val_accuracy_class_1
            }

            # Prepare level representations (concatenate batches)
            level_reps = {}
            if all_level_reps_l1:
                level_reps['L1'] = torch.cat(all_level_reps_l1, dim=0)
            if all_level_reps_l2:
                level_reps['L2'] = torch.cat(all_level_reps_l2, dim=0)
            if all_level_reps_l3:
                level_reps['L3'] = torch.cat(all_level_reps_l3, dim=0)

            # Compute all physics metrics
            physics_metrics = compute_all_physics_metrics(
                model=model,
                class_accuracies=class_accs,
                level_representations=level_reps,
                epoch=epoch + 1,
                task_complexity=config["task_complexity"]
            )

        # ===== ADAPTIVE CONTROL =====
        adjustments = None
        if controller is not None and physics_metrics:
            adjustments = controller.update(physics_metrics, epoch + 1, optimizer)

        # Log standard metrics
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*70}")
        print(f"  Train Loss: {train_loss:.4f} (task: {train_loss_task:.4f}, aux: {train_loss_aux:.4f}, cycle: {train_loss_cycle:.4f}, div: {train_loss_diversity:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (task: {val_loss_task:.4f}, aux: {val_loss_aux:.4f}, cycle: {val_loss_cycle:.4f}, div: {val_loss_diversity:.4f})")
        print(f"  Val Accuracy: {val_accuracy:.4f} (class 0: {val_accuracy_class_0:.4f}, class 1: {val_accuracy_class_1:.4f})")
        print(f"  Class Balance Δ: {class_balance_delta:.4f}")

        # Log current loss weights
        if controller is not None:
            print(f"\n  Current Loss Weights:")
            print(f"    diversity_weight: {controller.diversity_weight:.4f}")
            print(f"    cycle_weight: {controller.cycle_weight:.4f}")
            print(f"    learning_rate: {controller.lr:.6f}")

        # Log physics metrics
        if physics_metrics:
            print(f"\n  Physics Metrics:")
            print(f"    q_neural (safety factor): {physics_metrics['q_neural']:.3f} [{physics_metrics['stability']}]")
            print(f"    Coupling strength: {physics_metrics['coupling_strength']:.3f}")

            if 'T_L1' in physics_metrics:
                print(f"    Temperature L1: {physics_metrics['T_L1']:.3f}")
            if 'T_L2' in physics_metrics:
                print(f"    Temperature L2: {physics_metrics['T_L2']:.3f}")
            if 'T_L3' in physics_metrics:
                print(f"    Temperature L3: {physics_metrics['T_L3']:.3f}")
            if 'T_gradient' in physics_metrics:
                print(f"    Temperature gradient: {physics_metrics['T_gradient']:.3f} [{physics_metrics['profile_type']}]")

            print(f"    Lawson Q factor: {physics_metrics['Q_factor']:.3f} [{physics_metrics['status']}]")

            # Display warnings
            if physics_metrics['warnings']:
                print(f"\n  ⚠️  WARNINGS [{physics_metrics['alert_level']}]:")
                for warning in physics_metrics['warnings']:
                    print(f"    {warning}")

        # Log adaptive adjustments
        if adjustments and adjustments['actions']:
            print(f"\n  Adaptive Adjustments:")
            for action in adjustments['actions']:
                print(f"    {action}")

        # Save epoch data
        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_loss_task": train_loss_task,
            "train_loss_aux": train_loss_aux,
            "train_loss_cycle": train_loss_cycle,
            "train_loss_diversity": train_loss_diversity,
            "val_loss": val_loss,
            "val_loss_task": val_loss_task,
            "val_loss_aux": val_loss_aux,
            "val_loss_cycle": val_loss_cycle,
            "val_loss_diversity": val_loss_diversity,
            "val_accuracy": val_accuracy,
            "val_accuracy_class_0": val_accuracy_class_0,
            "val_accuracy_class_1": val_accuracy_class_1,
            "class_balance_delta": class_balance_delta,
        }

        # Add physics metrics to history
        if physics_metrics:
            epoch_data["physics_metrics"] = {
                "q_neural": physics_metrics['q_neural'],
                "stability": physics_metrics['stability'],
                "coupling_strength": physics_metrics['coupling_strength'],
                "T_L1": physics_metrics.get('T_L1', 0.0),
                "T_L2": physics_metrics.get('T_L2', 0.0),
                "T_L3": physics_metrics.get('T_L3', 0.0),
                "T_gradient": physics_metrics.get('T_gradient', 0.0),
                "profile_type": physics_metrics.get('profile_type', 'unknown'),
                "Q_factor": physics_metrics['Q_factor'],
                "lawson_status": physics_metrics['status'],
                "alert_level": physics_metrics['alert_level'],
                "warnings": physics_metrics['warnings']
            }

        # Add adaptive adjustments to history
        if adjustments:
            epoch_data["adaptive_adjustments"] = adjustments

        history.append(epoch_data)

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            patience_counter = 0
            print(f"\n  ✓ New best accuracy: {best_val_accuracy:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n  Early stopping triggered (patience={config['patience']})")
                break

    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS & COMBINED FIX ANALYSIS")
    print("="*70)

    results = {
        "variant_name": "6level_combined_fix",
        "config": config,
        "epochs_trained": epoch + 1,
        "training_time_seconds": None,  # TODO: track time
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "final_metrics": history[-1] if history else {},
        "history": history,
        "status": "completed"
    }

    if controller is not None:
        results["adjustment_history"] = controller.adjustment_history

    print(f"\nBest Val Accuracy: {best_val_accuracy:.4f}")
    print(f"Final Class Balance Δ: {history[-1]['class_balance_delta']:.4f}")
    print(f"Final Cycle Loss: {history[-1]['val_loss_cycle']:.4f}")
    print(f"Final Diversity Loss: {history[-1]['val_loss_diversity']:.4f}")

    # Analyze synergy
    print(f"\n{'='*70}")
    print("SYNERGY ANALYSIS")
    print(f"{'='*70}")

    if controller is not None and len(controller.adjustment_history) > 0:
        print(f"\nAdaptive Control Summary:")
        print(f"  Total adjustments made: {len([a for a in controller.adjustment_history if a['actions']])}")
        print(f"  Final diversity_weight: {controller.diversity_weight:.4f} (initial: {config['diversity_weight']:.4f})")
        print(f"  Final cycle_weight: {controller.cycle_weight:.4f} (initial: {config['cycle_weight']:.4f})")
        print(f"  Final learning_rate: {controller.lr:.6f} (initial: {config['learning_rate']:.6f})")

    # Comparison to baseline
    baseline_accuracy = 0.5126
    baseline_balance_delta = 0.2960

    print(f"\nComparison to 3-level fusion baseline:")
    print(f"  Accuracy: {best_val_accuracy:.4f} vs {baseline_accuracy:.4f} (Δ {best_val_accuracy - baseline_accuracy:+.4f})")
    print(f"  Balance Δ: {history[-1]['class_balance_delta']:.4f} vs {baseline_balance_delta:.4f} (Δ {history[-1]['class_balance_delta'] - baseline_balance_delta:+.4f})")

    # Success criteria from NSM-32
    if best_val_accuracy >= 0.55 and history[-1]['class_balance_delta'] < 0.40:
        print("\n✅ SUCCESS: Passed primary criteria (accuracy ≥55%, balance Δ <40%)")
    else:
        print("\n⚠️  PARTIAL: Did not meet all primary criteria")
        if best_val_accuracy < 0.55:
            print(f"   - Accuracy below target: {best_val_accuracy:.4f} < 0.55")
        if history[-1]['class_balance_delta'] >= 0.40:
            print(f"   - Balance delta above target: {history[-1]['class_balance_delta']:.4f} >= 0.40")

    # Save results
    output_path = "/tmp/6level_combined_fix_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


@app.local_entrypoint()
def main():
    """
    Local entrypoint for running combined fix validation.
    """
    print("Launching combined fix validation on Modal...")
    results = validate_combined_fix.remote()

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"\nFinal Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Final Balance Δ: {results['final_metrics']['class_balance_delta']:.4f}")

    # Display physics metrics summary
    if "physics_metrics" in results['final_metrics']:
        pm = results['final_metrics']['physics_metrics']
        print(f"\nFinal Physics Metrics:")
        print(f"  q_neural: {pm['q_neural']:.3f} [{pm['stability']}]")
        print(f"  Q factor: {pm['Q_factor']:.3f} [{pm['lawson_status']}]")
        print(f"  Alert level: {pm['alert_level']}")
