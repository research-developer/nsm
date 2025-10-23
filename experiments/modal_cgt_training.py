"""
Integrated Training + CGT Validation (NSM-34 Workstream A)

Trains a model while tracking Conway temperature and cooling dynamics to validate
collapse prediction operators. Results are logged in AGENTS.md-compliant format.

Usage:
    # Quick 5-epoch test
    modal run experiments/modal_cgt_training.py::train_with_cgt_tracking

    # Full 50-epoch production run
    modal run experiments/modal_cgt_training.py::train_with_cgt_tracking --epochs=50
"""

import modal
import json
from pathlib import Path
from datetime import datetime

# Modal setup
app = modal.App("nsm-cgt-training")
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Shared image with all dependencies
# Note: torch-scatter/sparse need pre-built wheels from PyG
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch==2.1.0",
        "numpy<2",
        "scipy",
        "tqdm",
        "networkx"
    )
    .run_commands(
        "pip install torch-scatter torch-sparse torch-geometric==2.4.0 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html"
    )
    .add_local_dir(PROJECT_ROOT / "nsm", remote_path="/root/nsm")
    .add_local_dir(PROJECT_ROOT / "experiments", remote_path="/root/experiments")
)

# Persistent volume for checkpoints and logs
volume = modal.Volume.from_name("nsm-cgt-training", create_if_missing=True)
VOLUME_DIR = "/vol"
CHECKPOINT_DIR = f"{VOLUME_DIR}/checkpoints"
RESULTS_DIR = f"{VOLUME_DIR}/results"


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=8.0,
    memory=32_000,
    timeout=7200,  # 2 hours
    volumes={VOLUME_DIR: volume},
    enable_memory_snapshot=True
)
def train_with_cgt_tracking(
    epochs: int = 5,
    domain: str = "planning",
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    cycle_weight: float = 0.01,
    num_problems: int = 2858,
    checkpoint_freq: int = 5,
    cgt_sample_freq: int = 1  # Measure CGT operators every N epochs
):
    """
    Train model with integrated CGT operator tracking.

    Tracks:
    - Conway temperature t(G) each epoch
    - Cooling rate (α,β → 0.5)
    - Collapse predictions (P1.2, P2.1)
    - Physics baseline (q_neural) for comparison
    """
    import torch
    import torch.nn as nn
    from torch_geometric.loader import DataLoader
    import sys
    import numpy as np

    sys.path.insert(0, "/root")

    from nsm.data.planning import PlanningDataset
    from nsm.models.chiral import FullChiralModel
    from nsm.training.cgt_metrics import (
        temperature_conway,
        CoolingMonitor,
        extract_hinge_parameter,
        compute_all_temperature_metrics
    )

    print("\n" + "="*80)
    print(f"CGT-TRACKED TRAINING: {domain.upper()} ({epochs} epochs)")
    print("="*80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # Initialize run data
    run_id = f"cgt_{domain}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    start_time = datetime.utcnow()

    # Setup dataset
    print("📊 Loading dataset...")
    dataset = PlanningDataset(num_problems=num_problems, split='train')
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"   Train: {train_size} | Val: {val_size}")

    # Initialize model
    print("🏗️  Initializing 6-level chiral model...")
    model = FullChiralModel(
        node_features=64,
        num_relations=22,
        num_classes=2,
        num_bases=8,
        pool_ratio=0.5,
        task_type='classification',
        dropout=0.1
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Initialize CGT monitors
    cooling_monitor = CoolingMonitor(window_size=5)

    # Storage for metrics
    metrics_history = []
    cgt_history = []

    # Training loop
    print(f"\n🚀 Starting training ({epochs} epochs)...\n")

    for epoch in range(epochs):
        # =================================================================
        # TRAINING PHASE
        # =================================================================
        model.train()
        train_loss = 0.0
        train_cycle_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.cuda()
            optimizer.zero_grad()

            output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)

            # Task loss
            task_loss = criterion(output['logits'], batch.y)

            # Cycle loss
            cycle_loss = output['cycle_loss_upper'] + output['cycle_loss_lower'] + output['cycle_loss_cross']

            # Total loss
            loss = task_loss + cycle_weight * cycle_loss
            loss.backward()
            optimizer.step()

            train_loss += task_loss.item()
            train_cycle_loss += cycle_loss.item()

            pred = output['logits'].argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        avg_cycle_loss = train_cycle_loss / len(train_loader)

        # =================================================================
        # VALIDATION PHASE
        # =================================================================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_class_0 = 0
        val_class_1 = 0
        class_0_total = 0
        class_1_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.cuda()
                output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)

                loss = criterion(output['logits'], batch.y)
                val_loss += loss.item()

                pred = output['logits'].argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)

                # Track per-class accuracy
                mask_0 = (batch.y == 0)
                mask_1 = (batch.y == 1)
                val_class_0 += (pred[mask_0] == 0).sum().item()
                val_class_1 += (pred[mask_1] == 1).sum().item()
                class_0_total += mask_0.sum().item()
                class_1_total += mask_1.sum().item()

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        acc_class_0 = val_class_0 / class_0_total if class_0_total > 0 else 0.0
        acc_class_1 = val_class_1 / class_1_total if class_1_total > 0 else 0.0
        balance_delta = abs(acc_class_0 - acc_class_1)

        # =================================================================
        # CGT OPERATOR TRACKING
        # =================================================================
        cgt_metrics = {}

        if epoch % cgt_sample_freq == 0:
            print(f"\n📐 Epoch {epoch+1}/{epochs} - Computing CGT operators...")

            # Sample a batch for temperature measurement
            sample_batch = next(iter(val_loader)).cuda()

            # Measure Conway temperature
            temp, temp_diag = temperature_conway(
                model,
                sample_batch.x,
                num_samples=10,
                metric='mse'
            )

            # Extract hinge parameters
            alpha = extract_hinge_parameter(model, level=2, parameter='alpha')
            beta = extract_hinge_parameter(model, level=2, parameter='beta')

            # Update cooling monitor
            cooling_rate = cooling_monitor.update(alpha, beta)
            cooling_stats = cooling_monitor.get_statistics()
            collapse_time = cooling_monitor.predict_collapse_time(threshold_temp=0.1)

            # Compute all temperature metrics
            all_temps = compute_all_temperature_metrics(
                model, sample_batch.x, num_samples=10
            )

            # Physics baseline (q_neural)
            q_neural = (acc_class_0 * acc_class_1 * 4) if (acc_class_0 > 0 and acc_class_1 > 0) else 0.0

            cgt_metrics = {
                'temperature_conway': float(temp),
                'temperature_neural': float(cooling_stats['current_temp']),
                'cooling_rate': float(cooling_rate) if cooling_rate is not None else None,
                'collapse_predicted_in_epochs': int(collapse_time) if collapse_time is not None else None,
                'alpha': float(alpha),
                'beta': float(beta),
                'q_neural': float(q_neural),
                'max_left': float(temp_diag['max_left']),
                'min_right': float(temp_diag['min_right']),
                'temperature_mse': float(all_temps['temperature_mse']),
                'temperature_cosine': float(all_temps['temperature_cosine'])
            }

            # Collapse risk assessment
            temp_risk = "HIGH" if temp < 0.2 else ("MEDIUM" if temp < 0.5 else "LOW")
            cooling_risk = "HIGH" if (cooling_rate and cooling_rate < -0.05) else ("MEDIUM" if (cooling_rate and cooling_rate < 0) else "LOW")

            print(f"   Temperature: {temp:.4f} (risk: {temp_risk})")
            print(f"   Neural Temp: {cooling_stats['current_temp']:.4f}")
            print(f"   Cooling Rate: {cooling_rate:.6f if cooling_rate else 'N/A'} (risk: {cooling_risk})")
            print(f"   α={alpha:.4f}, β={beta:.4f}")
            print(f"   Q_neural: {q_neural:.4f}")

            if collapse_time is not None:
                print(f"   ⚠️  Collapse predicted in {collapse_time} epochs")

            cgt_history.append({
                'epoch': epoch + 1,
                **cgt_metrics
            })

        # Store epoch metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': float(avg_train_loss),
            'train_accuracy': float(train_acc),
            'val_loss': float(avg_val_loss),
            'val_accuracy': float(val_acc),
            'accuracy_class_0': float(acc_class_0),
            'accuracy_class_1': float(acc_class_1),
            'balance_delta': float(balance_delta),
            'cycle_loss': float(avg_cycle_loss),
            **cgt_metrics
        }

        metrics_history.append(epoch_metrics)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train: loss={avg_train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={avg_val_loss:.4f}, acc={val_acc:.4f}")
        print(f"  Balance: Δ={balance_delta:.4f} (C0:{acc_class_0:.3f}, C1:{acc_class_1:.3f})")
        print(f"  Cycle: {avg_cycle_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = Path(CHECKPOINT_DIR) / f"{run_id}_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': epoch_metrics
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")

    end_time = datetime.utcnow()
    training_time = (end_time - start_time).total_seconds()

    # =================================================================
    # FINAL RESULTS
    # =================================================================
    best_epoch = max(metrics_history, key=lambda x: x['val_accuracy'])
    final_metrics = metrics_history[-1]

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Epoch: {best_epoch['epoch']}")
    print(f"Best Val Accuracy: {best_epoch['val_accuracy']:.4f}")
    print(f"Final Val Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"Final Balance Δ: {final_metrics['balance_delta']:.4f}")
    print(f"Training Time: {training_time:.1f}s ({training_time/60:.1f} min)")

    if cgt_history:
        print(f"\n📊 CGT Operator Summary:")
        temp_traj = [f"{h['temperature_conway']:.4f}" for h in cgt_history]
        cooling_traj = [f"{h['temperature_neural']:.4f}" for h in cgt_history]
        print(f"   Temperature trajectory: {temp_traj}")
        print(f"   Cooling trajectory: {cooling_traj}")

        # Check collapse predictions
        any_temp_collapse = any(h['temperature_conway'] < 0.2 for h in cgt_history)
        any_cooling_collapse = any((h['cooling_rate'] is not None and h['cooling_rate'] < -0.05) for h in cgt_history)

        print(f"\n   Prediction P1.2 (temp < 0.2): {'TRIGGERED' if any_temp_collapse else 'Not triggered'}")
        print(f"   Prediction P2.1 (rapid cooling): {'TRIGGERED' if any_cooling_collapse else 'Not triggered'}")

    # =================================================================
    # FORMAT RESULTS FOR LOGGING
    # =================================================================

    # Prepare experiment entry for training_log.jsonl
    experiment_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "run_data": {
            "run_id": run_id,
            "domain": domain,
            "status": "completed",
            "dataset_config": {
                "domain": domain,
                "split": "train",
                "total_size": num_problems,
                "train_size": train_size,
                "val_size": val_size,
                "is_balanced": True
            },
            "hyperparameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "cycle_loss_weight": cycle_weight,
                "pool_ratio": 0.5,
                "dropout": 0.1,
                "cgt_sample_freq": cgt_sample_freq
            },
            "architecture": {
                "variant": "6level_full_cgt",
                "description": "6-level chiral with CGT operator tracking",
                "num_levels": 6,
                "node_features": 64,
                "num_relations": 22
            },
            "metrics_history": metrics_history,
            "cgt_history": cgt_history,
            "best_val_loss": float(best_epoch['val_loss']),
            "best_val_accuracy": float(best_epoch['val_accuracy']),
            "best_epoch": int(best_epoch['epoch']),
            "final_metrics": {
                "accuracy": float(final_metrics['val_accuracy']),
                "accuracy_class_0": float(final_metrics['accuracy_class_0']),
                "accuracy_class_1": float(final_metrics['accuracy_class_1']),
                "class_balance_delta": float(final_metrics['balance_delta']),
                "task_loss": float(final_metrics['val_loss']),
                "cycle_loss": float(final_metrics['cycle_loss']),
                **({k: v for k, v in final_metrics.items() if k.startswith('temperature_') or k in ['alpha', 'beta', 'q_neural', 'cooling_rate']} if cgt_history else {})
            },
            "training_time_seconds": float(training_time),
            "start_time": start_time.isoformat() + "Z",
            "end_time": end_time.isoformat() + "Z",
            "experiment_type": "cgt_collapse_prediction",
            "findings": f"CGT-tracked training: {'temperature collapse risk detected' if any_temp_collapse else 'stable temperature'}, {'rapid cooling detected' if any_cooling_collapse else 'stable cooling'}"
        }
    }

    # Save results
    results_path = Path(RESULTS_DIR) / f"{run_id}_results.json"
    with open(results_path, 'w') as f:
        json.dump(experiment_entry, f, indent=2)

    print(f"\n💾 Results saved: {results_path}")
    print(f"📝 Ready for appending to experiments/training_log.jsonl")

    return experiment_entry


@app.local_entrypoint()
def main(epochs: int = 5):
    """
    Run CGT-tracked training with specified epochs.

    Args:
        epochs: Number of training epochs (default: 5 for quick test)
    """
    print(f"🚀 Launching CGT-tracked training ({epochs} epochs)...")
    result = train_with_cgt_tracking.remote(epochs=epochs)

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"Run ID: {result['run_data']['run_id']}")
    print(f"Final Accuracy: {result['run_data']['final_metrics']['accuracy']:.4f}")
    print(f"Balance Δ: {result['run_data']['final_metrics']['class_balance_delta']:.4f}")

    if 'temperature_conway' in result['run_data']['final_metrics']:
        print(f"Final Temperature: {result['run_data']['final_metrics']['temperature_conway']:.4f}")
        print(f"Final Q_neural: {result['run_data']['final_metrics']['q_neural']:.4f}")

    print(f"\n📊 View detailed results at Modal dashboard")
    print(f"💾 Results saved to volume: nsm-cgt-training")

    return result
