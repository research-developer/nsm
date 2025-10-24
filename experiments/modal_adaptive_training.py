"""
Physics-based adaptive training control for 6-level chiral architecture.

Implements real-time training adjustments based on physics-inspired metrics:
- When q_neural < 1.5: Boost diversity_weight to prevent collapse
- When temp_gradient < -0.1 (inverted): Boost cycle_weight to restore symmetry
- When Q_factor < 0.5: Reduce learning rate to allow consolidation

This adaptive control system treats training as a plasma confinement problem,
using fusion physics metrics to maintain stability and prevent class collapse.

References:
- NSM-33: Physics-Inspired Collapse Prediction Metrics
- NSM-32: 6-Level Chiral Architecture validation

Usage:
    modal run experiments/modal_adaptive_training.py::adaptive_train
"""

import modal
import sys
from pathlib import Path

# Modal app configuration
app = modal.App("nsm-adaptive-training")

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


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/checkpoints": volume}
)
def adaptive_train():
    """
    Train 6-level chiral model with physics-based adaptive control.

    Monitors physics metrics each epoch and adjusts hyperparameters:
    - Diversity weight: Prevents representation collapse
    - Cycle weight: Maintains WHY/WHAT symmetry
    - Learning rate: Controls training speed for stability
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
    print("PHYSICS-BASED ADAPTIVE TRAINING CONTROL - NSM-33")
    print("="*70)
    print("\nAdaptive control rules:")
    print("  1. q_neural < 1.5 → Increase diversity_weight by 0.03")
    print("  2. temp_gradient < -0.1 → Increase cycle_weight by 0.02")
    print("  3. Q_factor < 0.5 → Reduce learning_rate by 0.9x")
    print("="*70)

    # Configuration
    config = {
        "variant": "6level_adaptive",
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "seed": 42,
        "pool_ratio": 0.5,
        "dropout": 0.1,
        "patience": 20,

        # Loss weights (will be dynamically adjusted)
        "task_weight": 1.0,
        "aux_weight": 0.3,
        "cycle_weight_initial": 0.01,
        "diversity_weight_initial": 0.0,

        # Adaptive control parameters
        "diversity_boost": 0.03,
        "cycle_boost": 0.02,
        "lr_decay": 0.9,
        "max_diversity_weight": 0.3,
        "max_cycle_weight": 0.1,
        "min_learning_rate": 1e-6,

        # Physics thresholds
        "q_neural_threshold": 1.5,
        "temp_gradient_threshold": -0.1,
        "Q_factor_threshold": 0.5,
        "intervention_start_epoch": 1,

        # Optional focal loss
        "use_focal_loss": False,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,

        # Physics metrics
        "track_physics_metrics": True,
        "task_complexity": 1.0
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

    # Initialize model
    print("\nInitializing FullChiralModel (6-level) with adaptive control...")
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

    # Initialize dynamic loss weights
    cycle_weight = config["cycle_weight_initial"]
    diversity_weight = config["diversity_weight_initial"]

    # Initialize loss function (will update weights during training)
    criterion = ChiralCompositeLoss(
        task_weight=config["task_weight"],
        aux_weight=config["aux_weight"],
        cycle_weight=cycle_weight,
        diversity_weight=diversity_weight,
        use_focal_loss=config["use_focal_loss"],
        focal_alpha=config["focal_alpha"],
        focal_gamma=config["focal_gamma"]
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Intervention tracking
    interventions = []

    # Training loop
    print("\n" + "="*70)
    print("ADAPTIVE TRAINING WITH PHYSICS-BASED CONTROL")
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

            # Update criterion weights (in case they changed)
            criterion.cycle_weight = cycle_weight
            criterion.diversity_weight = diversity_weight

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
            train_loss_diversity += loss_dict['loss_diversity'].item()

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

                # Update criterion weights
                criterion.cycle_weight = cycle_weight
                criterion.diversity_weight = diversity_weight

                # Compute loss
                loss_dict = criterion(output, batch.y, task_type='classification')

                val_loss += loss_dict['loss'].item()
                val_loss_task += loss_dict['loss_task'].item()
                val_loss_aux += loss_dict['loss_task_aux'].item()
                val_loss_cycle += loss_dict['loss_cycle'].item()
                val_loss_diversity += loss_dict['loss_diversity'].item()

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

        # Log standard metrics
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*70}")
        print(f"  Train Loss: {train_loss:.4f} (task: {train_loss_task:.4f}, aux: {train_loss_aux:.4f}, cycle: {train_loss_cycle:.4f}, div: {train_loss_diversity:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (task: {val_loss_task:.4f}, aux: {val_loss_aux:.4f}, cycle: {val_loss_cycle:.4f}, div: {val_loss_diversity:.4f})")
        print(f"  Val Accuracy: {val_accuracy:.4f} (class 0: {val_accuracy_class_0:.4f}, class 1: {val_accuracy_class_1:.4f})")
        print(f"  Class Balance Δ: {class_balance_delta:.4f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Current diversity_weight: {diversity_weight:.4f}")
        print(f"  Current cycle_weight: {cycle_weight:.4f}")

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

        # ===== ADAPTIVE CONTROL INTERVENTIONS =====
        epoch_interventions = []

        if epoch >= config["intervention_start_epoch"] and physics_metrics:
            print(f"\n  {'='*66}")
            print(f"  ADAPTIVE CONTROL ANALYSIS")
            print(f"  {'='*66}")

            # Intervention 1: Boost diversity if q_neural too low
            if physics_metrics['q_neural'] < config['q_neural_threshold']:
                old_diversity = diversity_weight
                diversity_weight = min(
                    diversity_weight + config['diversity_boost'],
                    config['max_diversity_weight']
                )
                if diversity_weight > old_diversity:
                    intervention = {
                        'epoch': epoch + 1,
                        'type': 'diversity_boost',
                        'reason': f"q_neural={physics_metrics['q_neural']:.3f} < {config['q_neural_threshold']}",
                        'old_value': old_diversity,
                        'new_value': diversity_weight
                    }
                    interventions.append(intervention)
                    epoch_interventions.append(intervention)
                    print(f"  🔧 INTERVENTION: Boosted diversity_weight to {diversity_weight:.3f} (was {old_diversity:.3f})")
                    print(f"     Reason: Safety factor q_neural={physics_metrics['q_neural']:.3f} indicates instability")

            # Intervention 2: Boost cycle weight if temperature inverted
            if physics_metrics.get('T_gradient', 0) < config['temp_gradient_threshold']:
                old_cycle = cycle_weight
                cycle_weight = min(
                    cycle_weight + config['cycle_boost'],
                    config['max_cycle_weight']
                )
                if cycle_weight > old_cycle:
                    intervention = {
                        'epoch': epoch + 1,
                        'type': 'cycle_boost',
                        'reason': f"T_gradient={physics_metrics.get('T_gradient', 0):.3f} < {config['temp_gradient_threshold']}",
                        'old_value': old_cycle,
                        'new_value': cycle_weight
                    }
                    interventions.append(intervention)
                    epoch_interventions.append(intervention)
                    print(f"  🔧 INTERVENTION: Boosted cycle_weight to {cycle_weight:.3f} (was {old_cycle:.3f})")
                    print(f"     Reason: Inverted temperature profile detected (gradient={physics_metrics.get('T_gradient', 0):.3f})")

            # Intervention 3: Reduce LR if Q factor too low (after warmup)
            if physics_metrics['Q_factor'] < config['Q_factor_threshold'] and epoch > 3:
                old_lr = optimizer.param_groups[0]['lr']
                new_lr = max(
                    old_lr * config['lr_decay'],
                    config['min_learning_rate']
                )
                if new_lr < old_lr:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    intervention = {
                        'epoch': epoch + 1,
                        'type': 'lr_decay',
                        'reason': f"Q_factor={physics_metrics['Q_factor']:.3f} < {config['Q_factor_threshold']}",
                        'old_value': old_lr,
                        'new_value': new_lr
                    }
                    interventions.append(intervention)
                    epoch_interventions.append(intervention)
                    print(f"  🔧 INTERVENTION: Reduced learning_rate to {new_lr:.6f} (was {old_lr:.6f})")
                    print(f"     Reason: Low Q factor={physics_metrics['Q_factor']:.3f} indicates subignition")

            if not epoch_interventions:
                print(f"  ✓ No interventions needed - training is stable")

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
            "learning_rate": optimizer.param_groups[0]['lr'],
            "diversity_weight": diversity_weight,
            "cycle_weight": cycle_weight,
            "interventions": epoch_interventions
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
    print("FINAL RESULTS & INTERVENTION SUMMARY")
    print("="*70)

    results = {
        "variant_name": "6level_adaptive_control",
        "config": config,
        "epochs_trained": epoch + 1,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "final_metrics": history[-1] if history else {},
        "history": history,
        "interventions": interventions,
        "num_interventions": len(interventions),
        "status": "completed"
    }

    print(f"\nBest Val Accuracy: {best_val_accuracy:.4f}")
    print(f"Final Class Balance Δ: {history[-1]['class_balance_delta']:.4f}")
    print(f"Final Cycle Loss: {history[-1]['val_loss_cycle']:.4f}")
    print(f"\nTotal Interventions: {len(interventions)}")

    # Summarize interventions by type
    if interventions:
        print(f"\n{'='*70}")
        print("INTERVENTION SUMMARY")
        print(f"{'='*70}")

        intervention_types = {}
        for interv in interventions:
            itype = interv['type']
            if itype not in intervention_types:
                intervention_types[itype] = []
            intervention_types[itype].append(interv)

        for itype, intervs in intervention_types.items():
            print(f"\n{itype.upper()} ({len(intervs)} times):")
            for interv in intervs:
                print(f"  Epoch {interv['epoch']}: {interv['old_value']:.6f} → {interv['new_value']:.6f}")
                print(f"    Reason: {interv['reason']}")

    # Comparison to 3-level fusion baseline
    baseline_accuracy = 0.5126
    baseline_balance_delta = 0.2960

    print(f"\n{'='*70}")
    print("COMPARISON TO BASELINE")
    print(f"{'='*70}")
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

    # Assess intervention effectiveness
    print(f"\n{'='*70}")
    print("INTERVENTION EFFECTIVENESS ANALYSIS")
    print(f"{'='*70}")

    if len(interventions) > 0:
        # Check if balance improved after interventions
        pre_intervention_balance = history[0]['class_balance_delta']
        post_intervention_balance = history[-1]['class_balance_delta']

        print(f"\nClass Balance Delta:")
        print(f"  Start: {pre_intervention_balance:.4f}")
        print(f"  End: {post_intervention_balance:.4f}")
        print(f"  Change: {post_intervention_balance - pre_intervention_balance:+.4f}")

        if post_intervention_balance < pre_intervention_balance:
            print(f"  ✓ Interventions helped reduce imbalance")
        else:
            print(f"  ⚠️  Balance worsened despite interventions")

        # Check accuracy trajectory
        mid_accuracy = history[len(history)//2]['val_accuracy'] if len(history) > 1 else history[0]['val_accuracy']
        final_accuracy = history[-1]['val_accuracy']

        print(f"\nAccuracy Trajectory:")
        print(f"  Mid-training: {mid_accuracy:.4f}")
        print(f"  Final: {final_accuracy:.4f}")
        print(f"  Change: {final_accuracy - mid_accuracy:+.4f}")

        if final_accuracy > mid_accuracy:
            print(f"  ✓ Accuracy improved in later epochs")
        else:
            print(f"  ⚠️  Accuracy declined in later epochs")
    else:
        print("\nNo interventions were needed - training was stable throughout")

    # Save results
    output_path = "/tmp/6level_adaptive_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


@app.local_entrypoint()
def main():
    """
    Local entrypoint for running adaptive training.
    """
    print("Launching adaptive training on Modal...")
    results = adaptive_train.remote()

    print("\n" + "="*70)
    print("ADAPTIVE TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Final Balance Δ: {results['final_metrics']['class_balance_delta']:.4f}")
    print(f"Total Interventions: {results['num_interventions']}")

    # Display final physics metrics
    if "physics_metrics" in results['final_metrics']:
        pm = results['final_metrics']['physics_metrics']
        print(f"\nFinal Physics Metrics:")
        print(f"  q_neural: {pm['q_neural']:.3f} [{pm['stability']}]")
        print(f"  Q factor: {pm['Q_factor']:.3f} [{pm['lawson_status']}]")
        print(f"  Alert level: {pm['alert_level']}")
