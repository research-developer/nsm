"""
Modal validation: Adaptive physics-based training control.

Tests if physics-informed hyperparameter adaptation outperforms fixed baseline.

Control strategy:
- Monitor q_neural, temperature profile, Q factor
- Dynamically adjust diversity_weight, cycle_weight, learning_rate
- Compare to NSM-32 baseline (fixed hyperparams)

Usage:
    modal run experiments/modal_adaptive_validation.py::validate_adaptive
"""

import modal
import sys
from pathlib import Path

app = modal.App("nsm-adaptive-physics")
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy<2",
        "torch==2.1.0",
        "torch-geometric==2.4.0",
        "tqdm",
    )
    .run_commands(
        "pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html"
    )
    .add_local_dir(PROJECT_ROOT, "/root/NSM", copy=True, ignore=["*.pyc", "__pycache__", ".git", "logs", "checkpoints", "data", ".pytest_cache"])
)

volume = modal.Volume.from_name("nsm-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/checkpoints": volume}
)
def validate_adaptive():
    """Validate adaptive physics-based training."""
    import json
    import torch
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch
    from tqdm import tqdm

    sys.path.insert(0, "/root/NSM")

    from nsm.models.chiral import FullChiralModel
    from nsm.training.chiral_loss import ChiralCompositeLoss
    from nsm.training.physics_metrics import compute_all_physics_metrics
    from nsm.training.adaptive_physics_trainer import AdaptivePhysicsConfig, AdaptivePhysicsTrainer
    from nsm.data.planning_dataset import PlanningTripleDataset

    print("="*70)
    print("ADAPTIVE PHYSICS CONTROL VALIDATION - NSM-33 Track B")
    print("="*70)

    config = {
        "variant": "adaptive_physics",
        "epochs": 10,
        "batch_size": 64,
        "seed": 42,
        "pool_ratio": 0.5,
        "dropout": 0.1,
        "patience": 20,
    }

    torch.manual_seed(config["seed"])

    # Load dataset
    print("\nLoading Planning dataset...")
    full_dataset = PlanningTripleDataset(root="/tmp/planning", split="train", num_problems=4100)
    all_graphs = [full_dataset[i] for i in range(len(full_dataset))]

    train_size = 2000
    train_graphs = all_graphs[:train_size]
    val_graphs = all_graphs[train_size:]

    def pyg_collate(data_list):
        graphs = [item[0] for item in data_list]
        labels = torch.tensor([item[1] for item in data_list])
        batch = Batch.from_data_list(graphs)
        batch.y = labels
        return batch

    train_loader = DataLoader(train_graphs, batch_size=config["batch_size"], shuffle=True, collate_fn=pyg_collate)
    val_loader = DataLoader(val_graphs, batch_size=config["batch_size"], shuffle=False, collate_fn=pyg_collate)

    sample = next(iter(train_loader))
    node_features = sample.x.size(1)
    num_relations = int(sample.edge_type.max().item()) + 1
    num_classes = 2

    print(f"\nDataset: {node_features} features, {num_relations} relations, {num_classes} classes")

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullChiralModel(
        node_features=node_features,
        num_relations=num_relations,
        num_classes=num_classes,
        pool_ratio=config["pool_ratio"],
        task_type='classification',
        dropout=config["dropout"]
    ).to(device)

    # Initialize loss and optimizer
    criterion = ChiralCompositeLoss(
        task_weight=1.0,
        aux_weight=0.3,
        cycle_weight=0.01,  # Will be adapted
        diversity_weight=0.0,  # Will be adapted
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Initialize adaptive controller
    adaptive_config = AdaptivePhysicsConfig(
        initial_diversity_weight=0.0,
        initial_cycle_weight=0.01,
        initial_learning_rate=1e-4,
        q_unstable_threshold=1.0,
        q_critical_threshold=0.5,
        temp_inversion_threshold=-0.1,
        Q_factor_threshold=0.5,
        diversity_increment=0.05,
        cycle_increment=0.02,
        lr_decay_factor=0.9,
        check_every_n_epochs=1,
        cooldown_epochs=2
    )

    adaptive_trainer = AdaptivePhysicsTrainer(adaptive_config, optimizer, criterion)

    print("\n" + "="*70)
    print("TRAINING WITH ADAPTIVE PHYSICS CONTROL")
    print("="*70)

    history = []
    best_val_accuracy = 0.0

    for epoch in range(config["epochs"]):
        # Train
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            loss_dict = criterion(output, batch.y)

            optimizer.zero_grad()
            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss_dict['loss'].item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        correct_total = 0
        correct_class_0 = 0
        correct_class_1 = 0
        total_class_0 = 0
        total_class_1 = 0
        total = 0

        all_level_reps_l1 = []
        all_level_reps_l2 = []
        all_level_reps_l3 = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                batch = batch.to(device)
                output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)

                if 'x_l1' in output:
                    all_level_reps_l1.append(output['x_l1'].cpu())
                if 'x_l2' in output:
                    all_level_reps_l2.append(output['x_l2'].cpu())
                if 'x_l3' in output:
                    all_level_reps_l3.append(output['x_l3'].cpu())

                loss_dict = criterion(output, batch.y)
                val_loss += loss_dict['loss'].item()

                pred = output['logits'].argmax(dim=1)
                correct_total += (pred == batch.y).sum().item()
                total += batch.y.size(0)

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
        val_accuracy = correct_total / total
        val_accuracy_class_0 = correct_class_0 / total_class_0 if total_class_0 > 0 else 0
        val_accuracy_class_1 = correct_class_1 / total_class_1 if total_class_1 > 0 else 0
        class_balance_delta = abs(val_accuracy_class_0 - val_accuracy_class_1)

        # Compute physics metrics
        class_accs = {
            'accuracy_class_0': val_accuracy_class_0,
            'accuracy_class_1': val_accuracy_class_1
        }

        level_reps = {}
        if all_level_reps_l1:
            level_reps['L1'] = torch.cat(all_level_reps_l1, dim=0)
        if all_level_reps_l2:
            level_reps['L2'] = torch.cat(all_level_reps_l2, dim=0)
        if all_level_reps_l3:
            level_reps['L3'] = torch.cat(all_level_reps_l3, dim=0)

        physics_metrics = compute_all_physics_metrics(
            model=model,
            class_accuracies=class_accs,
            level_representations=level_reps,
            epoch=epoch + 1,
            task_complexity=1.0
        )

        # ADAPTIVE CONTROL: Adjust hyperparameters based on physics
        adaptation = adaptive_trainer.analyze_and_adapt(epoch + 1, physics_metrics)

        # Log
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}")
        print(f"{'='*70}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"  Class 0: {val_accuracy_class_0:.4f}, Class 1: {val_accuracy_class_1:.4f}, Δ: {class_balance_delta:.4f}")
        print(f"\n  Physics Metrics:")
        print(f"    q_neural: {physics_metrics['q_neural']:.3f} [{physics_metrics['stability']}]")
        print(f"    T_gradient: {physics_metrics.get('T_gradient', 0.0):.3f} [{physics_metrics.get('profile_type', 'unknown')}]")
        print(f"    Q factor: {physics_metrics['Q_factor']:.3f}")

        if adaptation['adapted']:
            print(f"\n  🎛️  ADAPTATION TRIGGERED:")
            for intervention in adaptation['interventions']:
                print(f"    {intervention}")
            hyperparams = adaptation['new_hyperparams']
            print(f"    New hyperparams: diversity={hyperparams['diversity_weight']:.3f}, cycle={hyperparams['cycle_weight']:.3f}, LR={hyperparams['learning_rate']:.4e}")
        else:
            print(f"\n  Status: No adaptation (reason: {adaptation.get('reason', 'N/A')})")

        if physics_metrics['warnings']:
            for warning in physics_metrics['warnings']:
                print(f"  {warning}")

        history.append({
            "epoch": epoch + 1,
            "val_accuracy": val_accuracy,
            "class_balance_delta": class_balance_delta,
            "physics_metrics": physics_metrics,
            "adaptation": adaptation
        })

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"\n  ✓ New best accuracy: {best_val_accuracy:.4f}")

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Best Val Accuracy: {best_val_accuracy:.4f}")
    print(f"Final Balance Δ: {history[-1]['class_balance_delta']:.4f}")

    intervention_summary = adaptive_trainer.get_intervention_summary()
    print(f"\nTotal Adaptations: {intervention_summary['total_interventions']}")

    baseline_accuracy = 0.4816
    print(f"\nComparison to fixed hyperparams baseline:")
    print(f"  Adaptive: {best_val_accuracy:.4f}")
    print(f"  Baseline: {baseline_accuracy:.4f}")
    print(f"  Improvement: {best_val_accuracy - baseline_accuracy:+.4f} ({(best_val_accuracy - baseline_accuracy)/baseline_accuracy*100:+.2f}%)")

    results = {
        "variant_name": "adaptive_physics_control",
        "config": config,
        "best_val_accuracy": best_val_accuracy,
        "history": history,
        "intervention_summary": intervention_summary
    }

    with open("/tmp/adaptive_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


@app.local_entrypoint()
def main():
    print("Launching adaptive physics validation...")
    results = validate_adaptive.remote()
    print(f"\nFinal Accuracy: {results['best_val_accuracy']:.4f}")
