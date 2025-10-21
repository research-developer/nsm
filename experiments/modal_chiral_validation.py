"""
Modal GPU validation script for chiral architecture - ATTENTION variant.

Tests attention-based hinge exchange mechanism.

Usage:
    modal run experiments/modal_chiral_validation.py::validate_attention
"""

import modal
import sys
from pathlib import Path

# Modal app configuration
app = modal.App("nsm-chiral-attention-validation")

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
def validate_attention():
    """
    Validate attention-based chiral architecture.
    """
    import json
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Batch
    from datetime import datetime
    from tqdm import tqdm

    # Add NSM to path
    sys.path.insert(0, "/root/NSM")

    from nsm.models.chiral import MinimalChiralModel
    from nsm.data.planning_dataset import PlanningTripleDataset

    print("="*60)
    print("CHIRAL ARCHITECTURE VALIDATION - ATTENTION VARIANT")
    print("="*60)

    # Configuration
    config = {
        "variant": "attention",
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "seed": 42,
        "cycle_weight": 0.01,
        "patience": 20,
        "pool_ratio": 0.5,
        "num_heads": 8,
        "dropout": 0.1
    }

    torch.manual_seed(config["seed"])

    # Load dataset
    print("\nLoading Planning dataset...")
    # Pre-generate all graphs as a list to avoid indexing issues
    full_dataset = PlanningTripleDataset(root="/tmp/planning", split="train", num_problems=4100)

    # Materialize all graphs into a list
    all_graphs = [full_dataset[i] for i in range(len(full_dataset))]

    # Split into train/val
    train_size = 2000
    train_graphs = all_graphs[:train_size]
    val_graphs = all_graphs[train_size:]

    # Create DataLoaders with explicit collate function
    def pyg_collate(data_list):
        return Batch.from_data_list(data_list)

    train_loader = DataLoader(train_graphs, batch_size=config["batch_size"], shuffle=True, collate_fn=pyg_collate)
    val_loader = DataLoader(val_graphs, batch_size=config["batch_size"], shuffle=False, collate_fn=pyg_collate)

    print(f"Train samples: {len(train_graphs)}")
    print(f"Val samples: {len(val_graphs)}")

    # Get data properties from first batch
    sample = next(iter(train_loader))
    node_features = sample.x.size(1)
    num_relations = int(sample.edge_type.max().item()) + 1
    num_classes = 2

    print(f"\nDataset properties:")
    print(f"  Node features: {node_features}")
    print(f"  Num relations: {num_relations}")
    print(f"  Num classes: {num_classes}")

    # Initialize model
    print("\nInitializing MinimalChiralModel (attention-based)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MinimalChiralModel(
        node_features=node_features,
        num_relations=num_relations,
        num_classes=num_classes,
        pool_ratio=config["pool_ratio"],
        task_type='classification'
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    history = []

    for epoch in range(config["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        train_task_loss = 0.0
        train_cycle_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"):
            batch = batch.to(device)

            # Forward pass
            output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)

            # Task loss
            task_loss = F.cross_entropy(output['logits'], batch.y)

            # Cycle loss
            cycle_loss = output['cycle_loss']

            # Total loss
            loss = task_loss + config["cycle_weight"] * cycle_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_task_loss += task_loss.item()
            train_cycle_loss += cycle_loss.item()

        train_loss /= len(train_loader)
        train_task_loss /= len(train_loader)
        train_cycle_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        val_task_loss = 0.0
        val_cycle_loss = 0.0
        correct_total = 0
        correct_class_0 = 0
        correct_class_1 = 0
        total_class_0 = 0
        total_class_1 = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]"):
                batch = batch.to(device)

                # Forward pass
                output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)

                # Task loss
                task_loss = F.cross_entropy(output['logits'], batch.y)

                # Cycle loss
                cycle_loss = output['cycle_loss']

                # Total loss
                loss = task_loss + config["cycle_weight"] * cycle_loss

                val_loss += loss.item()
                val_task_loss += task_loss.item()
                val_cycle_loss += cycle_loss.item()

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
        val_task_loss /= len(val_loader)
        val_cycle_loss /= len(val_loader)
        val_accuracy = correct_total / total
        val_accuracy_class_0 = correct_class_0 / total_class_0 if total_class_0 > 0 else 0
        val_accuracy_class_1 = correct_class_1 / total_class_1 if total_class_1 > 0 else 0
        class_balance_delta = abs(val_accuracy_class_0 - val_accuracy_class_1)

        # Log
        print(f"\nEpoch {epoch+1}/{config['epochs']}:")
        print(f"  Train Loss: {train_loss:.4f} (task: {train_task_loss:.4f}, cycle: {train_cycle_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (task: {val_task_loss:.4f}, cycle: {val_cycle_loss:.4f})")
        print(f"  Val Accuracy: {val_accuracy:.4f} (class 0: {val_accuracy_class_0:.4f}, class 1: {val_accuracy_class_1:.4f})")
        print(f"  Class Balance Δ: {class_balance_delta:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_accuracy_class_0": val_accuracy_class_0,
            "val_accuracy_class_1": val_accuracy_class_1,
            "class_balance_delta": class_balance_delta
        })

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  ✓ New best accuracy: {best_val_accuracy:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n  Early stopping triggered (patience={config['patience']})")
                break

    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    results = {
        "variant_name": "chiral_attention",
        "config": config,
        "epochs_trained": epoch + 1,
        "training_time_seconds": None,  # TODO: track time
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "final_metrics": history[-1] if history else {},
        "history": history,
        "status": "completed"
    }

    print(f"\nBest Val Accuracy: {best_val_accuracy:.4f}")
    print(f"Final Class Balance Δ: {history[-1]['class_balance_delta']:.4f}")
    print(f"Final Cycle Loss: {val_cycle_loss:.4f}")

    # Comparison to baseline
    baseline_accuracy = 0.433
    baseline_balance_delta = 0.953

    print(f"\nComparison to baseline:")
    print(f"  Accuracy: {best_val_accuracy:.4f} vs {baseline_accuracy:.4f} (Δ {best_val_accuracy - baseline_accuracy:+.4f})")
    print(f"  Balance Δ: {history[-1]['class_balance_delta']:.4f} vs {baseline_balance_delta:.4f} (Δ {history[-1]['class_balance_delta'] - baseline_balance_delta:+.4f})")

    if best_val_accuracy >= 0.50 and history[-1]['class_balance_delta'] < 0.50:
        print("\n✅ SUCCESS: Passed primary criteria (accuracy ≥50%, balance Δ <50%)")
    else:
        print("\n❌ FAILED: Did not meet primary criteria")

    # Save results
    output_path = "/tmp/chiral_attention_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


@app.local_entrypoint()
def main():
    """
    Local entrypoint for running validation.
    """
    print("Launching attention-based chiral validation on Modal...")
    results = validate_attention.remote()

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print(f"\nFinal Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Final Balance Δ: {results['final_metrics']['class_balance_delta']:.4f}")
