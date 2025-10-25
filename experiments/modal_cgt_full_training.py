"""
CGT Full Training with Checkpoint Integration (NSM-34).

Trains NSM models with CGT operator tracking for 15 epochs (NSM-33 standard).
Can optionally load pre-trained NSM-33 checkpoints as initialization.

This replaces the 5-epoch minimal training with production-ready validation.

Usage:
    # Train from scratch with CGT tracking
    modal run experiments/modal_cgt_full_training.py::train_from_scratch

    # Load NSM-33 checkpoint and continue with CGT tracking
    modal run experiments/modal_cgt_full_training.py::train_from_checkpoint --checkpoint=nsm-10x-baseline_best.pt

    # Track existing NSM-33 model without additional training
    modal run experiments/modal_cgt_full_training.py::track_checkpoint --checkpoint=nsm-10x-baseline_best.pt
"""

import modal
from pathlib import Path
from typing import Optional

app = modal.App("nsm-cgt-full-training")
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Use same image as NSM-33 for compatibility
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
    .add_local_dir(PROJECT_ROOT, "/root/NSM", copy=True, ignore=["*.pyc", "__pycache__", ".git", "logs", "data", ".pytest_cache"])
)

# Shared volume with NSM-33 checkpoints
volume = modal.Volume.from_name("nsm-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,  # 2 hours
    volumes={"/checkpoints": volume}
)
def train_nsm_with_cgt_tracking(
    epochs: int = 15,
    checkpoint_path: Optional[str] = None,
    dataset: str = "planning",
    num_problems: int = 2000,
    batch_size: int = 64,
    seed: int = 42
):
    """
    Train NSM model with full CGT operator tracking.

    Args:
        epochs: Number of training epochs (default: 15 like NSM-33)
        checkpoint_path: Optional path to pre-trained checkpoint in /checkpoints/
        dataset: Dataset type (planning, kg, causal)
        num_problems: Number of problems to train on
        batch_size: Batch size
        seed: Random seed
    """
    import json
    import sys
    import torch
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch
    from tqdm import tqdm
    from datetime import datetime

    sys.path.insert(0, "/root/NSM")

    from nsm.models.chiral import FullChiralModel
    from nsm.training.chiral_loss import ChiralCompositeLoss
    from nsm.data.planning_dataset import PlanningTripleDataset
    from nsm.training.cgt_metrics import temperature_conway, CoolingMonitor

    print("="*80)
    print("NSM-34 CGT FULL TRAINING")
    print("="*80)
    print(f"Epochs: {epochs}")
    print(f"Dataset: {dataset} (N={num_problems})")
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
    print("="*80)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print(f"\n📊 Loading {dataset} dataset...")
    full_dataset = PlanningTripleDataset(root=f"/tmp/{dataset}", split="train", num_problems=num_problems)
    all_graphs = [full_dataset[i] for i in range(len(full_dataset))]

    train_size = int(0.8 * len(all_graphs))
    train_graphs = all_graphs[:train_size]
    val_graphs = all_graphs[train_size:]

    def pyg_collate(data_list):
        graphs = [item[0] for item in data_list]
        labels = torch.tensor([item[1] for item in data_list])
        batch = Batch.from_data_list(graphs)
        batch.y = labels
        return batch

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, collate_fn=pyg_collate)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, collate_fn=pyg_collate)

    print(f"   Train: {len(train_graphs)} | Val: {len(val_graphs)}")

    # Initialize model
    sample = next(iter(train_loader))
    node_features = sample.x.size(1)
    num_relations = int(sample.edge_type.max().item()) + 1
    num_classes = 2

    model = FullChiralModel(
        node_features=node_features,
        num_relations=num_relations,
        num_classes=num_classes,
        pool_ratio=0.5,
        task_type='classification',
        dropout=0.1
    ).to(device)

    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path:
        full_path = Path("/checkpoints") / checkpoint_path
        if full_path.exists():
            checkpoint = torch.load(full_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"✅ Loaded checkpoint from epoch {start_epoch}")
        else:
            print(f"⚠️  Checkpoint not found: {full_path}, training from scratch")

    criterion = ChiralCompositeLoss(task_weight=1.0, aux_weight=0.3, cycle_weight=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Initialize CGT tracking
    cooling_monitor = CoolingMonitor(window_size=5)

    print("\n🚀 Starting training with CGT tracking...\n")

    history = []
    best_val_accuracy = 0.0

    # Special case: If epochs == 0 or start_epoch, just evaluate and track CGT
    if epochs == 0 or epochs == start_epoch:
        print("\n📊 Tracking-only mode (no training, just CGT evaluation)...\n")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = batch.to(device)
                output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                loss_dict = criterion(output, batch.y)

                val_loss += loss_dict['loss'].item()
                pred = output['logits'].argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct / total

        # CGT Operator Tracking
        print(f"\n📐 Computing CGT operators on loaded checkpoint...\n")

        with torch.no_grad():
            val_batch = next(iter(val_loader)).to(device)
            x_sample = val_batch.x

            temp, temp_diag = temperature_conway(model, x_sample, num_samples=20, metric='mse')

            print(f"   Conway Temperature: {temp:.4f}")
            if temp < 0.01:
                print(f"   ⚠️  Near-zero temperature")
            elif temp < 0.2:
                print(f"   ⚠️  Low temperature (collapse risk zone)")
            else:
                print(f"   ✅ Healthy temperature")

        # Save single evaluation result
        epoch_data = {
            "epoch": start_epoch,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "cgt_temperature": temp,
            "cgt_cooling_rate": 0.0
        }
        history.append(epoch_data)
        best_val_accuracy = val_accuracy  # For summary section

        print(f"\n{'='*80}")
        print(f"CGT TRACKING COMPLETE")
        print(f"{'='*80}")
        print(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"  CGT Temperature: {temp:.4f}")

    else:
        # Normal training loop
        for epoch in range(start_epoch, epochs):
            # Training
            model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                batch = batch.to(device)
                output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                loss_dict = criterion(output, batch.y)

                optimizer.zero_grad()
                loss_dict['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss_dict['loss'].item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    batch = batch.to(device)
                    output = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                    loss_dict = criterion(output, batch.y)

                    val_loss += loss_dict['loss'].item()
                    pred = output['logits'].argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)

            val_loss /= len(val_loader)
            val_accuracy = correct / total

            # CGT Operator Tracking
            print(f"\n📐 Epoch {epoch+1}/{epochs} - Computing CGT operators...")

            with torch.no_grad():
                # Sample a validation batch
                val_batch = next(iter(val_loader)).to(device)
                x_sample = val_batch.x

                # Conway temperature
                temp, temp_diag = temperature_conway(model, x_sample, num_samples=20, metric='mse')

                print(f"   Conway Temperature: {temp:.4f}")
                if temp < 0.01:
                    print(f"   ⚠️  Near-zero temperature (EXPECTED early in training)")
                elif temp < 0.2:
                    print(f"   ⚠️  Low temperature (collapse risk zone)")
                else:
                    print(f"   ✅ Healthy temperature")

                # Note: Cooling rate tracking requires hinge parameters (α/β)
                # FullChiralModel uses hinge layers, but we'd need to extract them
                # For now, just track Conway temperature
                cooling_rate = None

            # Log results
            epoch_data = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "cgt_temperature": temp,
                "cgt_cooling_rate": cooling_rate if cooling_rate is not None else 0.0
            }
            history.append(epoch_data)

            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*80}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            print(f"  CGT Temperature: {temp:.4f}")

            # Save checkpoint
            is_best = val_accuracy > best_val_accuracy
            if is_best:
                best_val_accuracy = val_accuracy
                print(f"  🌟 New best accuracy: {best_val_accuracy:.4f}")

                # Save best checkpoint directly
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': {"val_accuracy": val_accuracy, "cgt_temperature": temp},
                    'config': {"epochs": epochs, "dataset": dataset, "num_problems": num_problems},
                    'timestamp': timestamp
                }

                best_path = f"/checkpoints/nsm-cgt-{dataset}_best.pt"
                torch.save(checkpoint, best_path)
                print(f"  💾 Saved best checkpoint: {best_path}")

    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Val Accuracy: {best_val_accuracy:.4f}")
    print(f"Final CGT Temperature: {history[-1]['cgt_temperature']:.4f}")

    # Save results
    results = {
        "experiment": "nsm-34-cgt-full-training",
        "dataset": dataset,
        "epochs": epochs,
        "best_val_accuracy": best_val_accuracy,
        "history": history
    }

    results_path = f"/checkpoints/nsm-cgt-{dataset}-{datetime.now().strftime('%Y%m%d_%H%M%S')}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    volume.commit()

    return results


@app.local_entrypoint()
def train_from_scratch(epochs: int = 15):
    """Train from scratch with CGT tracking."""
    print(f"🚀 Training from scratch ({epochs} epochs)...")
    results = train_nsm_with_cgt_tracking.remote(epochs=epochs)
    print(f"\n✅ Final accuracy: {results['best_val_accuracy']:.4f}")


@app.local_entrypoint()
def train_from_checkpoint(checkpoint: str, epochs: int = 15):
    """Continue training from NSM-33 checkpoint."""
    print(f"🚀 Loading checkpoint: {checkpoint}")
    results = train_nsm_with_cgt_tracking.remote(epochs=epochs, checkpoint_path=checkpoint)
    print(f"\n✅ Final accuracy: {results['best_val_accuracy']:.4f}")


@app.local_entrypoint()
def track_checkpoint(checkpoint: str):
    """Track CGT operators on existing checkpoint (no training)."""
    print(f"📊 Tracking CGT operators on: {checkpoint}")
    # Just evaluate, no training
    results = train_nsm_with_cgt_tracking.remote(epochs=0, checkpoint_path=checkpoint)
    print(f"\n✅ CGT Temperature: {results['history'][0]['cgt_temperature']:.4f}")
