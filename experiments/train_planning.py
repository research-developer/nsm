"""
Training script for NSM-22: Planning Domain with Goal-Action Hierarchy.

Domain-specific implementation:
- 16 relations (requires, enables, achieves, precedes, etc.)
- 8 bases for R-GCN (50% parameter reduction)
- Pool ratio: 0.5 (50% nodes at abstract level - strong hierarchy)
- Goal achievement classification
- Temporal ordering constraints

Target metrics:
- Reconstruction error: <20%
- Goal achievement accuracy: ≥85%
- Temporal ordering accuracy: ≥80%
- Plan validity: ≥75%

Usage:
    python experiments/train_planning.py --epochs 100 --batch-size 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader
import argparse
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from nsm.data.planning_dataset import PlanningTripleDataset
from nsm.models import NSMModel
from nsm.training import NSMTrainer, compute_classification_metrics
from nsm.models.confidence.temperature import TemperatureScheduler
from nsm.evaluation.planning_metrics import (
    compute_goal_achievement_accuracy,
    compute_temporal_ordering_accuracy,
    compute_plan_validity
)


def compute_planning_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    task_type: str,
    dataset: PlanningTripleDataset = None
) -> dict:
    """Compute planning domain-specific metrics.

    Args:
        preds: Predicted logits
        labels: Ground truth labels
        task_type: Task type
        dataset: PlanningTripleDataset for hierarchical evaluation

    Returns:
        dict: Metrics including goal achievement, temporal ordering
    """
    metrics = compute_classification_metrics(preds, labels, task_type)

    # Domain-specific metrics
    if dataset is not None:
        # Goal achievement accuracy
        goal_acc = compute_goal_achievement_accuracy(preds, labels, dataset)
        metrics['goal_achievement'] = goal_acc

        # Temporal ordering accuracy
        temporal_acc = compute_temporal_ordering_accuracy(preds, labels, dataset)
        metrics['temporal_ordering'] = temporal_acc

        # Plan validity (all preconditions satisfied)
        plan_validity = compute_plan_validity(preds, labels, dataset)
        metrics['plan_validity'] = plan_validity

    return metrics


def create_planning_model(
    node_features: int,
    num_classes: int,
    device: torch.device
) -> NSMModel:
    """Create NSM model configured for planning domain.

    Configuration (from NSM-22):
    - 16 relations (requires, enables, achieves, precedes, etc.)
    - 8 bases (50% parameter reduction)
    - pool_ratio=0.5 (strong goal-action hierarchy)
    - ProductSemiring for confidence (multiplicative along precondition chains)

    Args:
        node_features: Node feature dimensionality
        num_classes: Number of output classes (e.g., goal achieved/failed)
        device: Device

    Returns:
        NSMModel: Configured model
    """
    model = NSMModel(
        node_features=node_features,
        num_relations=16,  # Domain-specific relation count
        num_classes=num_classes,
        num_bases=8,  # 50% parameter reduction
        pool_ratio=0.5,  # Strong hierarchy (goals vs actions)
        task_type='classification',
        num_levels=3  # Phase 1.5: 3-level hierarchy to break symmetry bias
    )

    return model.to(device)


def collate_fn(batch_list):
    """Collate function for PlanningTripleDataset.

    PyG Data objects need special handling for batching.
    """
    from torch_geometric.data import Batch

    # Batch PyG Data objects
    data_list = [item[0] for item in batch_list]
    labels = torch.tensor([item[1] for item in batch_list])

    batched_data = Batch.from_data_list(data_list)

    # Create batch dict
    batch = {
        'x': batched_data.x,
        'edge_index': batched_data.edge_index,
        'edge_type': batched_data.edge_type,
        'edge_attr': batched_data.edge_attr if hasattr(batched_data, 'edge_attr') else None,
        'batch': batched_data.batch,
        'y': labels
    }

    return batch


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    print("Loading planning dataset...")
    dataset = PlanningTripleDataset(
        root=args.data_dir,
        split='train',
        num_problems=args.num_plans,  # Fixed: use num_problems parameter
        seed=args.seed
    )

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train: {train_size} plans, Val: {val_size} plans")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    # Model
    print("Creating planning NSM model...")
    model = create_planning_model(
        node_features=args.node_features,
        num_classes=args.num_classes,
        device=device
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Temperature scheduler
    temp_scheduler = TemperatureScheduler(
        initial_temp=1.0,
        final_temp=0.3,
        decay_rate=0.9999,
        warmup_epochs=10
    )

    # Trainer
    print("Initializing trainer...")
    trainer = NSMTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        cycle_loss_weight=args.cycle_loss_weight,
        gradient_clip=args.gradient_clip,
        temp_scheduler=temp_scheduler,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard
    )

    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Checkpoint directory: {args.checkpoint_dir}")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        task_type='classification',
        compute_metrics=lambda p, l, t: compute_planning_metrics(p, l, t, dataset),
        early_stopping_patience=args.early_stopping_patience,
        save_best_only=True
    )

    # Save final results
    results = {
        'args': vars(args),
        'final_train_loss': history['train'][-1]['total_loss'],
        'final_val_loss': history['val'][-1]['total_loss'],
        'best_val_loss': trainer.best_val_loss,
        'final_metrics': history['val'][-1]
    }

    results_path = Path(args.checkpoint_dir) / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final metrics:")
    for key, value in history['val'][-1].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    print(f"\nResults saved to: {results_path}")
    print(f"Best model: {args.checkpoint_dir}/best_model.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NSM on Planning Domain')

    # Data
    parser.add_argument('--data-dir', type=str, default='data/planning',
                        help='Data directory')
    parser.add_argument('--num-plans', type=int, default=1000,
                        help='Number of planning scenarios')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Data loader workers')

    # Model
    parser.add_argument('--node-features', type=int, default=64,
                        help='Node feature dimensionality')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of output classes')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--cycle-loss-weight', type=float, default=0.1,
                        help='Weight for cycle consistency loss')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--early-stopping-patience', type=int, default=20,
                        help='Early stopping patience')

    # Logging
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/planning',
                        help='Checkpoint directory')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging interval (steps)')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases')
    parser.add_argument('--use-tensorboard', action='store_true',
                        help='Use Tensorboard')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    main(args)
