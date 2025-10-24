#!/usr/bin/env python3
"""
Demo: How to use the 24K planning dataset for 10-fold cross-validation.

Shows:
1. Basic usage with 24K problems
2. 10-fold cross-validation setup
3. Analyzing problems by tier
4. Integration with PyTorch DataLoader
"""

import sys
import os
import tempfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nsm.data.planning_dataset import PlanningTripleDataset


def demo_basic_usage():
    """Demo 1: Basic usage."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Usage")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 24K dataset
        dataset = PlanningTripleDataset(
            root=tmpdir,
            split='train',
            num_problems=24000,
            problems_per_split=True,
            seed=42
        )

        print(f"\nDataset created: {len(dataset)} problems")

        # Access a problem
        graph, label = dataset[0]
        print(f"\nExample problem (idx=0):")
        print(f"  Nodes: {graph.num_nodes}")
        print(f"  Edges: {graph.edge_index.size(1)}")
        print(f"  Label: {label.item()} ({'valid' if label.item() == 1 else 'invalid'})")

        # Get triples for the problem
        triples = dataset.get_problem_triples(0)
        print(f"  Triples: {len(triples)}")
        print(f"  Tier: {triples[0].metadata.get('tier', 'unknown')}")

        # Show tier distribution
        tier_counts = Counter()
        for i in range(0, 1000, 10):  # Sample first 1000
            triples = dataset.get_problem_triples(i)
            tier = triples[0].metadata.get('tier', -1) if triples else -1
            tier_counts[tier] += 1

        print(f"\nTier distribution (sample of 100):")
        for tier in sorted(tier_counts.keys()):
            print(f"  Tier {tier}: {tier_counts[tier]} problems")


def demo_10fold_cv():
    """Demo 2: 10-fold cross-validation."""
    print("\n" + "=" * 80)
    print("DEMO 2: 10-Fold Cross-Validation Setup")
    print("=" * 80)

    # In real usage, you would generate the full dataset once and reuse it
    print("\n# Pseudo-code for 10-fold CV:")
    print("""
from sklearn.model_selection import KFold
from nsm.data.planning_dataset import PlanningTripleDataset

# Generate 24K dataset
dataset = PlanningTripleDataset(
    root='data/planning_24k',
    split='train',
    num_problems=24000,
    problems_per_split=True,
    seed=42
)

# Setup 10-fold CV
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Train each fold
for fold, (train_idx, val_idx) in enumerate(kfold.split(range(24000))):
    print(f"Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")

    # Create train/val subsets
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    # Train model on this fold
    model = YourModel()
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader)
        val_loss = validate(model, val_loader)
        print(f"  Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

    # Save fold results
    results[fold] = evaluate(model, val_loader)

# Aggregate results across folds
avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
print(f"Average accuracy: {avg_accuracy:.4f}")
    """)

    # Show what each fold looks like
    print("\nFold sizes (24K / 10 = 2,400 per fold):")
    for fold in range(10):
        train_size = 24000 - 2400  # 9 folds
        val_size = 2400  # 1 fold
        print(f"  Fold {fold}: train={train_size}, val={val_size}")


def demo_tier_analysis():
    """Demo 3: Analyzing problems by tier."""
    print("\n" + "=" * 80)
    print("DEMO 3: Tier-Specific Analysis")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = PlanningTripleDataset(
            root=tmpdir,
            split='train',
            num_problems=1000,  # Smaller for demo
            problems_per_split=True,
            seed=42
        )

        print(f"\nAnalyzing {len(dataset)} problems by tier...")

        tier_stats = {0: [], 1: [], 2: []}

        for i in range(len(dataset)):
            triples = dataset.get_problem_triples(i)
            tier = triples[0].metadata.get('tier', -1) if triples else -1

            if tier >= 0:
                graph, label = dataset[i]
                tier_stats[tier].append({
                    'nodes': graph.num_nodes,
                    'edges': graph.edge_index.size(1),
                    'triples': len(triples),
                    'label': label.item()
                })

        print("\nStatistics by tier:")
        for tier in sorted(tier_stats.keys()):
            stats = tier_stats[tier]
            if stats:
                avg_nodes = sum(s['nodes'] for s in stats) / len(stats)
                avg_edges = sum(s['edges'] for s in stats) / len(stats)
                valid_pct = sum(s['label'] for s in stats) / len(stats) * 100

                print(f"  Tier {tier} ({len(stats)} problems):")
                print(f"    Avg nodes:  {avg_nodes:5.1f}")
                print(f"    Avg edges:  {avg_edges:5.1f}")
                print(f"    Valid:      {valid_pct:5.1f}%")


def demo_dataloader():
    """Demo 4: PyTorch DataLoader integration."""
    print("\n" + "=" * 80)
    print("DEMO 4: PyTorch DataLoader Integration")
    print("=" * 80)

    print("\n# Example DataLoader usage:")
    print("""
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

def collate_fn(batch):
    '''Custom collate for PyG graphs.'''
    graphs = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return Batch.from_data_list(graphs), labels

# Create dataset
dataset = PlanningTripleDataset(
    root='data/planning_24k',
    split='train',
    num_problems=24000,
    problems_per_split=True,
    seed=42
)

# Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

# Training loop
for batch_idx, (batch_graph, batch_labels) in enumerate(loader):
    # batch_graph: PyG Batch object
    # batch_labels: [batch_size] tensor

    # Forward pass
    output = model(batch_graph)
    loss = criterion(output, batch_labels)

    # Backward pass
    loss.backward()
    optimizer.step()

    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx}: loss={loss.item():.4f}")
    """)


def main():
    """Run all demos."""
    print("=" * 80)
    print("24K PLANNING DATASET - USAGE DEMONSTRATIONS")
    print("=" * 80)

    demo_basic_usage()
    demo_10fold_cv()
    demo_tier_analysis()
    demo_dataloader()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key Takeaways:

1. Basic Usage:
   - Use problems_per_split=True for 24K problems
   - Access problems via dataset[idx]
   - Get triples via get_problem_triples(idx)

2. 10-Fold CV:
   - Use sklearn.model_selection.KFold
   - Each fold: 21.6K train, 2.4K val
   - Aggregate results across folds

3. Tier Analysis:
   - Problems automatically assigned to tiers
   - Analyze performance by complexity
   - Metadata available in triples

4. DataLoader:
   - Use custom collate_fn for PyG
   - Batch graphs with Batch.from_data_list
   - Standard PyTorch training loop

Ready for 10x validation experiments!
    """)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
