"""
R-GCN Example: Confidence-Weighted Message Passing

Demonstrates R-GCN usage with synthetic semantic triple data:
1. Basic ConfidenceWeightedRGCN usage
2. Multi-layer HierarchicalRGCN
3. Learned confidence estimation
4. Training on node classification task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from nsm.models.rgcn import (
    ConfidenceWeightedRGCN,
    ConfidenceEstimator,
    HierarchicalRGCN
)


def example_1_basic_rgcn():
    """Example 1: Basic R-GCN layer."""
    print("=" * 60)
    print("Example 1: Basic Confidence-Weighted R-GCN")
    print("=" * 60)

    # Create synthetic graph
    num_nodes = 50
    num_edges = 150
    num_relations = 8  # Different predicate types

    x = torch.randn(num_nodes, 32)  # Node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))
    edge_attr = torch.rand(num_edges)  # Confidence [0, 1]

    # Create R-GCN layer
    rgcn = ConfidenceWeightedRGCN(
        in_channels=32,
        out_channels=64,
        num_relations=num_relations,
        num_bases=4  # Parameter reduction via basis decomposition
    )

    # Forward pass
    out = rgcn(x, edge_index, edge_type, edge_attr)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Number of relations: {num_relations}")
    print(f"Number of bases: {rgcn.num_bases}")

    # Calculate parameter reduction
    standard_params = num_relations * 32 * 64
    basis_params = rgcn.basis.numel() + rgcn.att.numel()
    reduction = 1 - (basis_params / standard_params)

    print(f"\nParameter Efficiency:")
    print(f"  Standard R-GCN: {standard_params:,} params")
    print(f"  Basis R-GCN: {basis_params:,} params")
    print(f"  Reduction: {reduction:.1%}")
    print()


def example_2_hierarchical_rgcn():
    """Example 2: Multi-layer R-GCN with residuals."""
    print("=" * 60)
    print("Example 2: Hierarchical R-GCN (Multi-Layer)")
    print("=" * 60)

    # Create graph
    num_nodes = 100
    num_edges = 300
    num_relations = 16

    x = torch.randn(num_nodes, 64)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))
    edge_attr = torch.rand(num_edges)

    # Multi-layer R-GCN
    model = HierarchicalRGCN(
        in_channels=64,
        hidden_channels=128,
        out_channels=64,
        num_relations=num_relations,
        num_layers=3,
        num_bases=8,
        dropout=0.1
    )

    # Forward pass
    out = model(x, edge_index, edge_type, edge_attr)

    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Layers: {model.num_layers}")
    print(f"Architecture:")
    print(f"  64 → 128 (layer 1)")
    print(f"  128 → 128 (layer 2)")
    print(f"  128 → 128 (layer 3)")
    print(f"  128 → 64 (output projection)")
    print(f"\nResidual connections + LayerNorm prevent vanishing gradients")
    print()


def example_3_confidence_estimation():
    """Example 3: Learning confidence from node features."""
    print("=" * 60)
    print("Example 3: Learned Confidence Estimation")
    print("=" * 60)

    # Create graph (no edge confidence provided)
    num_nodes = 75
    num_edges = 200
    num_relations = 10

    x = torch.randn(num_nodes, 48)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))

    # Confidence estimator
    estimator = ConfidenceEstimator(
        node_dim=48,
        num_relations=num_relations,
        hidden_dim=64
    )

    # Estimate confidence from node features
    confidence = estimator(x, edge_index, edge_type)

    print(f"Edges: {num_edges}")
    print(f"Estimated confidence shape: {confidence.shape}")
    print(f"Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"Mean confidence: {confidence.mean():.3f}")
    print(f"\nConfidence is learned from:")
    print(f"  - Source node features")
    print(f"  - Target node features")
    print(f"  - Edge type (predicate)")

    # Use estimated confidence in R-GCN
    rgcn = ConfidenceWeightedRGCN(
        in_channels=48,
        out_channels=48,
        num_relations=num_relations,
        num_bases=6
    )

    out = rgcn(x, edge_index, edge_type, confidence)
    print(f"\nR-GCN output shape: {out.shape}")
    print()


def example_4_node_classification():
    """Example 4: Training on node classification task."""
    print("=" * 60)
    print("Example 4: Node Classification Training")
    print("=" * 60)

    # Create synthetic task
    num_nodes = 200
    num_edges = 600
    num_relations = 12
    num_classes = 4

    x = torch.randn(num_nodes, 32)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))
    labels = torch.randint(0, num_classes, (num_nodes,))

    # Train/val split
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:150] = True
    val_mask = ~train_mask

    # Model
    class RGCNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.confidence_estimator = ConfidenceEstimator(
                node_dim=32,
                num_relations=num_relations,
                hidden_dim=64
            )
            self.rgcn = HierarchicalRGCN(
                in_channels=32,
                hidden_channels=64,
                out_channels=num_classes,
                num_relations=num_relations,
                num_layers=2,
                num_bases=6
            )

        def forward(self, x, edge_index, edge_type):
            confidence = self.confidence_estimator(x, edge_index, edge_type)
            logits = self.rgcn(x, edge_index, edge_type, confidence)
            return logits

    model = RGCNClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    print("Training...")
    losses = []

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()

        logits = model(x, edge_index, edge_type)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index, edge_type)
                val_loss = F.cross_entropy(logits[val_mask], labels[val_mask])
                train_acc = (logits[train_mask].argmax(dim=1) == labels[train_mask]).float().mean()
                val_acc = (logits[val_mask].argmax(dim=1) == labels[val_mask]).float().mean()

            print(f"Epoch {epoch+1:2d}: "
                  f"Train Loss={loss.item():.4f}, Val Loss={val_loss:.4f}, "
                  f"Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

    print(f"\nFinal training loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"Loss reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
    print()


def example_5_domain_specific():
    """Example 5: Domain-specific configurations."""
    print("=" * 60)
    print("Example 5: Domain-Specific Configurations")
    print("=" * 60)

    # Planning Domain (16 relations, strong hierarchy)
    print("Planning Domain Configuration:")
    planning_rgcn = HierarchicalRGCN(
        in_channels=64,
        hidden_channels=128,
        out_channels=64,
        num_relations=16,
        num_layers=2,
        num_bases=8
    )
    print(f"  Relations: 16")
    print(f"  Bases: 8 (50% of relations)")
    print(f"  Layers: 2 (moderate depth)")
    print()

    # Knowledge Graph Domain (66 relations, high diversity)
    print("Knowledge Graph Configuration:")
    kg_rgcn = HierarchicalRGCN(
        in_channels=64,
        hidden_channels=128,
        out_channels=64,
        num_relations=66,
        num_layers=2,
        num_bases=12
    )
    print(f"  Relations: 66")
    print(f"  Bases: 12 (18% of relations, high compression)")
    print(f"  Layers: 2")
    print()

    # Causal Domain (20 relations, epistemic confidence)
    print("Causal Domain Configuration:")
    causal_rgcn = HierarchicalRGCN(
        in_channels=64,
        hidden_channels=128,
        out_channels=64,
        num_relations=20,
        num_layers=2,
        num_bases=5
    )
    print(f"  Relations: 20")
    print(f"  Bases: 5 (25% of relations)")
    print(f"  Layers: 2")
    print(f"  Note: Confidence represents causal strength")
    print()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("R-GCN with Confidence Weighting - Examples")
    print("="*60 + "\n")

    example_1_basic_rgcn()
    example_2_hierarchical_rgcn()
    example_3_confidence_estimation()
    example_4_node_classification()
    example_5_domain_specific()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
