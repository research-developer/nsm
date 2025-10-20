"""
Tests for R-GCN implementation with confidence weighting.

Tests:
    - Basis decomposition parameter reduction
    - Confidence weighting correctness
    - Gradient flow through all parameters
    - Multi-relation handling
    - Multi-layer stacking with residuals
    - Shape consistency
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from nsm.models.rgcn import (
    ConfidenceWeightedRGCN,
    ConfidenceEstimator,
    HierarchicalRGCN
)


class TestConfidenceWeightedRGCN:
    """Tests for single-layer R-GCN."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph for testing."""
        # 10 nodes, 20 edges, 5 relation types
        x = torch.randn(10, 16)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_type = torch.randint(0, 5, (20,))
        edge_attr = torch.rand(20)  # Confidence [0, 1]

        return x, edge_index, edge_type, edge_attr

    def test_initialization(self):
        """Test R-GCN layer initialization."""
        model = ConfidenceWeightedRGCN(
            in_channels=16,
            out_channels=32,
            num_relations=5,
            num_bases=3
        )

        assert model.in_channels == 16
        assert model.out_channels == 32
        assert model.num_relations == 5
        assert model.num_bases == 3

        # Check parameter shapes
        assert model.basis.shape == (3, 16, 32)  # [num_bases, in, out]
        assert model.att.shape == (5, 3)  # [num_relations, num_bases]
        assert model.root.shape == (16, 32)  # [in, out]

    def test_basis_decomposition_parameter_reduction(self):
        """Verify basis decomposition reduces parameters.

        Standard R-GCN: num_relations × in_channels × out_channels
        Basis R-GCN: num_bases × in_channels × out_channels + num_relations × num_bases

        Expected reduction: ~70% for typical configurations
        """
        in_channels, out_channels = 64, 64
        num_relations = 50  # High diversity (e.g., knowledge graph)
        num_bases = 10

        # Standard R-GCN parameter count
        standard_params = num_relations * in_channels * out_channels

        # Basis R-GCN parameter count
        basis_params = (num_bases * in_channels * out_channels +
                       num_relations * num_bases)

        reduction = 1 - (basis_params / standard_params)

        print(f"\nParameter counts:")
        print(f"  Standard R-GCN: {standard_params:,}")
        print(f"  Basis R-GCN: {basis_params:,}")
        print(f"  Reduction: {reduction:.1%}")

        # Should achieve significant reduction
        assert reduction > 0.6, f"Expected >60% reduction, got {reduction:.1%}"

        # Verify actual model parameters
        model = ConfidenceWeightedRGCN(
            in_channels, out_channels, num_relations, num_bases
        )

        # Count parameters (excluding bias and root)
        actual_basis_params = (
            model.basis.numel() +  # num_bases × in × out
            model.att.numel()      # num_relations × num_bases
        )

        assert actual_basis_params == basis_params

    def test_forward_pass_shapes(self, simple_graph):
        """Test forward pass produces correct output shapes."""
        x, edge_index, edge_type, edge_attr = simple_graph

        model = ConfidenceWeightedRGCN(
            in_channels=16,
            out_channels=32,
            num_relations=5,
            num_bases=3
        )

        out = model(x, edge_index, edge_type, edge_attr)

        # Output should have correct shape
        assert out.shape == (10, 32)  # [num_nodes, out_channels]

        # Output should not contain NaN or Inf
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_confidence_weighting(self):
        """Verify messages are weighted by confidence values."""
        # Create simple graph: node 0 receives messages from nodes 1 and 2
        # Use different node features to amplify confidence effect
        x = torch.randn(3, 8)
        x[1] = torch.ones(8) * 2.0  # Node 1 has strong signal
        x[2] = torch.ones(8) * 0.5  # Node 2 has weak signal

        edge_index = torch.tensor([[1, 2], [0, 0]])  # 1->0, 2->0
        edge_type = torch.tensor([0, 0])  # Same relation type

        # Case 1: High confidence for strong signal (node 1)
        edge_attr_high = torch.tensor([1.0, 0.1])

        # Case 2: High confidence for weak signal (node 2)
        edge_attr_low = torch.tensor([0.1, 1.0])

        # Use 'add' aggregation to see raw weighted messages
        model = ConfidenceWeightedRGCN(
            in_channels=8,
            out_channels=8,
            num_relations=1,
            num_bases=1,
            aggr='add'  # 'add' preserves confidence weighting better than 'mean'
        )
        model.eval()

        with torch.no_grad():
            out_high = model(x, edge_index, edge_type, edge_attr_high)
            out_low = model(x, edge_index, edge_type, edge_attr_low)

        # Outputs should be different due to different confidence weighting
        # Node 0 receives different weighted messages
        diff_node0 = (out_high[0] - out_low[0]).abs().sum()

        # With different confidences and different source features, output must differ
        assert diff_node0 > 1e-4, f"Confidence weighting not working: diff={diff_node0:.6f}"

    def test_no_confidence_defaults_to_one(self, simple_graph):
        """Test that missing edge_attr defaults to confidence 1.0."""
        x, edge_index, edge_type, _ = simple_graph

        model = ConfidenceWeightedRGCN(
            in_channels=16,
            out_channels=16,
            num_relations=5,
            num_bases=3
        )
        model.eval()

        with torch.no_grad():
            # Without edge_attr
            out_no_conf = model(x, edge_index, edge_type)

            # With all-ones edge_attr
            edge_attr_ones = torch.ones(edge_index.size(1))
            out_with_ones = model(x, edge_index, edge_type, edge_attr_ones)

        # Outputs should be identical
        assert torch.allclose(out_no_conf, out_with_ones, atol=1e-5)

    def test_gradient_flow(self, simple_graph):
        """Verify gradients flow to all parameters."""
        x, edge_index, edge_type, edge_attr = simple_graph

        model = ConfidenceWeightedRGCN(
            in_channels=16,
            out_channels=16,
            num_relations=5,
            num_bases=3
        )

        # Forward + backward
        out = model(x, edge_index, edge_type, edge_attr)
        loss = out.mean()
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 1e-6, f"Vanishing gradient for {name}"

        print("\nGradient norms:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.grad.norm().item():.6f}")

    def test_multiple_relation_types(self):
        """Test handling of heterogeneous relation types."""
        # Graph with 3 different relation types
        x = torch.randn(5, 8)
        edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 4]
        ])
        edge_type = torch.tensor([0, 1, 2, 1])  # Different relations
        edge_attr = torch.rand(4)

        model = ConfidenceWeightedRGCN(
            in_channels=8,
            out_channels=8,
            num_relations=3,
            num_bases=2
        )

        out = model(x, edge_index, edge_type, edge_attr)

        assert out.shape == (5, 8)
        assert not torch.isnan(out).any()

    def test_self_loops_added(self):
        """Verify self-loops are added automatically."""
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1], [1, 2]])  # 2 edges
        edge_type = torch.tensor([0, 0])

        model = ConfidenceWeightedRGCN(
            in_channels=8,
            out_channels=8,
            num_relations=2,  # Includes self-loop relation
            num_bases=2
        )

        # Test by checking that forward pass works correctly
        out = model(x, edge_index, edge_type)

        assert out.shape == (5, 8), "Output shape incorrect"

        # Self-loops should allow nodes without incoming edges to have non-zero output
        # Node 0 has no incoming edges except self-loop
        assert out[0].abs().sum() > 0, "Node with only self-loop should have non-zero output"

        # All nodes should have non-zero output due to self-loops and root transformation
        for i in range(5):
            assert out[i].abs().sum() > 0, f"Node {i} output is zero (self-loops not working)"


class TestConfidenceEstimator:
    """Tests for learned confidence estimation."""

    def test_initialization(self):
        """Test confidence estimator initialization."""
        estimator = ConfidenceEstimator(
            node_dim=64,
            num_relations=10,
            hidden_dim=32
        )

        assert estimator.node_dim == 64
        assert estimator.num_relations == 10

    def test_confidence_estimation(self):
        """Test confidence estimation from node features."""
        estimator = ConfidenceEstimator(
            node_dim=16,
            num_relations=5,
            hidden_dim=32
        )

        x = torch.randn(10, 16)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_type = torch.randint(0, 5, (20,))

        confidence = estimator(x, edge_index, edge_type)

        # Check shape and range
        assert confidence.shape == (20,)
        assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_different_features_different_confidence(self):
        """Test that different node features produce different confidences."""
        estimator = ConfidenceEstimator(
            node_dim=8,
            num_relations=2,
            hidden_dim=16
        )
        estimator.eval()

        # Two nodes with different features
        x1 = torch.zeros(2, 8)
        x1[0] = torch.ones(8)

        x2 = torch.ones(2, 8)

        edge_index = torch.tensor([[0], [1]])  # Edge 0->1
        edge_type = torch.tensor([0])

        with torch.no_grad():
            conf1 = estimator(x1, edge_index, edge_type)
            conf2 = estimator(x2, edge_index, edge_type)

        # Should produce different confidences
        assert not torch.allclose(conf1, conf2, atol=1e-3)


class TestHierarchicalRGCN:
    """Tests for multi-layer R-GCN stack."""

    @pytest.fixture
    def graph_data(self):
        """Create graph data for testing."""
        x = torch.randn(20, 32)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_type = torch.randint(0, 8, (50,))
        edge_attr = torch.rand(50)

        return x, edge_index, edge_type, edge_attr

    def test_initialization(self):
        """Test multi-layer R-GCN initialization."""
        model = HierarchicalRGCN(
            in_channels=32,
            hidden_channels=64,
            out_channels=32,
            num_relations=8,
            num_layers=3,
            num_bases=4
        )

        assert model.num_layers == 3
        assert len(model.layers) == 3
        assert len(model.norms) == 3

    def test_forward_pass(self, graph_data):
        """Test forward pass through multiple layers."""
        x, edge_index, edge_type, edge_attr = graph_data

        model = HierarchicalRGCN(
            in_channels=32,
            hidden_channels=64,
            out_channels=32,
            num_relations=8,
            num_layers=2,
            num_bases=4
        )

        out = model(x, edge_index, edge_type, edge_attr)

        assert out.shape == (20, 32)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_residual_connections(self, graph_data):
        """Verify residual connections help gradient flow."""
        x, edge_index, edge_type, edge_attr = graph_data

        model = HierarchicalRGCN(
            in_channels=32,
            hidden_channels=32,
            out_channels=32,
            num_relations=8,
            num_layers=4,  # Deep network
            num_bases=4
        )

        out = model(x, edge_index, edge_type, edge_attr)
        loss = out.mean()
        loss.backward()

        # Check gradients in first and last layer
        first_layer_grad = model.layers[0].basis.grad.norm()
        last_layer_grad = model.layers[-1].basis.grad.norm()

        print(f"\nGradient norms in deep network:")
        print(f"  First layer: {first_layer_grad:.6f}")
        print(f"  Last layer: {last_layer_grad:.6f}")

        # Gradients should not vanish (>1e-6)
        assert first_layer_grad > 1e-6, "Vanishing gradient in first layer"
        assert last_layer_grad > 1e-6, "Vanishing gradient in last layer"

        # Ratio should not be extreme (residuals help)
        ratio = max(first_layer_grad, last_layer_grad) / min(first_layer_grad, last_layer_grad)
        assert ratio < 100, f"Gradient ratio too extreme: {ratio:.1f}"

    def test_layer_normalization(self, graph_data):
        """Test that layer normalization stabilizes activations."""
        x, edge_index, edge_type, edge_attr = graph_data

        model = HierarchicalRGCN(
            in_channels=32,
            hidden_channels=64,
            out_channels=32,
            num_relations=8,
            num_layers=3,
            num_bases=4
        )

        model.eval()

        with torch.no_grad():
            out = model(x, edge_index, edge_type, edge_attr)

        # Activations should be reasonably scaled
        assert out.mean().abs() < 10, "Activations not normalized"
        assert out.std() < 10, "Activation variance too high"

    def test_dropout_training_vs_eval(self, graph_data):
        """Test dropout behavior in training vs eval mode."""
        x, edge_index, edge_type, edge_attr = graph_data

        model = HierarchicalRGCN(
            in_channels=32,
            hidden_channels=64,
            out_channels=32,
            num_relations=8,
            num_layers=2,
            num_bases=4,
            dropout=0.5  # High dropout for testing
        )

        # Training mode: should have stochasticity
        model.train()
        out1_train = model(x, edge_index, edge_type, edge_attr)
        out2_train = model(x, edge_index, edge_type, edge_attr)

        # Eval mode: should be deterministic
        model.eval()
        with torch.no_grad():
            out1_eval = model(x, edge_index, edge_type, edge_attr)
            out2_eval = model(x, edge_index, edge_type, edge_attr)

        # Training outputs should differ (dropout randomness)
        assert not torch.allclose(out1_train, out2_train, atol=1e-5)

        # Eval outputs should be identical (no dropout)
        assert torch.allclose(out1_eval, out2_eval, atol=1e-6)


class TestIntegration:
    """Integration tests combining R-GCN with confidence estimation."""

    def test_rgcn_with_estimated_confidence(self):
        """Test R-GCN using learned confidence estimates."""
        # Create graph
        x = torch.randn(15, 32)
        edge_index = torch.randint(0, 15, (2, 40))
        edge_type = torch.randint(0, 6, (40,))

        # Estimate confidence
        estimator = ConfidenceEstimator(
            node_dim=32,
            num_relations=6,
            hidden_dim=32
        )
        confidence = estimator(x, edge_index, edge_type)

        # Use estimated confidence in R-GCN
        rgcn = ConfidenceWeightedRGCN(
            in_channels=32,
            out_channels=32,
            num_relations=6,
            num_bases=4
        )

        out = rgcn(x, edge_index, edge_type, confidence)

        assert out.shape == (15, 32)
        assert not torch.isnan(out).any()

    def test_end_to_end_training(self):
        """Test end-to-end training with R-GCN and confidence estimation."""
        # Create synthetic task: predict node labels
        num_nodes = 30
        x = torch.randn(num_nodes, 16)
        edge_index = torch.randint(0, num_nodes, (2, 80))
        edge_type = torch.randint(0, 4, (80,))
        labels = torch.randint(0, 2, (num_nodes,))

        # Build model
        estimator = ConfidenceEstimator(16, num_relations=4, hidden_dim=32)
        rgcn = HierarchicalRGCN(
            in_channels=16,
            hidden_channels=32,
            out_channels=2,  # Binary classification
            num_relations=4,
            num_layers=2,
            num_bases=3
        )

        optimizer = torch.optim.Adam(
            list(estimator.parameters()) + list(rgcn.parameters()),
            lr=0.01
        )

        # Training loop
        initial_loss = None
        final_loss = None

        for epoch in range(10):
            optimizer.zero_grad()

            # Estimate confidence
            confidence = estimator(x, edge_index, edge_type)

            # R-GCN forward
            logits = rgcn(x, edge_index, edge_type, confidence)

            # Loss
            loss = nn.functional.cross_entropy(logits, labels)

            if epoch == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            if epoch == 9:
                final_loss = loss.item()

        print(f"\nEnd-to-end training:")
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Reduction: {(1 - final_loss/initial_loss)*100:.1f}%")

        # Loss should decrease
        assert final_loss < initial_loss, "Training did not improve loss"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
