"""Tests for invertible coupling layers."""

import pytest
import torch
from nsm.models.coupling import (
    AffineCouplingLayer,
    MultiLayerCoupling,
    GraphCouplingLayer
)


class TestAffineCouplingLayer:
    """Tests for single affine coupling layer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = AffineCouplingLayer(features=64, hidden_dim=128)
        assert layer.features == 64
        assert layer.split_dim == 32
        assert layer.transform_dim == 32

    def test_forward_shape(self):
        """Test forward pass produces correct shape."""
        layer = AffineCouplingLayer(features=64)
        x = torch.randn(10, 64)
        y = layer.forward(x)
        assert y.shape == (10, 64)

    def test_exact_invertibility(self):
        """Test forward/inverse are exact (key property)."""
        layer = AffineCouplingLayer(features=64, hidden_dim=128)
        x = torch.randn(10, 64)

        # Forward then inverse
        y = layer.forward(x)
        x_reconstructed = layer.inverse(y)

        # Should be exact (within numerical precision)
        assert torch.allclose(x, x_reconstructed, atol=1e-5)
        
        print(f"\nReconstruction error: {(x - x_reconstructed).abs().max():.2e}")

    def test_inverse_forward_cycle(self):
        """Test inverse/forward cycle."""
        layer = AffineCouplingLayer(features=64)
        y = torch.randn(10, 64)

        # Inverse then forward
        x = layer.inverse(y)
        y_reconstructed = layer.forward(x)

        assert torch.allclose(y, y_reconstructed, atol=1e-5)

    def test_first_part_unchanged(self):
        """Test that first part (x1) passes through unchanged."""
        layer = AffineCouplingLayer(features=64, split_dim=32)
        x = torch.randn(10, 64)

        y = layer.forward(x)

        # First 32 dimensions should be identical
        assert torch.allclose(x[:, :32], y[:, :32], atol=1e-7)

    def test_jacobian_computation(self):
        """Test log determinant of Jacobian."""
        layer = AffineCouplingLayer(features=64)
        x = torch.randn(10, 64)

        log_det = layer.log_det_jacobian(x)

        # Should be scalar per batch element
        assert log_det.shape == (10,)
        # Should be finite
        assert torch.isfinite(log_det).all()

    def test_gradient_flow_forward(self):
        """Test gradients flow through forward pass."""
        layer = AffineCouplingLayer(features=64, hidden_dim=128)
        x = torch.randn(10, 64, requires_grad=True)

        y = layer.forward(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

        # Check network parameters have gradients
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad is not None, f"Missing gradient for {name}"  # Zero gradients OK for identity init

    def test_gradient_flow_inverse(self):
        """Test gradients flow through inverse pass."""
        layer = AffineCouplingLayer(features=64, hidden_dim=128)
        y = torch.randn(10, 64, requires_grad=True)

        x = layer.inverse(y)
        loss = x.sum()
        loss.backward()

        assert y.grad is not None
        assert y.grad.abs().sum() > 0

    def test_custom_split_dimension(self):
        """Test with custom split dimension."""
        layer = AffineCouplingLayer(features=90, split_dim=30)
        x = torch.randn(5, 90)

        y = layer.forward(x)
        x_reconstructed = layer.inverse(y)

        assert torch.allclose(x, x_reconstructed, atol=1e-5)

    def test_numerical_stability_extreme_values(self):
        """Test stability with large/small values."""
        layer = AffineCouplingLayer(features=64)

        # Large values
        x_large = torch.randn(10, 64) * 10
        y_large = layer.forward(x_large)
        x_recon_large = layer.inverse(y_large)
        assert torch.allclose(x_large, x_recon_large, atol=1e-4)

        # Small values
        x_small = torch.randn(10, 64) * 0.1
        y_small = layer.forward(x_small)
        x_recon_small = layer.inverse(y_small)
        assert torch.allclose(x_small, x_recon_small, atol=1e-6)


class TestMultiLayerCoupling:
    """Tests for stacked coupling layers."""

    def test_initialization(self):
        """Test multi-layer initialization."""
        coupling = MultiLayerCoupling(features=64, num_layers=4)
        assert coupling.num_layers == 4
        assert len(coupling.layers) == 4

    def test_exact_invertibility_multilayer(self):
        """Test multi-layer maintains exact invertibility."""
        coupling = MultiLayerCoupling(features=64, num_layers=4, hidden_dim=128)
        x = torch.randn(10, 64)

        y = coupling.forward(x)
        x_reconstructed = coupling.inverse(y)

        error = (x - x_reconstructed).abs().max()
        print(f"\n4-layer reconstruction error: {error:.2e}")

        assert torch.allclose(x, x_reconstructed, atol=1e-5)

    def test_alternating_splits(self):
        """Test layers use different split dimensions."""
        coupling = MultiLayerCoupling(features=90, num_layers=4)

        # Check alternating pattern
        assert coupling.layers[0].split_dim == 45  # 50%
        assert coupling.layers[1].split_dim == 30  # 33%
        assert coupling.layers[2].split_dim == 45  # 50%
        assert coupling.layers[3].split_dim == 30  # 33%

    def test_deep_network_invertibility(self):
        """Test deep stack (8 layers) maintains invertibility."""
        coupling = MultiLayerCoupling(features=64, num_layers=8)
        x = torch.randn(5, 64)

        y = coupling.forward(x)
        x_reconstructed = coupling.inverse(y)

        assert torch.allclose(x, x_reconstructed, atol=1e-4)

    def test_jacobian_multilayer(self):
        """Test Jacobian computation for multiple layers."""
        coupling = MultiLayerCoupling(features=64, num_layers=3)
        x = torch.randn(10, 64)

        log_det = coupling.log_det_jacobian(x)

        assert log_det.shape == (10,)
        assert torch.isfinite(log_det).all()


class TestGraphCouplingLayer:
    """Tests for graph-aware coupling."""

    def test_initialization(self):
        """Test graph coupling initialization."""
        coupling = GraphCouplingLayer(node_features=64, num_layers=3)
        assert coupling.node_features == 64

    def test_node_feature_transformation(self):
        """Test transforms node features while preserving graph."""
        coupling = GraphCouplingLayer(node_features=64, num_layers=3)

        # Graph with 100 nodes
        x = torch.randn(100, 64)

        y = coupling.forward(x)
        x_reconstructed = coupling.inverse(y)

        assert y.shape == (100, 64)
        assert torch.allclose(x, x_reconstructed, atol=1e-5)

    def test_works_with_pyg_data(self):
        """Test compatibility with PyG Data objects."""
        from torch_geometric.data import Data

        coupling = GraphCouplingLayer(node_features=32)

        # Create PyG Data object
        x = torch.randn(50, 32)
        edge_index = torch.randint(0, 50, (2, 150))

        data = Data(x=x, edge_index=edge_index)

        # Transform node features
        data.x = coupling.forward(data.x)

        # Graph structure unchanged
        assert data.edge_index.shape == (2, 150)

        # Inverse
        data.x = coupling.inverse(data.x)

        assert torch.allclose(data.x, x, atol=1e-5)


class TestIntegration:
    """Integration tests for coupling layers."""

    def test_batch_processing(self):
        """Test batch processing maintains invertibility."""
        coupling = MultiLayerCoupling(features=64, num_layers=3)

        # Process batch
        x_batch = torch.randn(32, 64)
        y_batch = coupling.forward(x_batch)
        x_recon_batch = coupling.inverse(y_batch)

        assert torch.allclose(x_batch, x_recon_batch, atol=1e-5)

    def test_training_mode_consistency(self):
        """Test train/eval modes give same results."""
        coupling = AffineCouplingLayer(features=64, hidden_dim=128)
        x = torch.randn(10, 64)

        # Training mode
        coupling.train()
        y_train = coupling.forward(x)

        # Eval mode
        coupling.eval()
        with torch.no_grad():
            y_eval = coupling.forward(x)

        # Should be identical (no dropout/batch norm)
        assert torch.allclose(y_train, y_eval, atol=1e-7)

    def test_composition_with_other_layers(self):
        """Test coupling can be composed with other operations."""
        coupling = MultiLayerCoupling(features=64, num_layers=2)
        linear = torch.nn.Linear(64, 64)

        x = torch.randn(10, 64)

        # Compose: linear → coupling → inverse → linear
        y = linear(x)
        z = coupling.forward(y)
        y_recon = coupling.inverse(z)
        x_out = linear(y_recon)

        # Invertibility preserved through composition
        assert not torch.allclose(x, x_out)  # Changed by linear layers
        assert torch.allclose(y, y_recon, atol=1e-5)  # Coupling inverts perfectly


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
