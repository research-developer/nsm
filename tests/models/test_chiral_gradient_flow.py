"""
Test gradient flow through chiral architecture (PR #17 Fix #1 validation).

Critical fix: Verify gradients propagate through all 6 levels after replacing
in-place tensor assignments with scatter_ operations.
"""

import pytest
import torch
from nsm.models.chiral import FullChiralModel


class TestChiralGradientFlow:
    """Test gradient flow through the 6-level chiral architecture."""

    @pytest.fixture
    def model_config(self):
        """Standard test configuration."""
        return {
            'node_features': 32,
            'num_relations': 5,
            'num_classes': 3,
            'num_bases': 2,
            'pool_ratio': 0.5,
            'task_type': 'classification',
            'dropout': 0.1
        }

    @pytest.fixture
    def sample_batch(self):
        """Create small test batch."""
        num_nodes = 20
        num_edges = 30

        x = torch.randn(num_nodes, 32, requires_grad=True)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_type = torch.randint(0, 5, (num_edges,))
        batch = torch.zeros(num_nodes, dtype=torch.long)

        return x, edge_index, edge_type, batch

    def test_gradient_flow_through_all_levels(self, model_config, sample_batch):
        """
        FIX #1 VALIDATION: Verify gradients propagate through all 6 levels.

        Before fix: In-place assignments (x[perm] = y) broke gradient graph
        After fix: scatter_ maintains computational graph
        """
        model = FullChiralModel(**model_config)
        x, edge_index, edge_type, batch = sample_batch

        # Forward pass
        outputs = model(x, edge_index, edge_type, batch)

        # Compute combined loss (classification + cycle losses)
        loss = outputs['logits'].mean()  # Classification loss proxy
        loss += outputs['cycle_loss_upper']
        loss += outputs['cycle_loss_lower']
        loss += outputs['cycle_loss_cross']

        # Backward pass
        loss.backward()

        # CRITICAL TEST: Verify gradients exist at all levels
        level_representations = [
            outputs['x_l1'],
            outputs['x_l2'],
            outputs['x_l3'],
            outputs['x_l4'],
            outputs['x_l5'],
            outputs['x_l6']
        ]

        for level_idx, x_level in enumerate(level_representations, 1):
            # Check gradient exists
            assert x_level.grad is not None, \
                f"Level {level_idx} has no gradient (gradient graph broken)"

            # Check gradient is non-zero (information flows)
            grad_norm = x_level.grad.norm().item()
            assert grad_norm > 1e-6, \
                f"Level {level_idx} has vanishing gradient: {grad_norm:.2e}"

    def test_scatter_maintains_gradient_connectivity(self, model_config, sample_batch):
        """
        Verify scatter_ operations maintain gradient connectivity.

        This tests the specific fix for lines 715-718 in chiral.py.
        """
        model = FullChiralModel(**model_config)
        x, edge_index, edge_type, batch = sample_batch

        # Forward pass
        outputs = model(x, edge_index, edge_type, batch)

        # Test cycle loss gradient path
        cycle_loss = outputs['cycle_loss_upper']
        cycle_loss.backward()

        # Verify L3 refined has gradient (flows through scatter_)
        assert outputs['x_l3'].grad is not None
        assert outputs['x_l3'].grad.abs().sum().item() > 0

    def test_numerical_stability_in_normalization(self, model_config):
        """
        FIX #3 VALIDATION: Verify epsilon additive prevents division issues.

        Test edge case: all same values (zero scale)
        """
        model = FullChiralModel(**model_config)

        # Create constant features (zero scale case)
        x = torch.ones(10, 32) * 0.5  # All same value
        edge_index = torch.tensor([[0, 1], [1, 0]])
        edge_type = torch.tensor([0, 0])

        # Should not crash or produce NaN
        outputs = model(x, edge_index, edge_type)

        # Check no NaN in normalized features
        assert not torch.isnan(outputs['logits']).any()
        assert not torch.isnan(outputs['x_l1']).any()
        assert not torch.isnan(outputs['x_l2']).any()
        assert not torch.isnan(outputs['x_l3']).any()

    def test_input_validation_in_hinge_exchange(self, model_config, sample_batch):
        """
        FIX #4 VALIDATION: Verify shape mismatch assertion works.

        This tests the input validation added to ChiralHingeExchange.forward()
        """
        model = FullChiralModel(**model_config)

        # Access hinge exchange directly
        hinge = model.hinge_l2_l5

        # Test matching shapes (should work)
        x_upper = torch.randn(5, 32)
        x_lower = torch.randn(5, 32)
        x_upper_ref, x_lower_ref = hinge(x_upper, x_lower)
        assert x_upper_ref.shape == x_upper.shape

        # Test mismatched shapes (should raise assertion)
        x_upper = torch.randn(5, 32)
        x_lower = torch.randn(3, 32)  # Different size
        with pytest.raises(AssertionError, match="Shape mismatch"):
            hinge(x_upper, x_lower)

    def test_proper_unpooling_vs_nearest_neighbor(self, model_config):
        """
        FIX #2 VALIDATION: Verify proper unpooling uses perm_large correctly.

        Before fix: Lossy nearest neighbor interpolation
        After fix: Proper unpooling using perm_large indices
        """
        model = FullChiralModel(**model_config)

        # Create test tensors
        x_small = torch.randn(5, 32)
        x_large_template = torch.randn(10, 32)
        perm_large = torch.tensor([0, 2, 4, 6, 8])  # Indices of selected nodes

        # Call _align_sizes (uses proper unpooling now)
        x_aligned = model._align_sizes(x_small, x_large_template, perm_large)

        # Verify: values should be at specified perm_large positions
        for i, perm_idx in enumerate(perm_large):
            assert torch.allclose(x_aligned[perm_idx], x_small[i]), \
                f"Unpooling failed: position {perm_idx} does not match x_small[{i}]"

        # Verify: other positions should be zeros
        mask = torch.zeros(10, dtype=torch.bool)
        mask[perm_large] = True
        zero_positions = ~mask
        assert torch.allclose(x_aligned[zero_positions], torch.zeros_like(x_aligned[zero_positions])), \
            "Non-selected positions should be zero"

    def test_end_to_end_gradient_magnitude(self, model_config, sample_batch):
        """
        Comprehensive test: Verify gradients have reasonable magnitude.

        Combines all fixes to ensure no gradient vanishing or explosion.
        """
        model = FullChiralModel(**model_config)
        x, edge_index, edge_type, batch = sample_batch

        # Forward pass
        outputs = model(x, edge_index, edge_type, batch)

        # Combined loss
        loss = outputs['logits'].mean()
        loss += outputs['cycle_loss_upper']
        loss += outputs['cycle_loss_lower']
        loss += outputs['cycle_loss_cross']

        # Backward pass
        loss.backward()

        # Check all model parameters have gradients
        params_without_grad = []
        params_with_low_grad = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    params_without_grad.append(name)
                else:
                    grad_norm = param.grad.norm().item()
                    if grad_norm < 1e-8:
                        params_with_low_grad.append((name, grad_norm))

        # Report any issues
        assert len(params_without_grad) == 0, \
            f"Parameters without gradient: {params_without_grad}"

        assert len(params_with_low_grad) == 0, \
            f"Parameters with vanishing gradients: {params_with_low_grad}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
