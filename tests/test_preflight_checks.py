"""
Unit tests for NSM-31 preflight check system.

Tests that preflight checks correctly identify problematic configurations
that led to NSM-31 training failures.
"""

import pytest
import torch
from torch.utils.data import TensorDataset

from nsm.evaluation.preflight_checks import (
    run_preflight_checks,
    check_dataset_balance,
    check_cycle_loss_weight,
    check_learning_rate,
    check_pyg_extensions,
    PreflightCheckError,
    PreflightCheckWarning
)


class TestDatasetBalance:
    """Test dataset balance checks (prevent class collapse)."""

    def test_balanced_dataset_passes(self):
        """Perfectly balanced dataset should pass."""
        # Create balanced dataset: 50/50 split
        data = torch.randn(100, 10)
        labels = torch.cat([torch.zeros(50), torch.ones(50)])
        dataset = TensorDataset(data, labels)

        result = check_dataset_balance(dataset, max_samples=100)

        assert result['is_balanced']
        assert result['minority_proportion'] == 0.5

    def test_imbalanced_dataset_fails(self):
        """Severely imbalanced dataset should raise error."""
        # Create imbalanced dataset: 90/10 split (minority 10%)
        data = torch.randn(100, 10)
        labels = torch.cat([torch.zeros(90), torch.ones(10)])
        dataset = TensorDataset(data, labels)

        with pytest.raises(PreflightCheckError, match="Severe class imbalance"):
            check_dataset_balance(dataset, max_samples=100)

    def test_moderately_imbalanced_warns(self):
        """Moderately imbalanced dataset should warn."""
        # Create moderate imbalance: 60/40 split
        data = torch.randn(100, 10)
        labels = torch.cat([torch.zeros(60), torch.ones(40)])
        dataset = TensorDataset(data, labels)

        with pytest.warns(PreflightCheckWarning, match="Moderate class imbalance"):
            result = check_dataset_balance(dataset, max_samples=100)
            assert not result['is_balanced']


class TestCycleLossWeight:
    """Test cycle loss weight validation (prevent gradient dominance)."""

    def test_recommended_weight_passes(self):
        """NSM-31 recommended weight (0.01) should pass."""
        result = check_cycle_loss_weight(0.01)

        assert result['is_safe']
        assert not result['is_critical']

    def test_high_weight_warns(self):
        """Weight above recommended but below critical should warn."""
        with pytest.warns(PreflightCheckWarning, match="exceeds recommended"):
            result = check_cycle_loss_weight(0.07)
            assert not result['is_safe']
            assert not result['is_critical']

    def test_critical_weight_fails(self):
        """Weight above critical threshold (0.1+) should fail."""
        with pytest.raises(PreflightCheckError, match="too high"):
            check_cycle_loss_weight(0.15)


class TestLearningRate:
    """Test learning rate validation (prevent training instability)."""

    def test_recommended_lr_passes(self):
        """NSM-31 recommended LR (5e-4) should pass."""
        result = check_learning_rate(5e-4)

        assert result['is_safe']
        assert not result['is_critical']

    def test_high_lr_warns(self):
        """LR above recommended but below critical should warn."""
        with pytest.warns(PreflightCheckWarning, match="exceeds recommended"):
            result = check_learning_rate(7e-4)
            assert not result['is_safe']
            assert not result['is_critical']

    def test_critical_lr_fails(self):
        """LR above critical threshold (1e-3+) should fail."""
        with pytest.raises(PreflightCheckError, match="too high"):
            check_learning_rate(1.5e-3)


class TestPyGExtensions:
    """Test PyTorch Geometric extension validation."""

    def test_sagpooling_works(self):
        """SAGPooling should work despite torch-scatter/sparse warnings."""
        result = check_pyg_extensions()

        assert result['pyg_available']
        assert result['sagpooling_works']
        assert 0.4 < result['pooling_ratio'] < 0.6  # ~0.5 ratio


class TestIntegration:
    """Test integrated preflight check workflow."""

    def test_good_parameters_pass(self):
        """NSM-31 Phase 1 parameters should pass all checks."""
        # Create balanced dataset
        data = torch.randn(100, 10)
        labels = torch.cat([torch.zeros(50), torch.ones(50)])
        dataset = TensorDataset(data, labels)

        results = run_preflight_checks(
            dataset=dataset,
            cycle_loss_weight=0.01,
            learning_rate=5e-4,
            strict=True
        )

        assert results['all_passed']
        assert not results.get('has_warnings', False)

    def test_bad_parameters_fail(self):
        """NSM-31 original parameters should fail checks."""
        # Create balanced dataset (not the issue)
        data = torch.randn(100, 10)
        labels = torch.cat([torch.zeros(50), torch.ones(50)])
        dataset = TensorDataset(data, labels)

        # Should pass with warnings (not strict)
        with pytest.warns(PreflightCheckWarning):
            results = run_preflight_checks(
                dataset=dataset,
                cycle_loss_weight=0.1,   # Too high!
                learning_rate=1e-3,      # Too high!
                strict=False  # Don't raise, just warn
            )

        assert results['all_passed']  # Passes but with warnings
        assert results['has_warnings']

    def test_imbalanced_without_weights_warns(self):
        """Imbalanced dataset without class weights should warn."""
        # Create moderately imbalanced dataset
        data = torch.randn(100, 10)
        labels = torch.cat([torch.zeros(55), torch.ones(45)])
        dataset = TensorDataset(data, labels)

        with pytest.warns(PreflightCheckWarning, match="imbalanced"):
            results = run_preflight_checks(
                dataset=dataset,
                cycle_loss_weight=0.01,
                learning_rate=5e-4,
                class_weights=None,  # No weights provided!
                strict=False
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
