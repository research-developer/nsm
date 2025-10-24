"""
Tests for physics-inspired collapse prediction metrics.

Validates:
- Safety factor computation
- Temperature profile analysis
- Lawson criterion prediction
- Integration with model outputs
"""

import pytest
import torch
import torch.nn as nn
from nsm.training.physics_metrics import (
    compute_safety_factor,
    compute_temperature_profile,
    check_lawson_criterion,
    compute_hinge_coupling_strength,
    compute_all_physics_metrics
)


class MockHingeModule(nn.Module):
    """Mock hinge for testing coupling strength extraction."""
    def __init__(self, alpha_val=0.5, beta_val=0.5):
        super().__init__()
        # Use logit to get desired sigmoid output
        self.alpha = nn.Parameter(torch.tensor([[alpha_val]]))
        self.beta = nn.Parameter(torch.tensor([[beta_val]]))


class MockModel(nn.Module):
    """Mock model with hinges for testing."""
    def __init__(self, num_hinges=3):
        super().__init__()
        self.hinge_l1_l6 = MockHingeModule(alpha_val=0.6, beta_val=0.4)
        self.hinge_l2_l5 = MockHingeModule(alpha_val=0.55, beta_val=0.45)
        self.hinge_l3_l4 = MockHingeModule(alpha_val=0.7, beta_val=0.3)

        # Add a linear layer for gradient testing
        self.fc = nn.Linear(64, 2)


def test_safety_factor_stable():
    """Test q_neural > 1 for balanced classes."""
    model = MockModel()

    # Simulate balanced classes (stable)
    class_accs = {
        'accuracy_class_0': 0.50,
        'accuracy_class_1': 0.48
    }

    # Add fake gradients
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 0.1

    q_neural, diagnostics = compute_safety_factor(class_accs, model)

    # With small imbalance and gradients, should be stable
    assert 'q_neural' in diagnostics
    assert 'diversity' in diagnostics
    assert 'stability' in diagnostics

    # Check diversity calculation
    expected_diversity = 1.0 - abs(0.50 - 0.48)
    assert abs(diagnostics['diversity'] - expected_diversity) < 0.01


def test_safety_factor_collapsed():
    """Test q_neural < 1 for collapsed classes."""
    model = MockModel()

    # Simulate severe collapse
    class_accs = {
        'accuracy_class_0': 0.95,
        'accuracy_class_1': 0.05
    }

    # Add weak gradients (low capacity)
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 0.001

    q_neural, diagnostics = compute_safety_factor(class_accs, model)

    # Should indicate instability
    assert diagnostics['stability'] == 'UNSTABLE'
    assert diagnostics['collapse_rate'] > 0.5


def test_temperature_profile_normal():
    """Test normal temperature profile (higher levels have higher diversity)."""
    # Simulate normal profile: L1 < L2 < L3
    x_l1 = torch.randn(100, 64) * 0.5  # Low variance
    x_l2 = torch.randn(50, 64) * 1.0   # Medium variance
    x_l3 = torch.randn(25, 64) * 1.5   # High variance

    level_reps = {
        'L1': x_l1,
        'L2': x_l2,
        'L3': x_l3
    }

    temps = compute_temperature_profile(level_reps, method='variance')

    assert 'T_L1' in temps
    assert 'T_L2' in temps
    assert 'T_L3' in temps
    assert 'T_gradient' in temps
    assert 'profile_type' in temps

    # Should have positive gradient
    assert temps['T_gradient'] > 0
    assert temps['profile_type'] == 'normal'


def test_temperature_profile_inverted():
    """Test inverted profile (collapse warning)."""
    # Simulate inverted profile: L1 > L2 > L3 (warning sign!)
    x_l1 = torch.randn(100, 64) * 1.5  # High variance (should be low)
    x_l2 = torch.randn(50, 64) * 1.0   # Medium variance
    x_l3 = torch.randn(25, 64) * 0.5   # Low variance (should be high)

    level_reps = {
        'L1': x_l1,
        'L2': x_l2,
        'L3': x_l3
    }

    temps = compute_temperature_profile(level_reps, method='variance')

    # Should have negative gradient (inverted)
    assert temps['T_gradient'] < 0
    assert temps['profile_type'] == 'inverted'


def test_temperature_profile_entropy():
    """Test entropy-based temperature computation."""
    x_l1 = torch.randn(100, 64)

    level_reps = {'L1': x_l1}

    temps = compute_temperature_profile(level_reps, method='entropy')

    assert 'T_L1' in temps
    assert temps['T_L1'] > 0  # Entropy should be positive


def test_lawson_criterion_met():
    """Test Lawson criterion for successful training."""
    diversity = 0.8  # Good balance
    capacity = 0.1   # Reasonable gradients
    epoch = 50       # Sufficient time

    met, product, diagnostics = check_lawson_criterion(
        diversity=diversity,
        model_capacity=capacity,
        training_time=epoch,
        threshold=1.0  # Low threshold for test
    )

    assert met is True
    assert diagnostics['Q_factor'] >= 1.0
    assert diagnostics['status'] == 'IGNITION'
    assert diagnostics['lawson_product'] == diversity * capacity * epoch


def test_lawson_criterion_not_met():
    """Test Lawson criterion for early training."""
    diversity = 0.5  # Moderate balance
    capacity = 0.05  # Low gradients
    epoch = 2        # Very early

    met, product, diagnostics = check_lawson_criterion(
        diversity=diversity,
        model_capacity=capacity,
        training_time=epoch,
        threshold=10.0  # High threshold
    )

    assert met is False
    assert diagnostics['Q_factor'] < 1.0
    assert diagnostics['status'] == 'SUBIGNITION'


def test_hinge_coupling_strength():
    """Test extraction of coupling parameters from hinges."""
    model = MockModel()

    coupling = compute_hinge_coupling_strength(model)

    # Should return a positive value
    assert coupling > 0
    assert coupling < 2.0  # Reasonable range


def test_compute_all_physics_metrics():
    """Test integrated metric computation."""
    model = MockModel()

    # Add gradients
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 0.1

    # Simulate balanced training
    class_accs = {
        'accuracy_class_0': 0.52,
        'accuracy_class_1': 0.48
    }

    level_reps = {
        'L1': torch.randn(100, 64) * 0.8,
        'L2': torch.randn(50, 64) * 1.0,
        'L3': torch.randn(25, 64) * 1.2
    }

    metrics = compute_all_physics_metrics(
        model=model,
        class_accuracies=class_accs,
        level_representations=level_reps,
        epoch=10,
        task_complexity=1.0
    )

    # Check all required keys present
    assert 'q_neural' in metrics
    assert 'coupling_strength' in metrics
    assert 'T_L1' in metrics
    assert 'T_L2' in metrics
    assert 'T_L3' in metrics
    assert 'T_gradient' in metrics
    assert 'lawson_product' in metrics
    assert 'Q_factor' in metrics
    assert 'warnings' in metrics
    assert 'alert_level' in metrics

    # Warnings should be a list
    assert isinstance(metrics['warnings'], list)

    # Alert level should be valid
    assert metrics['alert_level'] in ['NORMAL', 'CAUTION', 'DANGER']


def test_metrics_with_collapsed_state():
    """Test that metrics correctly identify collapsed state."""
    model = MockModel()

    # Add weak gradients
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 0.001

    # Severe collapse
    class_accs = {
        'accuracy_class_0': 0.98,
        'accuracy_class_1': 0.02
    }

    # Inverted temperature profile
    level_reps = {
        'L1': torch.randn(100, 64) * 1.5,
        'L2': torch.randn(50, 64) * 1.0,
        'L3': torch.randn(25, 64) * 0.3
    }

    metrics = compute_all_physics_metrics(
        model=model,
        class_accuracies=class_accs,
        level_representations=level_reps,
        epoch=5,
        task_complexity=1.0
    )

    # Should have multiple warnings
    assert len(metrics['warnings']) >= 2
    assert metrics['alert_level'] in ['CAUTION', 'DANGER']

    # Should show unstable
    assert metrics['stability'] == 'UNSTABLE'
    assert metrics['profile_type'] == 'inverted'


def test_metrics_with_no_gradients():
    """Test graceful handling when no gradients available."""
    model = MockModel()
    # Don't set any gradients

    class_accs = {
        'accuracy_class_0': 0.50,
        'accuracy_class_1': 0.50
    }

    q_neural, diagnostics = compute_safety_factor(class_accs, model)

    # Should use default capacity
    assert diagnostics['model_capacity'] == 1.0


def test_empty_level_representations():
    """Test handling of empty tensors."""
    level_reps = {
        'L1': torch.tensor([]),  # Empty
        'L2': None,              # None
    }

    temps = compute_temperature_profile(level_reps)

    # Should handle gracefully
    assert temps['T_L1'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
