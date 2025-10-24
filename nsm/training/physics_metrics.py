"""
Physics-inspired empirical heuristics for predicting class collapse in chiral neural architectures.

Implements fusion-plasma-inspired metrics:
- Safety factor q_neural (stability predictor)
- Representation variance profiles (diversity tracking)
- Lawson criterion analog (training success predictor)

**Note**: These are empirical heuristics (not rigorous isomorphisms) inspired by structural
similarities to fusion plasma systems. Dimensional analysis reveals they lack true physical
correspondence, but remain useful predictive tools validated through NSM-33 experiments.

**Peer Review**: Terminology updated per research-assistant feedback (2025-10-23).
See TERMINOLOGY_UPDATES.md for complete rationale and change log.

Mathematical parallels (structural, not isomorphic):
- Neural class collapse ↔ Plasma confinement loss
- α/β hinge parameters ↔ α/β fusion parameters
- Representation variance ↔ Temperature in fusion systems

References:
- Lawson, J.D. (1957). "Some Criteria for a Power Producing Thermonuclear Reactor"
- Wesson, J. (2011). "Tokamak Physics" (safety factor q)
- NSM-32: 6-Level Chiral Architecture validation results
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np


def compute_safety_factor(
    class_accuracies: Dict[str, float],
    model: nn.Module,
    coupling_strength: float = 1.0,
    epsilon: float = 1e-8
) -> Tuple[float, Dict[str, float]]:
    """
    Compute q_neural (safety factor analog for neural collapse prediction).

    The safety factor q in tokamaks measures stability against kink instabilities.
    Here, q_neural predicts training stability against class collapse:
        q > 1: Stable training, no collapse risk
        q < 1: Unstable, collapse imminent

    Formula (inspired by tokamak q = rB_φ / RB_θ):
        q_neural = (diversity × model_capacity) / (collapse_tendency × coupling + ε)

    Where:
        - diversity: Class balance (1 - |acc_0 - acc_1|), analogous to temperature
        - model_capacity: Gradient flow strength, analogous to magnetic field
        - collapse_tendency: Current class imbalance, analogous to plasma current
        - coupling: Hinge exchange strength (from α, β parameters)

    Args:
        class_accuracies: Dict with 'accuracy_class_0', 'accuracy_class_1'
        model: The neural network model
        coupling_strength: Effective hinge coupling (default: 1.0)
        epsilon: Numerical stability constant

    Returns:
        Tuple of (q_neural, diagnostics_dict)
        - q_neural > 1.0: Stable
        - q_neural < 1.0: Collapse risk
        - diagnostics: Breakdown of components
    """
    # Extract class accuracies
    acc_0 = class_accuracies.get('accuracy_class_0', 0.5)
    acc_1 = class_accuracies.get('accuracy_class_1', 0.5)

    # Diversity (temperature analog): How balanced are the classes?
    # diversity = 1 means perfect balance (50/50)
    # diversity = 0 means total collapse (100/0 or 0/100)
    diversity = 1.0 - abs(acc_0 - acc_1)

    # Model capacity (magnetic field analog): Gradient flow strength
    # Measures how much "energy" the model has to resist collapse
    grad_norms = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())

    if len(grad_norms) > 0:
        capacity = np.mean(grad_norms) + epsilon
    else:
        capacity = 1.0  # No gradients yet, assume unit capacity

    # Collapse tendency (plasma current analog): Current imbalance
    # High values indicate strong pressure toward collapse
    collapse_rate = abs(acc_0 - acc_1) + epsilon

    # Safety factor computation
    q_neural = (diversity * capacity) / (collapse_rate * coupling_strength + epsilon)

    # Diagnostic breakdown
    diagnostics = {
        'q_neural': q_neural,
        'diversity': diversity,
        'model_capacity': capacity,
        'collapse_rate': collapse_rate,
        'coupling_strength': coupling_strength,
        'stability': 'STABLE' if q_neural >= 1.0 else 'UNSTABLE'
    }

    return q_neural, diagnostics


def compute_temperature_profile(
    level_representations: Dict[str, torch.Tensor],
    method: str = 'variance'
) -> Dict[str, float]:
    """
    Compute representation variance profile at each hierarchical level.

    **Note**: "Temperature" here refers to representation variance/entropy, NOT thermal
    temperature. The term is borrowed from fusion physics by analogy but represents a
    fundamentally different quantity (statistical dispersion, not kinetic energy).

    In the fusion analogy: temperature profiles T(r) determine confinement quality.
    In neural networks: representation variance serves structurally analogous role:
        - High variance: Diverse, information-rich representations
        - Low variance: Collapsed, uniform representations
        - Inverted profile (variance decreasing with abstraction): Instability indicator

    Variance inversions empirically correlate with collapse events in NSM-33 experiments.

    Args:
        level_representations: Dict mapping level names to feature tensors
            e.g., {'L1': x_l1, 'L2': x_l2, 'L3': x_l3}
        method: 'variance' or 'entropy' for measurement

    Returns:
        Dict with:
            - 'T_{level}': Variance/entropy at each level (NOT thermal temperature)
            - 'T_gradient': Variance gradient (L1 → L3)
            - 'profile_type': 'normal', 'flat', or 'inverted'
    """
    temperatures = {}

    for level_name, x in level_representations.items():
        if x is None or x.numel() == 0:
            temperatures[f'T_{level_name}'] = 0.0
            continue

        if method == 'variance':
            # Variance-based measurement: Spread of representations
            temp = x.var(dim=0).mean().item()
        elif method == 'entropy':
            # Entropy-based measurement: Information content
            # Use softmax to get probability distribution
            probs = torch.softmax(x, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
            temp = entropy
        else:
            raise ValueError(f"Unknown method: {method}")

        temperatures[f'T_{level_name}'] = temp

    # Compute variance gradient (should be positive: L1 < L2 < L3 for healthy hierarchy)
    level_order = sorted([k for k in temperatures.keys() if k.startswith('T_L')])
    if len(level_order) >= 2:
        T_first = temperatures[level_order[0]]
        T_last = temperatures[level_order[-1]]
        gradient = T_last - T_first
        temperatures['T_gradient'] = gradient

        # Classify profile type
        if gradient > 0.1:
            temperatures['profile_type'] = 'normal'  # Higher levels hotter (good)
        elif gradient < -0.1:
            temperatures['profile_type'] = 'inverted'  # Collapse warning!
        else:
            temperatures['profile_type'] = 'flat'  # Neutral
    else:
        temperatures['T_gradient'] = 0.0
        temperatures['profile_type'] = 'unknown'

    return temperatures


def check_lawson_criterion(
    diversity: float,
    model_capacity: float,
    training_time: int,
    threshold: float = 1e3,
    task_complexity: float = 1.0
) -> Tuple[bool, float, Dict[str, float]]:
    """
    Neural Lawson criterion: Predicts if training will achieve target accuracy.

    In fusion, Lawson criterion determines if fusion will be self-sustaining:
        n·τ·T > threshold  (density × confinement time × temperature)

    Neural analog:
        diversity × capacity × training_time > threshold

    This predicts whether the training has sufficient "energy-confinement product"
    to reach target accuracy without collapse.

    Args:
        diversity: Class balance (1 - |acc_0 - acc_1|)
        model_capacity: Gradient flow strength
        training_time: Current epoch number
        threshold: Minimum product required (default: 1e3)
        task_complexity: Task difficulty multiplier (default: 1.0)

    Returns:
        Tuple of (criterion_met, product, diagnostics)
        - criterion_met: True if training likely to succeed
        - product: n·τ·T value
        - diagnostics: Breakdown
    """
    # Compute triple product (fusion analog)
    product = diversity * model_capacity * training_time

    # Adjust threshold by task complexity
    adjusted_threshold = threshold * task_complexity

    # Check criterion
    criterion_met = product >= adjusted_threshold

    # Compute Q factor (energy gain analog)
    # Q = product / threshold
    # Q > 1: "Ignition" - training succeeding
    # Q < 1: "Subignition" - needs more time/capacity
    Q_factor = product / adjusted_threshold if adjusted_threshold > 0 else 0.0

    diagnostics = {
        'lawson_product': product,
        'threshold': adjusted_threshold,
        'Q_factor': Q_factor,
        'diversity': diversity,
        'model_capacity': model_capacity,
        'training_time': training_time,
        'criterion_met': criterion_met,
        'status': 'IGNITION' if Q_factor >= 1.0 else 'SUBIGNITION'
    }

    return criterion_met, product, diagnostics


def compute_hinge_coupling_strength(model: nn.Module) -> float:
    """
    Extract effective coupling strength from hinge α, β parameters.

    In fusion, coupling parameters determine energy exchange between
    electron/ion fluids. Here, α and β determine information exchange
    between WHY/WHAT flows.

    Args:
        model: Model with ChiralHingeExchange modules

    Returns:
        Average coupling strength across all hinges
    """
    alphas = []
    betas = []

    # Find all hinge modules
    for name, module in model.named_modules():
        if 'hinge' in name.lower():
            if hasattr(module, 'alpha'):
                alpha = torch.sigmoid(module.alpha).mean().item()
                alphas.append(alpha)
            if hasattr(module, 'beta'):
                beta = torch.sigmoid(module.beta).mean().item()
                betas.append(beta)

    if len(alphas) > 0 and len(betas) > 0:
        # Coupling strength: How far from 0.5 (neutral) are the mixing parameters?
        # High coupling = strong exchange, low coupling = weak exchange
        avg_alpha = np.mean(alphas)
        avg_beta = np.mean(betas)

        # Distance from neutral (0.5)
        coupling = abs(avg_alpha - 0.5) + abs(avg_beta - 0.5) + 0.5
        return coupling
    else:
        return 1.0  # Default if no hinges found


def compute_all_physics_metrics(
    model: nn.Module,
    class_accuracies: Dict[str, float],
    level_representations: Dict[str, torch.Tensor],
    epoch: int,
    task_complexity: float = 1.0
) -> Dict[str, any]:
    """
    Compute all physics-inspired metrics in one call.

    Convenience function for training loop integration.

    Args:
        model: Neural network model
        class_accuracies: Per-class accuracy dict
        level_representations: Dict of level tensors
        epoch: Current training epoch
        task_complexity: Task difficulty multiplier

    Returns:
        Comprehensive metrics dict with:
            - q_neural and stability
            - Temperature profile
            - Lawson criterion
            - Coupling strength
            - Warnings/alerts
    """
    metrics = {}

    # 1. Coupling strength
    coupling = compute_hinge_coupling_strength(model)
    metrics['coupling_strength'] = coupling

    # 2. Safety factor
    q_neural, q_diagnostics = compute_safety_factor(
        class_accuracies=class_accuracies,
        model=model,
        coupling_strength=coupling
    )
    metrics['q_neural'] = q_neural
    metrics.update(q_diagnostics)

    # 3. Temperature profile
    temp_profile = compute_temperature_profile(level_representations)
    metrics.update(temp_profile)

    # 4. Lawson criterion
    diversity = q_diagnostics['diversity']
    capacity = q_diagnostics['model_capacity']
    criterion_met, product, lawson_diag = check_lawson_criterion(
        diversity=diversity,
        model_capacity=capacity,
        training_time=epoch,
        task_complexity=task_complexity
    )
    metrics.update(lawson_diag)

    # 5. Generate warnings
    warnings = []
    if q_neural < 1.0:
        warnings.append(f"⚠️  COLLAPSE RISK: q_neural = {q_neural:.3f} < 1.0")
    if temp_profile.get('profile_type') == 'inverted':
        warnings.append(f"⚠️  INSTABILITY: Inverted temperature profile")
    if not criterion_met:
        Q = lawson_diag['Q_factor']
        warnings.append(f"⚠️  SUBIGNITION: Q = {Q:.3f} < 1.0")

    metrics['warnings'] = warnings
    metrics['alert_level'] = 'DANGER' if len(warnings) >= 2 else ('CAUTION' if len(warnings) == 1 else 'NORMAL')

    return metrics


# Export public API
__all__ = [
    'compute_safety_factor',
    'compute_temperature_profile',
    'check_lawson_criterion',
    'compute_hinge_coupling_strength',
    'compute_all_physics_metrics'
]
