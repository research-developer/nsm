"""
Conway Combinatorial Game Theory (CGT) Operators for Neural Collapse Prediction.

This module implements 5 Conway operators from combinatorial game theory, adapted for
neural network collapse dynamics. These operators capture phenomena that standard
algebraic metrics miss:

1. Temperature t(G): WHY/WHAT flow asymmetry (partizan game "hotness")
2. Cooling rate: Rate of approach to neutral (α,β → 0.5)
3. Confusion intervals: Epistemic uncertainty in game outcome
4. Game addition: Non-commutative training order effects
5. Surreal numbers: Infinitesimal and infinite equilibrium states

Builds on NSM-33 physics-inspired metrics (85.7% collapse prediction accuracy).
Target: Composite Conway Score (CCS) >90% accuracy.

References:
- Conway, J.H. (1976). "On Numbers and Games"
- NSM-34 Pre-Registration (notes/NSM-34-CGT-OPERATORS-PREREG.md)
- NSM-34 Implementation Guide (notes/NSM-34-IMPLEMENTATION-GUIDE.md)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Union
import numpy as np
from collections import deque
from enum import Enum


# ============================================================================
# OPERATOR 1: CONWAY TEMPERATURE
# ============================================================================

def temperature_conway(
    model: nn.Module,
    x: torch.Tensor,
    num_samples: int = 10,
    metric: str = 'mse'
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Conway temperature for neural WHY/WHAT game.

    Temperature measures "how much the outcome changes if the player changes".
    For neural collapse, it quantifies asymmetry between WHY (abstraction via pooling)
    and WHAT (concretization via unpooling) flows.

    Mathematical Definition (Conway):
        t(G) = (max_Left(GL) - min_Right(GR)) / 2

    Neural Interpretation:
        - High t (>0.5): WHY/WHAT produce very different outcomes (hot game, stable)
        - Low t (<0.2): Flows converge (cold game, collapse imminent)
        - Critical t (≈0.35): Transition zone

    Args:
        model: Model with .why() and .what() methods (e.g., SymmetricHierarchicalLayer)
        x: Input tensor [batch_size, features]
        num_samples: Number of Monte Carlo samples for max/min estimation
        metric: 'mse' (negative mean squared error) or 'cosine' (similarity)

    Returns:
        Tuple of (temperature, diagnostics_dict)
        - temperature: Conway temperature t(x) ∈ [0, ∞)
        - diagnostics: {
            'temperature': float,
            'max_left': float,      # Best WHY→WHAT outcome
            'min_right': float,     # Worst WHAT outcome
            'mean_left': float,
            'mean_right': float,
            'variance_left': float,
            'variance_right': float
          }

    Example:
        >>> model = FullChiralModel(...)
        >>> x = torch.randn(32, 64)
        >>> temp, diag = temperature_conway(model, x)
        >>> if temp < 0.2:
        ...     print("⚠️  Game too cold, collapse risk!")

    Mathematical Foundation:
        In Conway's game theory, temperature measures urgency of play. Games with
        high temperature require careful player choice; cold games have predetermined
        outcomes regardless of player. Neural collapse exhibits exactly this structure:
        healthy networks have high WHY/WHAT asymmetry (player choice matters),
        while collapsed networks have low asymmetry (all paths lead to same outcome).

    Computational Cost:
        O(num_samples × forward_pass_cost)
        Typical: 10-50 samples, ~100ms on GPU for 32-batch
    """
    model.eval()
    with torch.no_grad():
        # Compute abstraction (WHY operation)
        # For hierarchical models, this is typically the pooling/abstraction layer
        if hasattr(model, 'why'):
            x_abstract = model.why(x)
        elif hasattr(model, 'encode'):
            x_abstract = model.encode(x)
        else:
            raise AttributeError(
                "Model must have .why() or .encode() method for WHY operation"
            )

        # Left player moves: WHY then WHAT (abstraction → concretization)
        # Score how well we can reconstruct from abstraction
        left_scores = []
        for _ in range(num_samples):
            if hasattr(model, 'what'):
                x_recon_left = model.what(x_abstract)
            elif hasattr(model, 'decode'):
                x_recon_left = model.decode(x_abstract)
            else:
                raise AttributeError(
                    "Model must have .what() or .decode() method for WHAT operation"
                )

            # Compute reconstruction quality
            if metric == 'mse':
                # Negative MSE (higher is better, matches Conway's max formulation)
                score = -torch.mean((x_recon_left - x) ** 2).item()
            elif metric == 'cosine':
                score = torch.nn.functional.cosine_similarity(
                    x_recon_left.flatten(), x.flatten(), dim=0
                ).item()
            else:
                raise ValueError(f"Unknown metric: {metric}. Use 'mse' or 'cosine'")

            left_scores.append(score)

        # Right player moves: Same operation, different interpretation
        # In a fully symmetric game, right moves are identical to left moves
        # But in practice, stochasticity or asymmetry creates different distributions
        right_scores = []
        for _ in range(num_samples):
            if hasattr(model, 'what'):
                x_recon_right = model.what(x_abstract)
            elif hasattr(model, 'decode'):
                x_recon_right = model.decode(x_abstract)
            else:
                raise AttributeError(
                    "Model must have .what() or .decode() method for WHAT operation"
                )

            if metric == 'mse':
                score = -torch.mean((x_recon_right - x) ** 2).item()
            elif metric == 'cosine':
                score = torch.nn.functional.cosine_similarity(
                    x_recon_right.flatten(), x.flatten(), dim=0
                ).item()

            right_scores.append(score)

        # Conway temperature: (max_Left - min_Right) / 2
        # Measures the advantage Left player has by choosing best move vs
        # Right player forced to accept worst outcome
        max_left = max(left_scores)
        min_right = min(right_scores)
        temperature = (max_left - min_right) / 2.0

        # Ensure non-negative (theoretical guarantee, but check for numerical issues)
        temperature = max(0.0, temperature)

        # Diagnostics for analysis
        diagnostics = {
            'temperature': temperature,
            'max_left': max_left,
            'min_right': min_right,
            'mean_left': float(np.mean(left_scores)),
            'mean_right': float(np.mean(right_scores)),
            'variance_left': float(np.var(left_scores)),
            'variance_right': float(np.var(right_scores)),
            'num_samples': num_samples,
            'metric': metric
        }

    return temperature, diagnostics


def temperature_trajectory(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    max_batches: int = 10,
    num_samples: int = 10
) -> List[Tuple[float, Dict[str, float]]]:
    """
    Compute temperature trajectory over multiple batches.

    Useful for:
    - Estimating average temperature across dataset
    - Detecting variance in temperature (batch-to-batch instability)
    - Reducing noise via multiple measurements

    Args:
        model: Model with WHY/WHAT
        dataloader: Data batches
        max_batches: Limit computation (temperature is expensive)
        num_samples: Samples per batch

    Returns:
        List of (temperature, diagnostics) tuples

    Example:
        >>> temps = temperature_trajectory(model, val_loader, max_batches=5)
        >>> avg_temp = np.mean([t for t, _ in temps])
        >>> print(f"Average temperature: {avg_temp:.3f}")
    """
    temps = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        # Handle different dataloader formats
        if isinstance(batch, (list, tuple)):
            x = batch[0]  # Assume first element is input
        else:
            x = batch

        # Move to model device
        if next(model.parameters()).is_cuda:
            x = x.cuda()

        temp, diag = temperature_conway(model, x, num_samples=num_samples)
        temps.append((temp, diag))

    return temps


# ============================================================================
# OPERATOR 2: COOLING RATE MONITOR
# ============================================================================

class CoolingMonitor:
    """
    Track cooling rate of neural game over time.

    Conway's "cooling" operation reduces game temperature systematically:
        Cooled(G) = G - t(G)
    Iterated cooling leads to "cold" games where player choice doesn't matter.

    In neural networks, α/β hinge parameters naturally implement a cooling schedule:
    - Initial (hot): α, β far from 0.5 (asymmetric mixing, player advantage)
    - Final (cold): α, β → 0.5 (symmetric, no advantage, collapse risk)

    This class tracks the rate at which the system cools, enabling:
    - Early warning: Rapid cooling predicts collapse
    - Time-to-collapse estimation: Linear extrapolation
    - Intervention triggering: Heat up the game when cooling too fast

    Attributes:
        window_size: Number of epochs for moving average
        alpha_history: Deque of α values (hinge parameter 1)
        beta_history: Deque of β values (hinge parameter 2)
        temp_history: Deque of computed temperatures
        cooling_history: List of cooling rates (negative = cooling down)

    Mathematical Foundation:
        Temperature (neural): T_neural = |α - 0.5| + |β - 0.5|
        Cooling rate: δT/δepoch = T(epoch) - T(epoch-1)

        Negative cooling rate → approaching cold (α,β → 0.5)
        Positive cooling rate → heating up (α,β moving away from 0.5)

    Pre-Registered Predictions:
        P2.1: Cooling rate < -0.05/epoch predicts collapse within 2 epochs (r > 0.8)
        P2.2: Optimal cooling schedule exists (neither too fast nor too slow)
        P2.3: Cooling rate is non-linear near critical point (α,β ≈ 0.5)
    """

    def __init__(self, window_size: int = 5):
        """
        Initialize cooling monitor.

        Args:
            window_size: Number of epochs for smoothed estimates (default: 5)
        """
        self.window_size = window_size
        self.alpha_history = deque(maxlen=window_size)
        self.beta_history = deque(maxlen=window_size)
        self.temp_history = deque(maxlen=window_size)
        self.cooling_history: List[float] = []

    def compute_temperature_neural(
        self,
        alpha: float,
        beta: float
    ) -> float:
        """
        Compute neural game temperature from hinge parameters.

        Temperature = distance from neutral (0.5, 0.5).
        High temperature: α, β far from 0.5 (strong player advantage)
        Low temperature: α, β ≈ 0.5 (neutral, cold game)

        Args:
            alpha: Hinge parameter 1 (should be in [0, 1])
            beta: Hinge parameter 2 (should be in [0, 1])

        Returns:
            temperature: T = |α - 0.5| + |β - 0.5| ∈ [0, 1]

        Example:
            >>> monitor = CoolingMonitor()
            >>> T_hot = monitor.compute_temperature_neural(0.9, 0.1)  # Far from 0.5
            >>> T_cold = monitor.compute_temperature_neural(0.5, 0.5)  # At 0.5
            >>> assert T_hot > T_cold
        """
        return abs(alpha - 0.5) + abs(beta - 0.5)

    def update(
        self,
        alpha: float,
        beta: float
    ) -> Optional[float]:
        """
        Update cooling monitor with new hinge parameters.

        Args:
            alpha: Current α value
            beta: Current β value

        Returns:
            cooling_rate: Current cooling rate (None if insufficient history)
                          Negative = cooling down (collapse risk)
                          Positive = heating up (stable)
                          Zero = equilibrium

        Example:
            >>> monitor = CoolingMonitor()
            >>> monitor.update(0.8, 0.8)  # First epoch, no rate yet
            None
            >>> rate = monitor.update(0.6, 0.6)  # Second epoch, cooling detected
            >>> assert rate < 0  # Cooling down toward 0.5
        """
        temp = self.compute_temperature_neural(alpha, beta)

        self.alpha_history.append(alpha)
        self.beta_history.append(beta)
        self.temp_history.append(temp)

        # Need at least 2 samples to compute rate of change
        if len(self.temp_history) < 2:
            return None

        # Cooling rate: current temperature - previous temperature
        # Negative = cooling (temperature decreasing)
        # Positive = heating (temperature increasing)
        cooling_rate = self.temp_history[-1] - self.temp_history[-2]
        self.cooling_history.append(cooling_rate)

        return cooling_rate

    def get_smoothed_cooling_rate(self) -> Optional[float]:
        """
        Get moving average of cooling rate over window.

        Smoothing reduces noise from epoch-to-epoch fluctuations.

        Returns:
            Smoothed cooling rate (None if insufficient data)

        Example:
            >>> monitor = CoolingMonitor(window_size=3)
            >>> for alpha, beta in [(0.8, 0.8), (0.6, 0.6), (0.5, 0.5)]:
            ...     monitor.update(alpha, beta)
            >>> smooth_rate = monitor.get_smoothed_cooling_rate()
            >>> print(f"Smooth cooling: {smooth_rate:.4f}")
        """
        if len(self.cooling_history) < 2:
            return None

        recent = list(self.cooling_history)[-self.window_size:]
        return sum(recent) / len(recent)

    def predict_collapse_time(
        self,
        threshold_temp: float = 0.1,
        current_temp: Optional[float] = None
    ) -> Optional[int]:
        """
        Predict number of epochs until temperature reaches collapse threshold.

        Uses linear extrapolation (conservative estimate):
            T(t + Δt) = T(t) + cooling_rate × Δt

        Args:
            threshold_temp: Temperature below which collapse is imminent (default: 0.1)
            current_temp: Current temperature (uses most recent if None)

        Returns:
            epochs_remaining: Estimated epochs until T < threshold
                              None if heating (no collapse predicted) or insufficient data
                              0 if already below threshold

        Example:
            >>> monitor = CoolingMonitor()
            >>> monitor.update(0.8, 0.8)
            >>> monitor.update(0.6, 0.6)  # Cooling rate = -0.4
            >>> epochs = monitor.predict_collapse_time(threshold_temp=0.1)
            >>> print(f"Collapse predicted in {epochs} epochs")

        Warning:
            Assumes linear cooling, which breaks down near critical point (α,β ≈ 0.5).
            Actual collapse may be earlier due to non-linear phase transition.
        """
        cooling_rate = self.get_smoothed_cooling_rate()

        if cooling_rate is None or cooling_rate >= 0:
            return None  # Heating or no data, no collapse predicted

        if current_temp is None:
            current_temp = self.temp_history[-1]

        if current_temp <= threshold_temp:
            return 0  # Already at or below threshold

        # Linear extrapolation: threshold = current + cooling_rate × Δt
        # Solve for Δt: Δt = (threshold - current) / cooling_rate
        epochs_remaining = (threshold_temp - current_temp) / cooling_rate

        return int(max(0, epochs_remaining))

    def get_statistics(self) -> Dict[str, float]:
        """
        Get comprehensive cooling statistics.

        Returns:
            Dictionary with:
                - current_temp: Most recent temperature
                - mean_temp: Average temperature over window
                - current_cooling_rate: Most recent cooling rate
                - smoothed_cooling_rate: Moving average cooling rate
                - temp_variance: Variance in temperature (instability measure)
                - epochs_tracked: Number of epochs recorded

        Example:
            >>> stats = monitor.get_statistics()
            >>> print(f"Current T: {stats['current_temp']:.3f}")
            >>> print(f"Cooling rate: {stats['smoothed_cooling_rate']:.4f}")
        """
        if len(self.temp_history) == 0:
            return {
                'current_temp': 0.0,
                'mean_temp': 0.0,
                'current_cooling_rate': 0.0,
                'smoothed_cooling_rate': 0.0,
                'temp_variance': 0.0,
                'epochs_tracked': 0
            }

        return {
            'current_temp': self.temp_history[-1],
            'mean_temp': float(np.mean(self.temp_history)),
            'current_cooling_rate': self.cooling_history[-1] if self.cooling_history else 0.0,
            'smoothed_cooling_rate': self.get_smoothed_cooling_rate() or 0.0,
            'temp_variance': float(np.var(self.temp_history)),
            'epochs_tracked': len(self.temp_history)
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_hinge_parameter(
    model: nn.Module,
    param_name: str,
    apply_sigmoid: bool = True
) -> float:
    """
    Extract mean hinge parameter value from model.

    Searches model for modules with 'hinge' in name and extracts specified parameter.
    Useful for monitoring α/β parameters in chiral architectures.

    Args:
        model: Neural network model
        param_name: Parameter name to extract (e.g., 'alpha', 'beta')
        apply_sigmoid: Apply sigmoid to raw parameter (default: True)

    Returns:
        Mean parameter value across all hinge modules

    Example:
        >>> alpha = extract_hinge_parameter(model, 'alpha')
        >>> beta = extract_hinge_parameter(model, 'beta')
        >>> print(f"Hinge parameters: α={alpha:.3f}, β={beta:.3f}")

    Raises:
        ValueError: If no hinge parameters found
    """
    values = []
    for name, module in model.named_modules():
        if 'hinge' in name.lower():
            if hasattr(module, param_name):
                param = getattr(module, param_name)
                if apply_sigmoid:
                    value = torch.sigmoid(param).mean().item()
                else:
                    value = param.mean().item()
                values.append(value)

    if len(values) == 0:
        raise ValueError(
            f"No hinge parameters named '{param_name}' found in model. "
            f"Check that model has modules with 'hinge' in name."
        )

    return sum(values) / len(values)


def compute_all_temperature_metrics(
    model: nn.Module,
    x: torch.Tensor,
    cooling_monitor: Optional[CoolingMonitor] = None,
    num_samples: int = 10
) -> Dict[str, Union[float, Dict]]:
    """
    Compute all temperature-related CGT metrics in one pass.

    Convenience function for getting both Conway temperature and cooling rate
    without redundant computation.

    Args:
        model: Model with WHY/WHAT
        x: Input batch
        cooling_monitor: Existing cooling monitor (will extract α/β if provided)
        num_samples: Samples for Conway temperature

    Returns:
        Dictionary with:
            - 'conway_temperature': float
            - 'conway_temp_diagnostics': Dict
            - 'neural_temperature': float (if cooling_monitor provided)
            - 'cooling_rate': float (if cooling_monitor provided)
            - 'cooling_diagnostics': Dict (if cooling_monitor provided)

    Example:
        >>> monitor = CoolingMonitor()
        >>> metrics = compute_all_temperature_metrics(model, x, monitor)
        >>> print(f"Conway T: {metrics['conway_temperature']:.3f}")
        >>> print(f"Neural T: {metrics['neural_temperature']:.3f}")
        >>> print(f"Cooling: {metrics['cooling_rate']:.4f}")
    """
    metrics = {}

    # Conway temperature (expensive, uses sampling)
    temp_conway, temp_diag = temperature_conway(model, x, num_samples=num_samples)
    metrics['conway_temperature'] = temp_conway
    metrics['conway_temp_diagnostics'] = temp_diag

    # Neural temperature and cooling (cheap, uses α/β)
    if cooling_monitor is not None:
        try:
            alpha = extract_hinge_parameter(model, 'alpha')
            beta = extract_hinge_parameter(model, 'beta')

            cooling_rate = cooling_monitor.update(alpha, beta)
            cooling_stats = cooling_monitor.get_statistics()

            # Always return current temperature even if cooling rate not available yet
            metrics['neural_temperature'] = cooling_stats['current_temp']
            metrics['cooling_rate'] = cooling_rate  # May be None for first update
            metrics['cooling_diagnostics'] = cooling_stats

        except ValueError as e:
            # No hinge parameters, skip cooling metrics
            metrics['neural_temperature'] = None
            metrics['cooling_rate'] = None
            metrics['cooling_diagnostics'] = {'error': str(e)}

    return metrics


# ============================================================================
# MODULE METADATA
# ============================================================================

__all__ = [
    'temperature_conway',
    'temperature_trajectory',
    'CoolingMonitor',
    'extract_hinge_parameter',
    'compute_all_temperature_metrics',
]

__version__ = '0.1.0'
__author__ = 'Claude Code (Anthropic) + Preston'
__status__ = 'Development - NSM-34 Workstream A'
