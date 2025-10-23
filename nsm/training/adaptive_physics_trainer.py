"""
Adaptive physics-based training control.

Uses fusion-plasma isomorphism metrics to dynamically adjust hyperparameters:
- When q_neural < 1.0: Increase diversity weight (raise "temperature")
- When temperature inverted: Increase cycle weight (improve confinement)
- When Q factor low: Reduce learning rate (cool down)

Tests if physics-informed adaptation outperforms fixed hyperparameters.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AdaptivePhysicsConfig:
    """Configuration for physics-based adaptive training."""

    # Initial hyperparameters
    initial_diversity_weight: float = 0.0
    initial_cycle_weight: float = 0.01
    initial_learning_rate: float = 1e-4

    # Physics thresholds for intervention
    q_unstable_threshold: float = 1.0  # q < 1.0 triggers action
    q_critical_threshold: float = 0.5  # q < 0.5 triggers aggressive action
    temp_inversion_threshold: float = -0.1  # gradient < -0.1 triggers action
    Q_factor_threshold: float = 0.5  # Q < 0.5 triggers cooling

    # Adaptation rates
    diversity_increment: float = 0.05  # How much to increase per step
    diversity_max: float = 0.5  # Maximum diversity weight
    cycle_increment: float = 0.02
    cycle_max: float = 0.2
    lr_decay_factor: float = 0.9  # Multiply LR by this when cooling

    # Adaptation frequency
    check_every_n_epochs: int = 1
    cooldown_epochs: int = 2  # Wait N epochs after intervention

    # Control mode
    enable_q_control: bool = True
    enable_temp_control: bool = True
    enable_Q_control: bool = True


class AdaptivePhysicsTrainer:
    """
    Training controller that uses physics metrics for adaptive hyperparameter tuning.

    Implements fusion reactor-inspired control strategy:
    1. Monitor plasma stability (q_neural)
    2. Detect temperature inversions
    3. Track energy confinement (Q factor)
    4. Adjust "control parameters" to maintain stability
    """

    def __init__(
        self,
        config: AdaptivePhysicsConfig,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module
    ):
        self.config = config
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # Current hyperparameters (will be adapted)
        self.diversity_weight = config.initial_diversity_weight
        self.cycle_weight = config.initial_cycle_weight
        self.learning_rate = config.initial_learning_rate

        # Intervention tracking
        self.last_intervention_epoch = -999
        self.intervention_history = []

        # Metrics history
        self.physics_history = []

    def should_intervene(self, epoch: int) -> bool:
        """Check if enough time has passed since last intervention."""
        cooldown_satisfied = (epoch - self.last_intervention_epoch) >= self.config.cooldown_epochs
        check_frequency = epoch % self.config.check_every_n_epochs == 0
        return cooldown_satisfied and check_frequency

    def analyze_and_adapt(
        self,
        epoch: int,
        physics_metrics: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Analyze physics metrics and adapt hyperparameters if needed.

        Args:
            epoch: Current training epoch
            physics_metrics: Dict from compute_all_physics_metrics()

        Returns:
            Dict with:
                - adapted: bool, whether adaptation occurred
                - interventions: List of changes made
                - new_hyperparams: Updated values
        """

        # Store history
        self.physics_history.append({
            'epoch': epoch,
            'q_neural': physics_metrics['q_neural'],
            'temp_gradient': physics_metrics.get('T_gradient', 0.0),
            'Q_factor': physics_metrics['Q_factor'],
            'diversity_weight': self.diversity_weight,
            'cycle_weight': self.cycle_weight,
            'learning_rate': self.learning_rate
        })

        if not self.should_intervene(epoch):
            return {
                'adapted': False,
                'reason': 'cooldown period',
                'interventions': []
            }

        interventions = []
        adapted = False

        # Extract metrics
        q_neural = physics_metrics['q_neural']
        temp_gradient = physics_metrics.get('T_gradient', 0.0)
        Q_factor = physics_metrics['Q_factor']

        # CONTROL 1: Stability (q_neural)
        if self.config.enable_q_control:
            if q_neural < self.config.q_critical_threshold:
                # CRITICAL: Aggressive intervention
                increment = self.config.diversity_increment * 2.0
                new_diversity = min(self.diversity_weight + increment, self.config.diversity_max)

                if new_diversity > self.diversity_weight:
                    old_val = self.diversity_weight
                    self.diversity_weight = new_diversity
                    interventions.append(f"🚨 CRITICAL q={q_neural:.3f}: diversity {old_val:.3f} → {new_diversity:.3f}")
                    adapted = True

            elif q_neural < self.config.q_unstable_threshold:
                # WARNING: Moderate intervention
                new_diversity = min(self.diversity_weight + self.config.diversity_increment,
                                   self.config.diversity_max)

                if new_diversity > self.diversity_weight:
                    old_val = self.diversity_weight
                    self.diversity_weight = new_diversity
                    interventions.append(f"⚠️  Unstable q={q_neural:.3f}: diversity {old_val:.3f} → {new_diversity:.3f}")
                    adapted = True

        # CONTROL 2: Temperature profile (inversion)
        if self.config.enable_temp_control:
            if temp_gradient < self.config.temp_inversion_threshold:
                # Inverted profile: Strengthen cycle consistency to enforce hierarchy
                new_cycle = min(self.cycle_weight + self.config.cycle_increment,
                               self.config.cycle_max)

                if new_cycle > self.cycle_weight:
                    old_val = self.cycle_weight
                    self.cycle_weight = new_cycle
                    interventions.append(f"🌡️  Inverted T gradient={temp_gradient:.3f}: cycle {old_val:.3f} → {new_cycle:.3f}")
                    adapted = True

        # CONTROL 3: Energy confinement (Q factor)
        if self.config.enable_Q_control:
            if Q_factor < self.config.Q_factor_threshold:
                # Low Q: Cool down learning rate
                new_lr = self.learning_rate * self.config.lr_decay_factor

                if new_lr < self.learning_rate:
                    old_val = self.learning_rate
                    self.learning_rate = new_lr

                    # Update optimizer learning rate
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr

                    interventions.append(f"❄️  Low Q={Q_factor:.3f}: LR {old_val:.4e} → {new_lr:.4e}")
                    adapted = True

        # Update loss function weights
        if hasattr(self.loss_fn, 'diversity_weight'):
            self.loss_fn.diversity_weight = self.diversity_weight
        if hasattr(self.loss_fn, 'cycle_weight'):
            self.loss_fn.cycle_weight = self.cycle_weight

        if adapted:
            self.last_intervention_epoch = epoch
            self.intervention_history.append({
                'epoch': epoch,
                'interventions': interventions,
                'physics_metrics': physics_metrics,
                'new_hyperparams': {
                    'diversity_weight': self.diversity_weight,
                    'cycle_weight': self.cycle_weight,
                    'learning_rate': self.learning_rate
                }
            })

        return {
            'adapted': adapted,
            'interventions': interventions,
            'new_hyperparams': {
                'diversity_weight': self.diversity_weight,
                'cycle_weight': self.cycle_weight,
                'learning_rate': self.learning_rate
            }
        }

    def get_current_hyperparams(self) -> Dict[str, float]:
        """Get current hyperparameter values."""
        return {
            'diversity_weight': self.diversity_weight,
            'cycle_weight': self.cycle_weight,
            'learning_rate': self.learning_rate
        }

    def get_intervention_summary(self) -> Dict[str, any]:
        """Get summary of all interventions made."""
        return {
            'total_interventions': len(self.intervention_history),
            'history': self.intervention_history,
            'final_hyperparams': self.get_current_hyperparams(),
            'physics_trajectory': self.physics_history
        }


# Export public API
__all__ = [
    'AdaptivePhysicsConfig',
    'AdaptivePhysicsTrainer'
]
