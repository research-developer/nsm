"""
Temperature scheduling for confidence propagation.

Anneals temperature from high (smooth/differentiable) to low (discrete/sharp)
during training. Used for soft min/max operations in semirings.
"""

import torch
from typing import Optional


class TemperatureScheduler:
    """Exponential temperature annealing scheduler.

    Smoothly transitions from high temperature (soft, differentiable operations)
    to low temperature (sharp, discrete-like operations) during training.

    Mathematical formulation:
        τ(epoch) = max(τ_final, τ_initial * decay^epoch)

    Use cases:
    - Soft min/max in MinMaxSemiring
    - Gumbel-softmax in discrete reasoning
    - Attention temperature in LearnedSemiring

    Args:
        initial_temp (float): Starting temperature (smooth). Default: 1.0
        final_temp (float): Minimum temperature (sharp). Default: 0.3
        decay_rate (float): Exponential decay rate per epoch. Default: 0.9999
        warmup_epochs (int, optional): Epochs to stay at initial_temp

    Example:
        >>> scheduler = TemperatureScheduler(initial=1.0, final=0.3)
        >>>
        >>> for epoch in range(1000):
        ...     temp = scheduler.step()
        ...     model.set_temperature(temp)
        ...     train_epoch()
        >>>
        >>> # Temperature decays: 1.0 → 0.9 → 0.8 → ... → 0.3
    """

    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.3,
        decay_rate: float = 0.9999,
        warmup_epochs: int = 0
    ):
        assert initial_temp > 0, "Initial temperature must be positive"
        assert final_temp > 0, "Final temperature must be positive"
        assert 0 < decay_rate < 1, "Decay rate must be in (0, 1)"
        assert warmup_epochs >= 0, "Warmup epochs must be non-negative"

        self.initial = initial_temp
        self.final = final_temp
        self.decay = decay_rate
        self.warmup_epochs = warmup_epochs

        self.current_temp = initial_temp
        self.epoch = 0

    def step(self) -> float:
        """Advance one epoch and return current temperature.

        Returns:
            float: Current temperature after update

        Example:
            >>> scheduler = TemperatureScheduler(1.0, 0.3, 0.99)
            >>> temp1 = scheduler.step()  # 1.0 (first step)
            >>> temp2 = scheduler.step()  # 0.99
            >>> temp3 = scheduler.step()  # 0.9801
        """
        # Warmup: stay at initial temperature
        if self.epoch < self.warmup_epochs:
            self.epoch += 1
            return self.initial

        # Exponential decay with floor
        self.current_temp = max(
            self.final,
            self.initial * (self.decay ** (self.epoch - self.warmup_epochs))
        )

        self.epoch += 1
        return self.current_temp

    def get_temperature(self) -> float:
        """Get current temperature without advancing epoch.

        Returns:
            float: Current temperature

        Example:
            >>> scheduler = TemperatureScheduler()
            >>> temp = scheduler.get_temperature()  # 1.0
            >>> temp = scheduler.get_temperature()  # Still 1.0 (no step)
        """
        return self.current_temp

    def set_temperature(self, temp: float) -> None:
        """Manually set temperature (for debugging/evaluation).

        Args:
            temp (float): Temperature to set
        """
        assert temp > 0, "Temperature must be positive"
        self.current_temp = temp

    def reset(self) -> None:
        """Reset to initial temperature and epoch 0.

        Useful for restarting training or multiple runs.

        Example:
            >>> scheduler.step()  # epoch=1, temp=0.9999
            >>> scheduler.reset()
            >>> scheduler.get_temperature()  # 1.0 (reset)
        """
        self.current_temp = self.initial
        self.epoch = 0

    def state_dict(self) -> dict:
        """Get scheduler state for checkpointing.

        Returns:
            dict: State dictionary

        Example:
            >>> state = scheduler.state_dict()
            >>> # Save to checkpoint
            >>> torch.save({'scheduler': state}, 'checkpoint.pt')
        """
        return {
            'initial': self.initial,
            'final': self.final,
            'decay': self.decay,
            'warmup_epochs': self.warmup_epochs,
            'current_temp': self.current_temp,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint.

        Args:
            state_dict (dict): State dictionary from state_dict()

        Example:
            >>> checkpoint = torch.load('checkpoint.pt')
            >>> scheduler.load_state_dict(checkpoint['scheduler'])
        """
        self.initial = state_dict['initial']
        self.final = state_dict['final']
        self.decay = state_dict['decay']
        self.warmup_epochs = state_dict.get('warmup_epochs', 0)
        self.current_temp = state_dict['current_temp']
        self.epoch = state_dict['epoch']

    def __repr__(self) -> str:
        return (f"TemperatureScheduler(initial={self.initial:.3f}, "
                f"final={self.final:.3f}, current={self.current_temp:.3f}, "
                f"epoch={self.epoch})")


class AdaptiveTemperatureScheduler(TemperatureScheduler):
    """Adaptive temperature scheduling based on validation metrics.

    Reduces temperature only when validation performance improves,
    preventing premature sharpening.

    Args:
        initial_temp (float): Starting temperature
        final_temp (float): Minimum temperature
        decay_rate (float): Decay rate when metric improves
        patience (int): Epochs without improvement before reducing temp
        metric_mode (str): 'min' or 'max' (whether lower/higher is better)

    Example:
        >>> scheduler = AdaptiveTemperatureScheduler(patience=5, metric_mode='min')
        >>>
        >>> for epoch in range(100):
        ...     val_loss = validate()
        ...     temp = scheduler.step(val_loss)
        ...     # Temperature only decreases when val_loss improves
    """

    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.3,
        decay_rate: float = 0.95,
        patience: int = 10,
        metric_mode: str = 'min'
    ):
        super().__init__(initial_temp, final_temp, decay_rate)

        self.patience = patience
        self.metric_mode = metric_mode

        self.best_metric = float('inf') if metric_mode == 'min' else float('-inf')
        self.epochs_without_improvement = 0

    def step(self, metric: Optional[float] = None) -> float:
        """Step based on validation metric.

        Args:
            metric (float, optional): Validation metric value.
                If None, behaves like standard scheduler.

        Returns:
            float: Current temperature
        """
        if metric is None:
            return super().step()

        # Check if metric improved
        improved = False
        if self.metric_mode == 'min':
            improved = metric < self.best_metric
        else:
            improved = metric > self.best_metric

        if improved:
            self.best_metric = metric
            self.epochs_without_improvement = 0

            # Reduce temperature on improvement
            if self.current_temp > self.final:
                self.current_temp = max(
                    self.final,
                    self.current_temp * self.decay
                )
        else:
            self.epochs_without_improvement += 1

        self.epoch += 1
        return self.current_temp

    def __repr__(self) -> str:
        return (f"AdaptiveTemperatureScheduler(current={self.current_temp:.3f}, "
                f"best_metric={self.best_metric:.4f}, "
                f"epochs_without_improvement={self.epochs_without_improvement})")
