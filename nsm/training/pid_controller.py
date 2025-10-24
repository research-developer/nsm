"""
PID Controller for adaptive hyperparameter tuning.

Implements proportional-integral-derivative control with anti-windup
to replace fixed-increment adaptation in AdaptivePhysicsTrainer.

Based on Control Theory isomorphism (analysis/additional_isomorphisms.md).
"""

import torch
from typing import Optional


class PIDController:
    """
    PID controller with anti-windup for neural training control.

    Standard PID equation:
        u(t) = Kp × e(t) + Ki × ∫e(τ)dτ + Kd × de/dt

    Where:
        - e(t) = error signal (setpoint - measurement)
        - Kp = proportional gain (immediate response)
        - Ki = integral gain (accumulated error correction)
        - Kd = derivative gain (rate-of-change damping)

    Anti-windup prevents integral term from accumulating when output saturates.

    Example usage:
        >>> pid = PIDController(Kp=0.1, Ki=0.01, Kd=0.05, output_limits=(0, 0.5))
        >>> error = 1.0 - q_neural  # Target q=1.0
        >>> adjustment = pid.update(error, dt=1.0)
        >>> diversity_weight = diversity_weight + adjustment

    Control theory mapping:
        - Plant: Neural network (class balance dynamics)
        - Setpoint: Target balance ψ = 1.0
        - Measurement: Current balance ψ(t)
        - Control input: Diversity weight adjustment
        - Disturbance: Stochastic gradients

    Tuning guidelines:
        - Kp: Higher → faster response, more overshoot
        - Ki: Higher → eliminates steady-state error, may cause oscillation
        - Kd: Higher → reduces overshoot, dampens oscillations
        - Optimal damping ratio: ζ ≈ 1.0 (critically damped)

    Args:
        Kp: Proportional gain (default: 0.1)
        Ki: Integral gain (default: 0.01)
        Kd: Derivative gain (default: 0.05)
        output_limits: (min, max) tuple for output clamping (default: (0, 0.5))
        integral_limit: Maximum absolute value for integral term (anti-windup)

    Attributes:
        integral: Accumulated integral term ∫e(τ)dτ
        prev_error: Previous error for derivative calculation
        prev_time: Previous update time for dt calculation
    """

    def __init__(
        self,
        Kp: float = 0.1,
        Ki: float = 0.01,
        Kd: float = 0.05,
        output_limits: tuple[float, float] = (0.0, 0.5),
        integral_limit: Optional[float] = None
    ):
        # PID gains
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        # Output saturation limits
        self.output_min, self.output_max = output_limits

        # Anti-windup: Limit integral accumulation
        if integral_limit is None:
            # Default: integral can contribute up to half of max output
            self.integral_limit = (self.output_max - self.output_min) / (2 * self.Ki) if self.Ki > 0 else float('inf')
        else:
            self.integral_limit = integral_limit

        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

        # Tracking for diagnostics
        self.history = {
            'error': [],
            'proportional': [],
            'integral': [],
            'derivative': [],
            'output': [],
            'saturated': []
        }

    def update(self, error: float, dt: float = 1.0) -> float:
        """
        Compute PID control output given current error.

        Args:
            error: Current error signal (setpoint - measurement)
                   For balance control: error = 1.0 - ψ(t)
                   where ψ = 1 - |acc₀ - acc₁|
            dt: Time step since last update (default: 1.0 epoch)

        Returns:
            Control output (adjustment to apply to hyperparameter)
        """
        # Proportional term: Immediate response to current error
        proportional = self.Kp * error

        # Integral term: Accumulated error over time
        # Anti-windup: Clamp integral to prevent runaway
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        integral = self.Ki * self.integral

        # Derivative term: Rate of change of error (dampens oscillations)
        if self.prev_time is not None:
            derivative = self.Kd * (error - self.prev_error) / dt
        else:
            derivative = 0.0  # First call: no history for derivative

        # Combined PID output
        output = proportional + integral + derivative

        # Apply output limits (saturation)
        output_clamped = max(self.output_min, min(self.output_max, output))
        saturated = (output_clamped != output)

        # Anti-windup: Reset integral if output saturated
        if saturated and self.Ki > 0:
            # Back-calculate what integral should be to not saturate
            # output = Kp*e + Ki*I + Kd*d  =>  I = (output_clamped - Kp*e - Kd*d) / Ki
            self.integral = (output_clamped - proportional - derivative) / self.Ki

        # Update state for next iteration
        self.prev_error = error
        self.prev_time = (self.prev_time or 0) + dt

        # Record history for diagnostics
        self.history['error'].append(error)
        self.history['proportional'].append(proportional)
        self.history['integral'].append(integral)
        self.history['derivative'].append(derivative)
        self.history['output'].append(output_clamped)
        self.history['saturated'].append(saturated)

        return output_clamped

    def reset(self):
        """Reset controller state (integral, derivative history)."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.history = {
            'error': [],
            'proportional': [],
            'integral': [],
            'derivative': [],
            'output': [],
            'saturated': []
        }

    def get_diagnostics(self) -> dict:
        """
        Get diagnostic information about controller behavior.

        Returns:
            Dictionary with:
                - current_state: {integral, prev_error}
                - gains: {Kp, Ki, Kd}
                - limits: {output_min, output_max, integral_limit}
                - history: {error, proportional, integral, derivative, output, saturated}
                - metrics: {total_integral, max_error, saturation_fraction}
        """
        saturation_fraction = (sum(self.history['saturated']) / len(self.history['saturated'])
                               if self.history['saturated'] else 0.0)

        return {
            'current_state': {
                'integral': self.integral,
                'prev_error': self.prev_error
            },
            'gains': {
                'Kp': self.Kp,
                'Ki': self.Ki,
                'Kd': self.Kd
            },
            'limits': {
                'output_min': self.output_min,
                'output_max': self.output_max,
                'integral_limit': self.integral_limit
            },
            'history': self.history,
            'metrics': {
                'total_integral': self.integral,
                'max_error': max(abs(e) for e in self.history['error']) if self.history['error'] else 0.0,
                'saturation_fraction': saturation_fraction
            }
        }

    def tune_gains(self, Kp: Optional[float] = None, Ki: Optional[float] = None, Kd: Optional[float] = None):
        """
        Update PID gains without resetting state.

        Useful for adaptive tuning or manual calibration.

        Args:
            Kp: New proportional gain (None = keep current)
            Ki: New integral gain (None = keep current)
            Kd: New derivative gain (None = keep current)
        """
        if Kp is not None:
            self.Kp = Kp
        if Ki is not None:
            self.Ki = Ki
            # Adjust integral limit if Ki changes
            if Ki > 0:
                self.integral_limit = (self.output_max - self.output_min) / (2 * Ki)
        if Kd is not None:
            self.Kd = Kd


# Export public API
__all__ = ['PIDController']
