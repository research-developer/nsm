# NSM-34 Implementation Guide: Conway Operators for Neural Collapse

**Date**: 2025-10-23
**Status**: Implementation guide for pre-registered study (NSM-34-CGT-OPERATORS-PREREG.md)
**Audience**: Implementers, code reviewers, future researchers

---

## Quick Start

This guide provides concrete code implementations for all 5 Conway operators identified in the pre-registration. Each section includes:
- Mathematical definition
- Concrete PyTorch implementation
- Usage examples
- Edge cases and optimizations

---

## Operator 1: Conway Temperature

### Mathematical Definition

For partizan game G = {GL | GR}:

```
t(G) = (max_Left(GL) - min_Right(GR)) / 2
```

In neural networks:
- **Left player (WHY)**: Abstraction via pooling
- **Right player (WHAT)**: Concretization via unpooling
- **Temperature**: Asymmetry in reconstruction quality

### Implementation

```python
import torch
import torch.nn as nn
from typing import Tuple, Dict

def temperature_conway(
    model: nn.Module,
    x: torch.Tensor,
    num_samples: int = 10,
    metric: str = 'mse'
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Conway temperature for neural WHY/WHAT game.

    Args:
        model: Model with .why() and .what() methods (e.g., SymmetricHierarchicalLayer)
        x: Input tensor [batch_size, features]
        num_samples: Number of Monte Carlo samples for max/min
        metric: 'mse' or 'cosine' for reconstruction quality

    Returns:
        Tuple of (temperature, diagnostics_dict)
        - temperature: Conway temperature t(x)
        - diagnostics: left_max, right_min, mean_reconstruction

    Example:
        >>> model = FullChiralModel(...)
        >>> x = torch.randn(32, 64)
        >>> temp, diag = temperature_conway(model, x)
        >>> if temp < 0.2:
        ...     print("Warning: Game too cold, collapse risk!")
    """
    model.eval()
    with torch.no_grad():
        # Compute abstraction
        x_abstract = model.why(x)

        # Left player moves: WHY then WHAT (abstraction → concretization)
        left_scores = []
        for _ in range(num_samples):
            x_recon_left = model.what(x_abstract)
            if metric == 'mse':
                score = -torch.mean((x_recon_left - x) ** 2).item()  # Negative MSE (higher better)
            elif metric == 'cosine':
                score = torch.nn.functional.cosine_similarity(
                    x_recon_left.flatten(), x.flatten(), dim=0
                ).item()
            left_scores.append(score)

        # Right player moves: WHAT then WHY (concretization → abstraction)
        right_scores = []
        for _ in range(num_samples):
            # Note: In practice, WHAT(WHY(x)) and WHY(WHAT(x)) may differ
            # due to stochasticity or non-commutativity
            x_recon_right = model.what(x_abstract)  # Same operation, different interpretation
            if metric == 'mse':
                score = -torch.mean((x_recon_right - x) ** 2).item()
            elif metric == 'cosine':
                score = torch.nn.functional.cosine_similarity(
                    x_recon_right.flatten(), x.flatten(), dim=0
                ).item()
            right_scores.append(score)

        # Conway temperature: (max_Left - min_Right) / 2
        max_left = max(left_scores)
        min_right = min(right_scores)
        temperature = (max_left - min_right) / 2

        # Diagnostics
        diagnostics = {
            'temperature': temperature,
            'max_left': max_left,
            'min_right': min_right,
            'mean_left': sum(left_scores) / len(left_scores),
            'mean_right': sum(right_scores) / len(right_scores),
            'variance_left': torch.tensor(left_scores).var().item(),
            'variance_right': torch.tensor(right_scores).var().item()
        }

    return temperature, diagnostics


def temperature_trajectory(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    max_batches: int = 10
) -> list:
    """
    Compute temperature trajectory over multiple batches.

    Args:
        model: Model with WHY/WHAT
        dataloader: Data batches
        max_batches: Limit computation (temperature expensive)

    Returns:
        List of (temperature, diagnostics) tuples
    """
    temps = []
    for i, (x, _) in enumerate(dataloader):
        if i >= max_batches:
            break
        temp, diag = temperature_conway(model, x)
        temps.append((temp, diag))

    return temps
```

### Usage Example

```python
# In training loop
import matplotlib.pyplot as plt

temperature_history = []

for epoch in range(num_epochs):
    # Training step
    train_epoch(model, train_loader, optimizer)

    # Compute Conway temperature
    temp, diag = temperature_conway(model, val_batch)
    temperature_history.append(temp)

    # Early warning
    if temp < 0.2:
        print(f"⚠️  Epoch {epoch}: Conway temperature = {temp:.3f} < 0.2 (collapse risk!)")
        # Intervention: Increase diversity weight
        loss_fn.diversity_weight += 0.05

# Visualization
plt.plot(temperature_history)
plt.axhline(y=0.2, color='r', linestyle='--', label='Collapse threshold')
plt.xlabel('Epoch')
plt.ylabel('Conway Temperature t(x)')
plt.legend()
plt.title('Neural Game Temperature Trajectory')
plt.savefig('conway_temperature.png')
```

---

## Operator 2: Cooling Rate

### Mathematical Definition

Rate at which game approaches cold state:

```
cooling_rate(t) = temperature(t) - temperature(t-1)
```

For neural networks using α/β hinge parameters:

```
temp_neural(t) = |α(t) - 0.5| + |β(t) - 0.5|
cooling_rate(t) = temp_neural(t) - temp_neural(t-1)
```

Negative cooling rate → game cooling down → diversity loss

### Implementation

```python
from collections import deque
from typing import Optional

class CoolingMonitor:
    """
    Track cooling rate of neural game over time.

    Attributes:
        window_size: Number of epochs for moving average
        alpha_history: Deque of α values
        beta_history: Deque of β values
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.alpha_history = deque(maxlen=window_size)
        self.beta_history = deque(maxlen=window_size)
        self.temp_history = deque(maxlen=window_size)
        self.cooling_history = []

    def compute_temperature_neural(
        self,
        alpha: float,
        beta: float
    ) -> float:
        """
        Compute neural game temperature from hinge parameters.

        Temperature = distance from neutral (0.5).
        High temperature: α, β far from 0.5 (strong player advantage)
        Low temperature: α, β ≈ 0.5 (neutral, cold game)
        """
        return abs(alpha - 0.5) + abs(beta - 0.5)

    def update(
        self,
        alpha: float,
        beta: float
    ) -> Optional[float]:
        """
        Update cooling monitor with new hinge parameters.

        Returns:
            cooling_rate: Current cooling rate (None if insufficient history)
                          Negative = cooling down (collapse risk)
                          Positive = heating up (stable)
        """
        temp = self.compute_temperature_neural(alpha, beta)

        self.alpha_history.append(alpha)
        self.beta_history.append(beta)
        self.temp_history.append(temp)

        # Need at least 2 samples to compute rate
        if len(self.temp_history) < 2:
            return None

        # Cooling rate: current - previous
        cooling_rate = self.temp_history[-1] - self.temp_history[-2]
        self.cooling_history.append(cooling_rate)

        return cooling_rate

    def get_smoothed_cooling_rate(self) -> Optional[float]:
        """
        Get moving average of cooling rate.

        Returns:
            Smoothed cooling rate over window
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
        Predict number of epochs until temperature reaches threshold.

        Assumes linear cooling rate (conservative estimate).

        Returns:
            epochs_remaining: Estimated epochs until collapse
                              None if heating or insufficient data
        """
        cooling_rate = self.get_smoothed_cooling_rate()

        if cooling_rate is None or cooling_rate >= 0:
            return None  # Heating or no data

        if current_temp is None:
            current_temp = self.temp_history[-1]

        if current_temp <= threshold_temp:
            return 0  # Already below threshold

        # Linear extrapolation: T(t + Δt) = T(t) + cooling_rate * Δt
        # Solve: threshold = current + cooling_rate * Δt
        epochs_remaining = (threshold_temp - current_temp) / cooling_rate

        return int(max(0, epochs_remaining))
```

### Usage Example

```python
# Initialize monitor
cooling_monitor = CoolingMonitor(window_size=5)

for epoch in range(num_epochs):
    train_epoch(model, train_loader, optimizer)

    # Extract hinge parameters
    alpha = extract_hinge_parameter(model, 'alpha')  # See helper below
    beta = extract_hinge_parameter(model, 'beta')

    # Update cooling monitor
    cooling_rate = cooling_monitor.update(alpha, beta)

    if cooling_rate is not None:
        # Check for rapid cooling
        if cooling_rate < -0.05:
            print(f"⚠️  Epoch {epoch}: Rapid cooling detected (rate={cooling_rate:.4f})")

            # Predict collapse time
            epochs_until_collapse = cooling_monitor.predict_collapse_time()
            if epochs_until_collapse is not None and epochs_until_collapse < 3:
                print(f"   Collapse predicted in {epochs_until_collapse} epochs!")

                # Intervention: Heat up the game (increase asymmetry)
                # This is counterintuitive but prevents premature cooling
                for name, module in model.named_modules():
                    if hasattr(module, 'alpha'):
                        # Push α away from 0.5
                        with torch.no_grad():
                            module.alpha.data += 0.1 * torch.sign(module.alpha.data - 0.5)

# Helper function
def extract_hinge_parameter(model: nn.Module, param_name: str) -> float:
    """Extract mean hinge parameter value from model."""
    values = []
    for name, module in model.named_modules():
        if 'hinge' in name.lower():
            if hasattr(module, param_name):
                param = getattr(module, param_name)
                values.append(torch.sigmoid(param).mean().item())

    return sum(values) / len(values) if values else 0.5
```

---

## Operator 3: Confusion Interval

### Mathematical Definition

For game G:

```
[G_L, G_R] = [pessimistic_value, optimistic_value]
width = G_R - G_L  (epistemic uncertainty)
```

In neural networks:
- **G_L**: Worst-case reconstruction (WHY loses information)
- **G_R**: Best-case reconstruction (WHAT adds information)
- **Width**: How much outcome depends on player choice

### Implementation

```python
from typing import Tuple
import numpy as np

def confusion_interval(
    model: nn.Module,
    x: torch.Tensor,
    num_samples: int = 100,
    confidence_metric: str = 'cycle_loss'
) -> Tuple[float, float, float, Dict[str, any]]:
    """
    Compute Conway confusion interval for neural game.

    Args:
        model: Model with WHY/WHAT
        x: Input tensor
        num_samples: Monte Carlo samples
        confidence_metric: 'cycle_loss', 'mse', or 'cosine'

    Returns:
        (c_L, c_R, width, diagnostics)
        - c_L: Pessimistic confidence (Left player worst case)
        - c_R: Optimistic confidence (Right player best case)
        - width: Epistemic uncertainty (how confused is the game?)
        - diagnostics: Distribution statistics
    """
    model.eval()
    with torch.no_grad():
        # Compute WHY(WHAT) cycle
        x_abstract = model.why(x)

        # Sample multiple reconstructions
        scores = []
        for _ in range(num_samples):
            x_recon = model.what(x_abstract)

            if confidence_metric == 'cycle_loss':
                # Cycle consistency (lower is better)
                score = 1.0 - torch.mean((x_recon - x) ** 2).item()  # Invert to "confidence"
            elif confidence_metric == 'mse':
                score = -torch.mean((x_recon - x) ** 2).item()
            elif confidence_metric == 'cosine':
                score = torch.nn.functional.cosine_similarity(
                    x_recon.flatten(), x.flatten(), dim=0
                ).item()

            scores.append(score)

        # Confusion interval
        c_L = min(scores)  # Pessimistic (worst reconstruction)
        c_R = max(scores)  # Optimistic (best reconstruction)
        width = c_R - c_L  # Epistemic uncertainty

        # Diagnostics
        diagnostics = {
            'c_L': c_L,
            'c_R': c_R,
            'width': width,
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'q25': np.percentile(scores, 25),
            'q75': np.percentile(scores, 75),
            'samples': scores  # Full distribution
        }

    return c_L, c_R, width, diagnostics


def confusion_width_trajectory(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = 50,
    max_batches: int = 5
) -> list:
    """
    Track confusion width over multiple batches.

    Returns:
        List of (width, diagnostics) tuples
    """
    widths = []
    for i, (x, _) in enumerate(dataloader):
        if i >= max_batches:
            break

        c_L, c_R, width, diag = confusion_interval(model, x, num_samples)
        widths.append((width, diag))

    return widths


def stability_prediction(
    confusion_history: list,
    window: int = 3,
    threshold: float = 0.3
) -> Tuple[bool, str]:
    """
    Predict stability based on confusion width trend.

    Args:
        confusion_history: List of confusion widths
        window: Lookback window
        threshold: Width above which instability predicted

    Returns:
        (is_stable, reason)
    """
    if len(confusion_history) < window:
        return True, "Insufficient history"

    recent_widths = confusion_history[-window:]
    mean_width = sum(recent_widths) / len(recent_widths)

    if mean_width > threshold:
        return False, f"High confusion width ({mean_width:.3f} > {threshold})"

    # Check for increasing trend
    if all(recent_widths[i] < recent_widths[i+1] for i in range(len(recent_widths)-1)):
        return False, f"Rapidly increasing confusion"

    return True, "Stable confusion"
```

### Usage Example

```python
confusion_widths = []

for epoch in range(num_epochs):
    train_epoch(model, train_loader, optimizer)

    # Compute confusion interval
    c_L, c_R, width, diag = confusion_interval(
        model, val_batch, num_samples=50
    )
    confusion_widths.append(width)

    # Stability check
    if epoch >= 3:
        is_stable, reason = stability_prediction(confusion_widths, window=3)

        if not is_stable:
            print(f"⚠️  Epoch {epoch}: Instability predicted - {reason}")
            print(f"   Confusion: [{c_L:.3f}, {c_R:.3f}], width={width:.3f}")

            # Intervention: Reduce learning rate (tighten confusion)
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot confusion width trajectory
ax1.plot(confusion_widths, marker='o')
ax1.axhline(y=0.3, color='r', linestyle='--', label='Instability threshold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Confusion Width')
ax1.set_title('Epistemic Uncertainty Trajectory')
ax1.legend()

# Plot distribution at final epoch (if diagnostics saved)
final_diag = diag
ax2.hist(final_diag['samples'], bins=30, alpha=0.7)
ax2.axvline(x=c_L, color='b', linestyle='--', label=f'c_L={c_L:.3f}')
ax2.axvline(x=c_R, color='r', linestyle='--', label=f'c_R={c_R:.3f}')
ax2.set_xlabel('Reconstruction Score')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Confusion Distribution (Epoch {epoch})')
ax2.legend()

plt.tight_layout()
plt.savefig('confusion_analysis.png')
```

---

## Operator 4: Game Addition (Non-Commutative)

### Mathematical Definition

```
G + H = {GL + H, G + HL | GR + H, G + HR}
```

**Key property**: G + H ≠ H + G (order matters)

In neural training:
- G = train on class 0 first
- H = train on class 1 first
- G + H ≠ H + G captures hysteresis

### Implementation

```python
import copy
from typing import Dict, Tuple

def game_addition_neural(
    model: nn.Module,
    data_A: torch.utils.data.DataLoader,
    data_B: torch.utils.data.DataLoader,
    num_epochs_per_game: int = 5,
    optimizer_factory: callable = None
) -> Dict[str, any]:
    """
    Measure non-commutativity in neural training (game addition).

    Trains two identical models with reversed data order:
    - Model AB: Train on A, then train on B
    - Model BA: Train on B, then train on A

    Args:
        model: Initial model (will be copied)
        data_A: First dataset (e.g., class 0 samples)
        data_B: Second dataset (e.g., class 1 samples)
        num_epochs_per_game: Epochs for each "game"
        optimizer_factory: Function returning fresh optimizer

    Returns:
        Dictionary with:
            - 'outcome_AB': Final accuracy for A→B order
            - 'outcome_BA': Final accuracy for B→A order
            - 'commutativity_gap': |AB - BA| (non-commutativity measure)
            - 'trajectory_AB': Epoch-wise metrics for AB
            - 'trajectory_BA': Epoch-wise metrics for BA
    """
    if optimizer_factory is None:
        optimizer_factory = lambda m: torch.optim.Adam(m.parameters(), lr=1e-3)

    # Model for order A → B
    model_AB = copy.deepcopy(model)
    optimizer_AB = optimizer_factory(model_AB)

    # Model for order B → A
    model_BA = copy.deepcopy(model)
    optimizer_BA = optimizer_factory(model_BA)

    trajectory_AB = []
    trajectory_BA = []

    # Game G (train on A)
    print("Training AB: Game G (dataset A)...")
    for epoch in range(num_epochs_per_game):
        metrics = train_epoch(model_AB, data_A, optimizer_AB)
        trajectory_AB.append(metrics)

    # Game H (train on B)
    print("Training AB: Game H (dataset B)...")
    for epoch in range(num_epochs_per_game):
        metrics = train_epoch(model_AB, data_B, optimizer_AB)
        trajectory_AB.append(metrics)

    # Game H (train on B first)
    print("Training BA: Game H (dataset B)...")
    for epoch in range(num_epochs_per_game):
        metrics = train_epoch(model_BA, data_B, optimizer_BA)
        trajectory_BA.append(metrics)

    # Game G (train on A second)
    print("Training BA: Game G (dataset A)...")
    for epoch in range(num_epochs_per_game):
        metrics = train_epoch(model_BA, data_A, optimizer_BA)
        trajectory_BA.append(metrics)

    # Evaluate final outcomes
    outcome_AB = evaluate_model(model_AB, test_loader)
    outcome_BA = evaluate_model(model_BA, test_loader)

    # Commutativity gap
    commutativity_gap = abs(outcome_AB - outcome_BA)

    results = {
        'outcome_AB': outcome_AB,
        'outcome_BA': outcome_BA,
        'commutativity_gap': commutativity_gap,
        'trajectory_AB': trajectory_AB,
        'trajectory_BA': trajectory_BA,
        'is_commutative': commutativity_gap < 0.01  # Threshold for "approximately commutative"
    }

    return results


def hysteresis_loop_experiment(
    model: nn.Module,
    data_full: torch.utils.data.DataLoader,
    diversity_schedule: list,
    optimizer_factory: callable
) -> Dict[str, any]:
    """
    Test hysteresis by varying diversity weight up then down.

    Args:
        model: Model to train
        data_full: Full dataset
        diversity_schedule: List of diversity weights (e.g., [0, 0.2, 0.4, 0.4, 0.2, 0])
        optimizer_factory: Optimizer factory

    Returns:
        Dictionary with trajectory and hysteresis area
    """
    model_copy = copy.deepcopy(model)
    optimizer = optimizer_factory(model_copy)

    trajectory = []

    for epoch, diversity_weight in enumerate(diversity_schedule):
        # Update loss function
        loss_fn.diversity_weight = diversity_weight

        # Train epoch
        metrics = train_epoch(model_copy, data_full, optimizer)
        metrics['diversity_weight'] = diversity_weight
        trajectory.append(metrics)

    # Compute hysteresis area (simplified)
    # Area between up-path and down-path in (diversity, balance) space
    midpoint = len(diversity_schedule) // 2
    up_path = trajectory[:midpoint]
    down_path = trajectory[midpoint:]

    # Integrate |balance_up(d) - balance_down(d)|
    hysteresis_area = 0
    for i in range(len(down_path)):
        balance_up = up_path[i]['class_balance']
        balance_down = down_path[i]['class_balance']
        hysteresis_area += abs(balance_up - balance_down)

    return {
        'trajectory': trajectory,
        'hysteresis_area': hysteresis_area,
        'has_hysteresis': hysteresis_area > 0.1  # Threshold
    }
```

### Usage Example

```python
# Prepare class-specific dataloaders
data_class_0 = filter_by_class(train_loader, class_id=0)
data_class_1 = filter_by_class(train_loader, class_id=1)

# Test non-commutativity
results = game_addition_neural(
    model=model,
    data_A=data_class_0,
    data_B=data_class_1,
    num_epochs_per_game=5
)

print(f"Outcome A→B: {results['outcome_AB']:.2f}%")
print(f"Outcome B→A: {results['outcome_BA']:.2f}%")
print(f"Commutativity gap: {results['commutativity_gap']:.2f}%")

if results['commutativity_gap'] > 5.0:
    print("⚠️  Strong non-commutativity detected (path dependence)")

# Visualize trajectories
plt.figure(figsize=(10, 6))
plt.plot([m['accuracy'] for m in results['trajectory_AB']], label='A→B', marker='o')
plt.plot([m['accuracy'] for m in results['trajectory_BA']], label='B→A', marker='s')
plt.axvline(x=5, color='k', linestyle='--', alpha=0.3, label='Game boundary')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title(f'Game Addition Non-Commutativity (Gap={results["commutativity_gap"]:.2f}%)')
plt.legend()
plt.savefig('game_addition.png')
```

---

## Operator 5: Surreal Classification

### Mathematical Definition

Surreal numbers extend ℝ:

```
ε = {0 | 1/2, 1/4, ...}  (infinitesimal)
ω = {1, 2, 3, ... | }    (infinity)
```

Neural collapse states:
- **0**: Stable equilibrium (balanced, robust)
- **ε**: Unstable equilibrium (nascent collapse)
- **1/2**: Moderate imbalance
- **1**: Active collapse
- **ω**: Irreversible collapse

### Implementation

```python
from enum import Enum
from typing import Tuple

class SurrealState(Enum):
    """Surreal number classification for neural collapse."""
    ZERO = "zero"          # Stable equilibrium
    EPSILON = "epsilon"    # Nascent collapse (unstable zero)
    HALF = "half"          # Moderate imbalance
    ONE = "one"            # Active collapse
    OMEGA = "omega"        # Irreversible collapse


def surreal_collapse_state(
    balance_delta: float,
    q_neural: float,
    temp_gradient: float,
    grad_norm: float,
    sensitivity: Optional[float] = None
) -> Tuple[SurrealState, str, Dict[str, float]]:
    """
    Classify neural collapse state using surreal number hierarchy.

    Args:
        balance_delta: Class imbalance |acc_0 - acc_1|
        q_neural: Safety factor (from NSM-33)
        temp_gradient: Temperature profile (T_L3 - T_L1)
        grad_norm: Gradient magnitude
        sensitivity: Perturbation sensitivity (optional, for epsilon detection)

    Returns:
        (state, explanation, diagnostics)
    """
    diagnostics = {
        'balance_delta': balance_delta,
        'q_neural': q_neural,
        'temp_gradient': temp_gradient,
        'grad_norm': grad_norm,
        'sensitivity': sensitivity
    }

    # ZERO: Stable equilibrium
    if balance_delta < 0.05:
        # Check stability indicators
        if q_neural >= 1.0 and temp_gradient > 0:
            return (
                SurrealState.ZERO,
                "Stable equilibrium (balanced, q>1, normal temp profile)",
                diagnostics
            )

        # EPSILON: Unstable equilibrium (nascent collapse)
        if sensitivity is not None and sensitivity > 10.0:
            return (
                SurrealState.EPSILON,
                f"Nascent collapse (balance near zero but sensitivity={sensitivity:.1f}x)",
                diagnostics
            )

        if q_neural < 1.0 or temp_gradient < -0.1:
            return (
                SurrealState.EPSILON,
                f"Nascent collapse (balance near zero but q={q_neural:.2f}<1 or inverted temp)",
                diagnostics
            )

    # HALF: Moderate imbalance
    if 0.05 <= balance_delta < 0.4:
        return (
            SurrealState.HALF,
            f"Moderate imbalance (Δ={balance_delta:.2f})",
            diagnostics
        )

    # ONE: Active collapse
    if 0.4 <= balance_delta < 0.7:
        return (
            SurrealState.ONE,
            f"Active collapse (Δ={balance_delta:.2f})",
            diagnostics
        )

    # OMEGA: Irreversible collapse
    if balance_delta >= 0.7:
        # Check for gradient death (infinitesimal gradients)
        if grad_norm < 1e-6:
            return (
                SurrealState.OMEGA,
                f"Irreversible collapse (Δ={balance_delta:.2f}, gradient death)",
                diagnostics
            )
        else:
            return (
                SurrealState.ONE,
                f"Severe collapse (Δ={balance_delta:.2f} but gradients exist, may recover)",
                diagnostics
            )

    # Fallback
    return (
        SurrealState.HALF,
        "Uncertain state",
        diagnostics
    )


def epsilon_sensitivity_test(
    model: nn.Module,
    x: torch.Tensor,
    perturbation_scale: float = 0.01,
    num_trials: int = 10
) -> float:
    """
    Measure sensitivity to perturbations (epsilon state detection).

    High sensitivity near zero balance → nascent collapse.

    Args:
        model: Neural network
        x: Input batch
        perturbation_scale: Noise magnitude
        num_trials: Number of perturbation samples

    Returns:
        sensitivity: Mean |Δbalance| / perturbation_scale
    """
    model.eval()
    with torch.no_grad():
        # Baseline balance
        baseline_balance = compute_class_balance(model, x)

        # Perturb and measure
        sensitivities = []
        for _ in range(num_trials):
            x_perturbed = x + torch.randn_like(x) * perturbation_scale
            perturbed_balance = compute_class_balance(model, x_perturbed)

            sensitivity = abs(perturbed_balance - baseline_balance) / perturbation_scale
            sensitivities.append(sensitivity)

    return sum(sensitivities) / len(sensitivities)


def compute_class_balance(model: nn.Module, x: torch.Tensor) -> float:
    """
    Compute class balance metric.

    Returns:
        balance: 1 - |acc_0 - acc_1| ∈ [0, 1]
    """
    logits = model(x)
    preds = logits.argmax(dim=1)

    acc_0 = (preds == 0).float().mean().item()
    acc_1 = (preds == 1).float().mean().item()

    return 1.0 - abs(acc_0 - acc_1)
```

### Usage Example

```python
surreal_history = []

for epoch in range(num_epochs):
    train_epoch(model, train_loader, optimizer)

    # Gather metrics
    balance_delta = compute_balance_delta(model, val_loader)
    q_neural, _ = compute_safety_factor(class_accuracies, model)
    temp_profile = compute_temperature_profile(level_representations)
    temp_gradient = temp_profile['T_gradient']
    grad_norm = compute_gradient_norm(model)

    # Epsilon sensitivity test (expensive, only if near-zero)
    sensitivity = None
    if balance_delta < 0.05:
        sensitivity = epsilon_sensitivity_test(model, val_batch)

    # Classify state
    state, explanation, diag = surreal_collapse_state(
        balance_delta, q_neural, temp_gradient, grad_norm, sensitivity
    )

    surreal_history.append({
        'epoch': epoch,
        'state': state,
        'explanation': explanation,
        **diag
    })

    # Epsilon warning
    if state == SurrealState.EPSILON:
        print(f"⚠️  Epoch {epoch}: EPSILON state detected!")
        print(f"   {explanation}")
        print(f"   Next epoch likely to show discrete collapse jump")

        # Intervention: Strengthen regularization
        loss_fn.diversity_weight += 0.1
        loss_fn.cycle_weight += 0.02

    # Omega state (irreversible)
    if state == SurrealState.OMEGA:
        print(f"🔴 Epoch {epoch}: OMEGA state (irreversible collapse)")
        print(f"   {explanation}")
        print(f"   Consider resetting model or aggressive intervention")

        # Nuclear option: Reset learning rate, reheat game
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr
        for module in model.modules():
            if hasattr(module, 'alpha'):
                module.alpha.data.uniform_(-1, 1)  # Randomize hinge parameters

# Visualization: Surreal state timeline
fig, ax = plt.subplots(figsize=(12, 6))

state_to_value = {
    SurrealState.ZERO: 0,
    SurrealState.EPSILON: 0.25,
    SurrealState.HALF: 0.5,
    SurrealState.ONE: 1.0,
    SurrealState.OMEGA: 1.5
}

epochs = [h['epoch'] for h in surreal_history]
states = [state_to_value[h['state']] for h in surreal_history]

ax.plot(epochs, states, marker='o', linewidth=2)
ax.set_yticks([0, 0.25, 0.5, 1.0, 1.5])
ax.set_yticklabels(['0 (Stable)', 'ε (Nascent)', '½ (Moderate)', '1 (Collapse)', 'ω (Irreversible)'])
ax.set_xlabel('Epoch')
ax.set_ylabel('Surreal Collapse State')
ax.set_title('Neural Collapse Surreal Number Classification')
ax.grid(axis='y', alpha=0.3)
plt.savefig('surreal_timeline.png')
```

---

## Integration: Composite Conway Score (CCS)

### Combined Predictor

```python
class ConwayCollapsePredictor:
    """
    Unified collapse predictor using all 5 Conway operators.

    Combines:
    1. Temperature (hot/cold)
    2. Cooling rate (dynamics)
    3. Confusion width (uncertainty)
    4. Surreal state (equilibrium type)
    5. Temperature gradient (hierarchy health)
    """

    def __init__(
        self,
        temp_threshold: float = 0.2,
        cooling_threshold: float = -0.05,
        confusion_threshold: float = 0.3,
        weights: Optional[Dict[str, float]] = None
    ):
        self.temp_threshold = temp_threshold
        self.cooling_threshold = cooling_threshold
        self.confusion_threshold = confusion_threshold

        # Default weights (can be learned via logistic regression)
        self.weights = weights or {
            'temperature': 0.25,
            'cooling': 0.20,
            'confusion': 0.20,
            'surreal': 0.20,
            'gradient': 0.15
        }

        # History for dynamic metrics
        self.cooling_monitor = CoolingMonitor(window_size=5)

    def predict(
        self,
        model: nn.Module,
        x: torch.Tensor,
        class_accuracies: Dict[str, float],
        level_representations: Dict[str, torch.Tensor],
        alpha: float,
        beta: float
    ) -> Tuple[float, Dict[str, any]]:
        """
        Compute Composite Conway Score (CCS).

        Returns:
            (ccs, diagnostics)
            - ccs ∈ [0, 1]: Stability score (1 = stable, 0 = collapse imminent)
            - diagnostics: All operator outputs
        """
        diagnostics = {}

        # 1. Temperature
        temp, temp_diag = temperature_conway(model, x, num_samples=10)
        temp_score = 1.0 if temp > self.temp_threshold else 0.0
        diagnostics['temperature'] = temp
        diagnostics['temp_score'] = temp_score

        # 2. Cooling rate
        cooling_rate = self.cooling_monitor.update(alpha, beta)
        if cooling_rate is not None:
            cooling_score = 1.0 if cooling_rate > self.cooling_threshold else 0.0
        else:
            cooling_score = 1.0  # No history, assume stable
        diagnostics['cooling_rate'] = cooling_rate
        diagnostics['cooling_score'] = cooling_score

        # 3. Confusion width
        c_L, c_R, width, conf_diag = confusion_interval(model, x, num_samples=50)
        confusion_score = 1.0 if width < self.confusion_threshold else 0.0
        diagnostics['confusion_width'] = width
        diagnostics['confusion_score'] = confusion_score

        # 4. Surreal state
        balance_delta = abs(
            class_accuracies['accuracy_class_0'] -
            class_accuracies['accuracy_class_1']
        )
        q_neural = compute_safety_factor(class_accuracies, model)[0]
        temp_gradient = level_representations.get('T_gradient', 0.0)
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

        state, explanation, surreal_diag = surreal_collapse_state(
            balance_delta, q_neural, temp_gradient, grad_norm
        )

        surreal_score = {
            SurrealState.ZERO: 1.0,
            SurrealState.EPSILON: 0.3,
            SurrealState.HALF: 0.6,
            SurrealState.ONE: 0.2,
            SurrealState.OMEGA: 0.0
        }[state]

        diagnostics['surreal_state'] = state
        diagnostics['surreal_explanation'] = explanation
        diagnostics['surreal_score'] = surreal_score

        # 5. Temperature gradient
        gradient_score = 1.0 if temp_gradient > 0 else 0.0
        diagnostics['temp_gradient'] = temp_gradient
        diagnostics['gradient_score'] = gradient_score

        # Composite score (weighted average)
        ccs = (
            self.weights['temperature'] * temp_score +
            self.weights['cooling'] * cooling_score +
            self.weights['confusion'] * confusion_score +
            self.weights['surreal'] * surreal_score +
            self.weights['gradient'] * gradient_score
        )

        diagnostics['ccs'] = ccs
        diagnostics['collapse_risk'] = 'HIGH' if ccs < 0.4 else ('MEDIUM' if ccs < 0.7 else 'LOW')

        return ccs, diagnostics
```

### Usage Example

```python
# Initialize predictor
predictor = ConwayCollapsePredictor()

for epoch in range(num_epochs):
    train_epoch(model, train_loader, optimizer)

    # Gather all required metrics
    class_accuracies = compute_class_accuracies(model, val_loader)
    level_representations = extract_level_representations(model, val_batch)
    alpha = extract_hinge_parameter(model, 'alpha')
    beta = extract_hinge_parameter(model, 'beta')

    # Predict collapse risk
    ccs, diagnostics = predictor.predict(
        model, val_batch, class_accuracies, level_representations, alpha, beta
    )

    print(f"Epoch {epoch}: CCS={ccs:.3f} ({diagnostics['collapse_risk']} risk)")

    # Detailed breakdown
    print(f"  Temperature: {diagnostics['temperature']:.3f} ({'✅' if diagnostics['temp_score'] else '❌'})")
    print(f"  Cooling rate: {diagnostics['cooling_rate']:.4f} ({'✅' if diagnostics['cooling_score'] else '❌'})")
    print(f"  Confusion: {diagnostics['confusion_width']:.3f} ({'✅' if diagnostics['confusion_score'] else '❌'})")
    print(f"  State: {diagnostics['surreal_state'].value} - {diagnostics['surreal_explanation']}")

    # Intervention based on CCS
    if ccs < 0.4:
        print("⚠️  HIGH COLLAPSE RISK - Initiating interventions")

        # Multi-pronged intervention
        loss_fn.diversity_weight += 0.1
        loss_fn.cycle_weight += 0.02
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9
```

---

## Performance Optimization Tips

### 1. Reduce Sampling Overhead

```python
# Adaptive sampling based on stability
def adaptive_num_samples(ccs: float, base_samples: int = 50) -> int:
    """
    Use fewer samples when stable, more when uncertain.
    """
    if ccs > 0.7:
        return base_samples // 2  # Stable, fewer samples needed
    elif ccs < 0.4:
        return base_samples * 2  # Unstable, need precision
    else:
        return base_samples
```

### 2. Compute Conway Metrics Infrequently

```python
# Only compute expensive metrics every N epochs
compute_conway = (epoch % 5 == 0) or (ccs < 0.5)

if compute_conway:
    ccs, diagnostics = predictor.predict(...)
else:
    # Use cached CCS
    pass
```

### 3. GPU Acceleration for Confusion Intervals

```python
# Vectorized confusion interval (all samples in parallel)
def confusion_interval_fast(model, x, num_samples=100):
    x_abstract = model.why(x)

    # Repeat abstract representation
    x_abstract_repeated = x_abstract.repeat(num_samples, 1, 1)

    # Single batch reconstruction
    x_recons = model.what(x_abstract_repeated)

    # Vectorized scoring
    x_repeated = x.repeat(num_samples, 1, 1)
    scores = -torch.mean((x_recons - x_repeated) ** 2, dim=[1, 2])

    return scores.min().item(), scores.max().item(), (scores.max() - scores.min()).item()
```

---

## Testing and Validation

### Unit Tests Template

```python
# tests/test_conway_operators.py

import pytest
import torch
from nsm.game_theory.conway_operators import (
    temperature_conway,
    cooling_rate,
    confusion_interval,
    surreal_collapse_state
)

def test_temperature_range():
    """Temperature should be non-negative."""
    model = MockModel()
    x = torch.randn(32, 64)

    temp, diag = temperature_conway(model, x)

    assert temp >= 0, "Temperature must be non-negative"
    assert diag['max_left'] >= diag['min_right'], "Left max should >= Right min"


def test_cooling_rate_sign():
    """Cooling rate should be negative when approaching 0.5."""
    # α, β moving toward 0.5
    alpha_prev, beta_prev = 0.8, 0.8
    alpha_curr, beta_curr = 0.6, 0.6

    rate = cooling_rate(alpha_curr, beta_curr, alpha_prev, beta_prev)

    assert rate < 0, "Should be cooling (negative rate)"


def test_confusion_width_bounds():
    """Confusion width should be in [0, 2]."""
    model = MockModel()
    x = torch.randn(32, 64)

    c_L, c_R, width, diag = confusion_interval(model, x)

    assert 0 <= width <= 2, f"Width {width} out of bounds"
    assert c_L <= c_R, "Left bound should be <= Right bound"


def test_surreal_state_transitions():
    """Test all surreal state transitions."""
    # Stable zero
    state, _, _ = surreal_collapse_state(0.01, 1.5, 0.2, 1e-3)
    assert state == SurrealState.ZERO

    # Epsilon (unstable zero)
    state, _, _ = surreal_collapse_state(0.01, 0.5, -0.2, 1e-3)
    assert state == SurrealState.EPSILON

    # Omega (gradient death)
    state, _, _ = surreal_collapse_state(0.8, 0.1, -0.3, 1e-8)
    assert state == SurrealState.OMEGA
```

---

## Summary: Implementation Checklist

- [ ] Implement `temperature_conway()` with Monte Carlo sampling
- [ ] Implement `CoolingMonitor` class for tracking α/β dynamics
- [ ] Implement `confusion_interval()` with epistemic uncertainty
- [ ] Implement `game_addition_neural()` for non-commutativity tests
- [ ] Implement `surreal_collapse_state()` classifier
- [ ] Integrate into `ConwayCollapsePredictor` unified system
- [ ] Add unit tests (12+ test cases)
- [ ] Profile computational overhead (target: <15% added time)
- [ ] Validate on NSM-33 pilot data (N=2,000)
- [ ] Compare to physics metrics (NSM-33 baseline: 85.7%)
- [ ] Document all hyperparameters (thresholds, weights)
- [ ] Create visualization suite (6+ plots)

**Estimated Implementation Time**: 5-7 days (one developer)

**Dependencies**: PyTorch, PyTorch Geometric, NSM physics metrics module

---

**END OF IMPLEMENTATION GUIDE**
