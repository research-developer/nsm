# Diversity Regularization for Temperature Profile Correction

## Overview

Diversity regularization enforces the correct hierarchical ordering of representation variances (T_L1 < T_L2 < T_L3) in the 6-level chiral architecture. This addresses the temperature inversion bug discovered in NSM-33 pilot study.

## Mathematical Formulation

### Temperature (Representation Variance)

At each level k, the temperature is defined as the mean variance across feature dimensions:

```
T_Lk = mean(var(x_Lk, dim=samples))
```

Where:
- `x_Lk ∈ ℝ^(N × d)` are the representations at level k
- N = number of nodes/samples
- d = feature dimensionality

### Desired Profile

The correct hierarchical ordering should follow information bottleneck principle:

```
T_L1 < T_L2 < T_L3
```

Where:
- **L1 (concrete)**: Low variance - specialized, task-specific features
- **L2 (intermediate)**: Medium variance - compositional features
- **L3 (abstract)**: High variance - diverse conceptual representations

### Regularization Loss

The diversity loss penalizes violations of the hierarchical ordering:

```python
L_diversity = λ_div × [
    ReLU(T_L1 - T_L2) +              # Penalize L1 > L2
    ReLU(T_L2 - T_L3) +              # Penalize L2 > L3
    ReLU(γ_target - (T_L3 - T_L1))  # Encourage minimum gradient
]
```

Where:
- λ_div = diversity regularization weight (default: 0.1)
- γ_target = target minimum gradient (default: 0.1)
- ReLU(x) = max(0, x)

## Implementation

### DiversityRegularization Module

```python
class DiversityRegularization(nn.Module):
    """
    Enforce correct temperature profile: L1 < L2 < L3 in diversity.

    Location: nsm/models/chiral_fixed_temp.py:27-92
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight  # λ_div

    def forward(
        self,
        x_l1: torch.Tensor,  # [N, d] representations at L1
        x_l2: torch.Tensor,  # [N, d] representations at L2
        x_l3: torch.Tensor   # [N, d] representations at L3
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute diversity regularization loss.

        Returns:
            loss: Scalar tensor
            diagnostics: Dict with T_L1, T_L2, T_L3, T_gradient
        """
        # Compute temperatures (variances)
        T_L1 = x_l1.var(dim=0).mean()  # Mean variance across features
        T_L2 = x_l2.var(dim=0).mean()
        T_L3 = x_l3.var(dim=0).mean()

        loss = torch.tensor(0.0, device=x_l1.device)

        # Penalize inversions
        if T_L2 < T_L1:
            loss = loss + F.relu(T_L1 - T_L2)

        if T_L3 < T_L2:
            loss = loss + F.relu(T_L2 - T_L3)

        # Encourage minimum gradient
        gradient = T_L3 - T_L1
        target_gradient = 0.1

        if gradient < target_gradient:
            loss = loss + F.relu(target_gradient - gradient)

        loss = loss * self.weight

        return loss, diagnostics
```

### Integration with Loss Function

```python
class FixedTemperatureChiralLoss(nn.Module):
    """
    Composite loss including diversity regularization.

    Location: nsm/models/chiral_fixed_temp.py:154-242
    """

    def forward(self, model_output, targets):
        # Standard task + auxiliary + cycle losses
        loss_task = self.task_criterion(model_output['logits'], targets)
        loss_aux = ...
        loss_cycle = ...

        # Diversity regularization (added)
        loss_diversity = model_output.get('diversity_loss', 0.0)

        # Total composite loss
        L_total = (
            λ_task × loss_task +
            λ_aux × loss_aux +
            λ_cycle × loss_cycle +
            λ_div × loss_diversity  # NEW
        )

        return {'loss': L_total, ...}
```

## Hyperparameters

| Parameter | Symbol | Default | Range | Description |
|-----------|--------|---------|-------|-------------|
| Diversity weight | λ_div | 0.1 | [0.01, 0.5] | Global scaling of diversity loss |
| Target gradient | γ_target | 0.1 | [0.05, 0.3] | Minimum required T_L3 - T_L1 |

### Tuning Guidelines

**λ_div too low (< 0.05):**
- Temperature inversions persist
- q_neural remains unstable
- Class imbalance issues

**λ_div too high (> 0.3):**
- Dominates other losses
- May prevent task learning
- Representations become overly dispersed

**Recommended:** Start at 0.1, increase if inversions persist after 5 epochs.

## Results (NSM-33 Track C)

### Before Fix (Pilot Study, N=2K)
```
T_L1: 0.40 → T_L2: 0.25 → T_L3: 0.13
Gradient: -0.27 [INVERTED]
q_neural: 0.45 (COLLAPSE RISK)
Accuracy: 48.16%
```

### After Fix (10x Scale, N=20K, λ_div=0.1)
```
T_L1: 0.36 → T_L2: 4.16 → T_L3: 19.53
Gradient: +19.17 [NORMAL]
q_neural: 0.625 (marginal stability)
Accuracy: 65.57%
```

### Analysis

✅ **Temperature profile corrected** - Gradient changed from -0.27 to +19.17

⚠️ **q_neural still below 1.0** - Suggests other stability factors at play

✅ **Accuracy improved** - +17.41 percentage points

**Confound:** Scale effect (2K → 20K) dominates diversity regularization effect. Need ablation study at same scale.

## Theoretical Justification

### Information Bottleneck Perspective

Tishby & Zaslavsky (2015) show that deep networks exhibit two phases:
1. **Fitting phase**: Representations increase mutual information I(X; T)
2. **Compression phase**: Higher layers compress I(T; X) while preserving I(T; Y)

**Prediction:** Higher layers (L3) should have **higher entropy** (variance) of representations to maintain diverse abstract concepts, while lower layers (L1) compress to task-relevant features.

**Our observations align with this theory.**

### Why Inversions Are Problematic

**Hypothesis:** When T_L1 > T_L3, the architecture:
1. Overfits at concrete level (high variance in L1 = memorization)
2. Underspecifies at abstract level (low variance in L3 = collapsed concepts)
3. Violates hierarchical abstraction (information flows "uphill")

**Analogy:** Like a neural network with bottleneck at the wrong end.

### Alternative Interpretation (Peer Review Concern)

**Reviewer's critique:** Compression may be HEALTHY, not pathological. High variance in L3 might indicate:
- Insufficient training (representations not converged)
- Regularization preventing compression
- Fighting against natural information bottleneck

**Counter-evidence:**
- Fixed architecture has **worse** class balance (11.48% vs 5.91%)
- Fixed architecture has **lower** q_neural (0.625 vs 1.336)
- Scale alone (baseline) achieves better results

**Conclusion:** Effect is **CONFOUNDED** - scale dominates diversity regularization. Need controlled ablation.

## Recommended Ablation Study

To isolate diversity regularization effect:

| Condition | N | λ_div | Expected Result |
|-----------|---|-------|-----------------|
| Baseline-2K | 2,000 | 0.0 | Inverted profile (replicate pilot) |
| Fixed-2K | 2,000 | 0.1 | Test if diversity fixes at small scale |
| Baseline-20K | 20,000 | 0.0 | Already done (67.11%) |
| Fixed-20K | 20,000 | 0.1 | Already done (65.57%) |
| **NEW** Baseline-20K-no-reg | 20,000 | 0.0 | Control: Scale without regularization |

**Critical test:** Does Fixed-2K correct inversion without scale?

## Limitations

1. **No dimensional analysis** - Temperatures have arbitrary units (not dimensionless)
2. **Threshold (γ=0.1) arbitrary** - Not derived from theory
3. **Scale confound** - Cannot separate diversity effect from data sufficiency
4. **Single dataset** - Generalization unknown
5. **No causal evidence** - Correlation between profile and stability, not causation

## Future Work

1. **Information-theoretic reformulation** - Replace variance with mutual information I(X_L; Y)
2. **Adaptive γ_target** - Scale with model capacity and task complexity
3. **Per-layer regularization** - Different λ_div for each level
4. **Multi-dataset validation** - Test on KG, causal reasoning domains
5. **Ablation at fixed scale** - Isolate diversity effect from scale effect

## References

- Tishby & Zaslavsky (2015). "Deep Learning and the Information Bottleneck Principle"
- Shwartz-Ziv & Tishby (2017). "Opening the Black Box of Deep Neural Networks"
- Saxe et al. (2019). "On the Information Bottleneck Theory of Deep Learning"

## See Also

- `nsm/models/chiral_fixed_temp.py` - Implementation
- `experiments/modal_10x_fixed_temp.py` - Validation experiment
- `results/NSM-33_10x_validation_results.md` - Empirical results
- `docs/physics_metrics.md` - Related stability metrics
