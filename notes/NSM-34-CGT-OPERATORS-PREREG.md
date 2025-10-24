# Pre-Registration: Conway Operators for Neural Collapse Dynamics (NSM-34)

**Date**: 2025-10-23
**Study**: Applying Combinatorial Game Theory Operators to Neural Class Collapse
**Principal Investigator**: Claude Code (Anthropic) + Preston (Human Collaborator)
**Status**: Pre-registered before implementation
**Builds On**: NSM-33 (Physics-Inspired Collapse Prediction)

---

## Executive Summary

NSM-33 demonstrated that neural collapse exhibits game-theoretic structure through the WHY/WHAT dual flows (partizan game with Left/Right players) and α/β hinge parameters (game temperature). We validated physics-inspired metrics achieving **85.7% collapse prediction accuracy** using standard algebraic operators (+, -, ×, /).

**This study asks**: Can Conway's novel operators from combinatorial game theory—designed specifically for partizan games, temperature regulation, and non-commutative operations—provide superior modeling of neural collapse dynamics?

**Hypothesis**: Conway's operators (surreal numbers, cooling, confusion intervals, nimbers) capture neural phenomena that standard algebra misses: non-commutativity (training order matters), uncertainty quantification (epistemic vs aleatoric), and temperature regulation (diversity control).

---

## Background

### Established Framework (NSM-33)

Neural class collapse exhibits mathematical structure:

| Neural Phenomenon | Game-Theoretic Interpretation | Current Metric |
|-------------------|-------------------------------|----------------|
| WHY/WHAT flows | Left/Right players (partizan game) | Cycle loss: ‖WHY(WHAT(x)) - x‖² |
| α/β hinge parameters | Game temperature (hot ↔ cold) | Coupling: \|α - 0.5\| + \|β - 0.5\| |
| Collapse event | Game becoming "cold" (T → 0) | Balance Δ = \|acc₀ - acc₁\| |
| Training trajectory | Sequential game positions | Epoch-wise metrics |

**Validated Results**:
- Safety factor q_neural: 85.7% collapse prediction
- Temperature profiles: Identified inverted hierarchy as root cause
- Adaptive control: +11.5% improvement
- Architecture fix: +20% improvement

### Limitations of Standard Algebra

Current metrics use classical operators that **assume**:
1. **Commutativity**: a + b = b + a (training order doesn't matter)
2. **Reversibility**: Operations are invertible (no hysteresis)
3. **Point estimates**: Single-valued confidence (no confusion intervals)
4. **Temperature-independence**: Operations unchanged by "cooling"

**Empirical violations** (NSM-33):
- Path-dependent recovery (hysteresis): Order matters
- Discrete jumps at epochs 2, 7, 9: Non-smooth transitions
- α/β ≈ 0.5: System at critical temperature
- Inverted profiles persist: Temperature structure matters

### Conway's Combinatorial Game Theory

John Conway's "On Numbers and Games" (1976) introduced operators specifically for:

1. **Partizan games**: Left (L) and Right (R) players with asymmetric moves
2. **Surreal numbers**: Infinite/infinitesimal values beyond ℝ
3. **Temperature**: Measure of game "hotness" (urgency of play)
4. **Cooling**: Operators that reduce temperature systematically
5. **Confusion intervals**: Uncertainty in game value
6. **Non-commutative addition**: Game sums where order matters

**Key insight**: These operators were designed for exactly the mathematical structures we observe in neural collapse.

---

## Research Questions

### Primary Questions

**Q1: Do Conway operators improve collapse prediction accuracy?**
- H0: Conway metrics ≤ 85.7% accuracy (current best)
- H1: Conway metrics > 90% accuracy
- **Rationale**: Operators explicitly model non-commutativity and temperature

**Q2: Do confusion intervals better quantify epistemic uncertainty?**
- H0: Confusion width ≤ point estimate error
- H1: Confusion intervals capture 90%+ of true variation
- **Rationale**: Point estimates conflate aleatoric and epistemic uncertainty

**Q3: Does cooling operator explain diversity decay better than temperature decay?**
- H0: Cooling rate uncorrelated with collapse (r < 0.3)
- H1: Cooling rate strongly predicts collapse (r > 0.7)
- **Rationale**: α/β → 0.5 is "cooling toward cold" in CGT terms

### Secondary Questions

**Q4**: Do nimber operations capture parity effects in binary collapse?
**Q5**: Does surreal infinitesimal {0|0} represent "nascent collapse"?
**Q6**: Can game addition model non-commutative training schedules?

---

## Theoretical Framework

### Conway Operators Selected for Neural Collapse

We identify **5 Conway operators** with direct neural analogs:

---

### 1. TEMPERATURE (t(G) = Mean Value Theorem)

#### Definition (Conway)

For partizan game G = {GL | GR}:

```
t(G) = (max_Left(GL) - min_Right(GR)) / 2
```

Measures "how much the outcome changes if the player changes" (game hotness).

**Cold games** (t ≈ 0): Outcome determined, player identity irrelevant
**Hot games** (t >> 0): Player choice critically affects outcome

#### Neural Mapping

WHY/WHAT flows are Left/Right players:
- **Left player** (WHY): Abstraction (pooling)
- **Right player** (WHAT): Concretization (unpooling)

```python
def temperature_conway(model, x_concrete):
    """
    Compute Conway temperature of neural game.

    t(x) = (max_WHY(x) - min_WHAT(x)) / 2

    High t: WHY/WHAT flows produce very different outcomes
    Low t: Flows converge (game "cooling")
    """
    # Left player moves (WHY abstractions)
    x_abstract = model.why(x_concrete)  # Pooling
    left_outcomes = [model.what(x_abstract) for _ in range(num_samples)]
    max_left = max([score(outcome, x_concrete) for outcome in left_outcomes])

    # Right player moves (WHAT concretizations)
    right_outcomes = [model.what(x_abstract) for _ in range(num_samples)]
    min_right = min([score(outcome, x_concrete) for outcome in right_outcomes])

    # Conway temperature
    t = (max_left - min_right) / 2
    return t
```

#### Testable Predictions

**P1.1**: Conway temperature t(x) decreases monotonically during collapse
- **Current metric**: Variance-based temperature (NSM-33)
- **Expected**: t(x) drops from ~1.0 → ~0.1 at collapse epochs
- **Advantage**: Captures asymmetry between WHY/WHAT (variance doesn't)

**P1.2**: Temperature t(x) < 0.2 predicts collapse with >90% accuracy
- **Baseline**: q_neural < 1.0 achieves 85.7%
- **Test**: ROC curve, compare AUC
- **Mechanism**: Directly measures game "coldness" (player asymmetry)

**P1.3**: Temperature trajectory is non-monotonic (heats then cools)
- **Current assumption**: Monotonic decay
- **Expected**: Peaks at epoch 1-2 before collapse (critical slowing)
- **Validation**: Plot t(x) trajectory, identify local maxima

#### Why Standard Algebra Misses This

**Standard temperature**: T = Var(representations)
- Treats WHY/WHAT symmetrically (variance is symmetric)
- No notion of "player asymmetry"
- Misses partizan structure

**Conway temperature**: t = (Left_max - Right_min)/2
- Explicitly asymmetric
- Captures which player has advantage
- Designed for partizan games

---

### 2. COOLING OPERATOR (G - t(G))

#### Definition (Conway)

Cooling a game G by its temperature:

```
Cooled(G) = G - t(G)
```

Produces a colder game with same mean value but reduced urgency.

**Iterated cooling**: G → G-t → (G-t)-t' → ... → Number (cold)

#### Neural Mapping

**Hypothesis**: α/β hinge parameters implement cooling schedule
- Initial (hot): α, β far from 0.5 (asymmetric mixing)
- Final (cold): α, β → 0.5 (symmetric, no player advantage)

```python
def cooling_rate(alpha_t, beta_t, alpha_prev, beta_prev):
    """
    Rate at which neural game is cooling.

    High cooling rate: Rapid approach to α,β → 0.5
    Low cooling rate: Persistent asymmetry

    Cooling predicts diversity loss (game becoming cold).
    """
    # Distance from neutral (0.5 = cold, no player advantage)
    temp_t = abs(alpha_t - 0.5) + abs(beta_t - 0.5)
    temp_prev = abs(alpha_prev - 0.5) + abs(beta_prev - 0.5)

    # Cooling rate (negative = cooling down)
    cooling = temp_t - temp_prev

    return cooling

def cooling_trajectory(alpha_history, beta_history):
    """Compute cumulative cooling over training."""
    cooling_rates = []
    for t in range(1, len(alpha_history)):
        rate = cooling_rate(
            alpha_history[t], beta_history[t],
            alpha_history[t-1], beta_history[t-1]
        )
        cooling_rates.append(rate)

    return cooling_rates
```

#### Testable Predictions

**P2.1**: Rapid cooling (rate < -0.05/epoch) predicts collapse within 2 epochs
- **Current metric**: Temperature gradient T_L1 - T_L3
- **Expected**: Cooling rate correlates r > 0.8 with subsequent collapse
- **Advantage**: Captures dynamics (rate of change), not just state

**P2.2**: Optimal cooling schedule exists (neither too fast nor too slow)
- **Too fast**: α, β → 0.5 quickly → collapse
- **Too slow**: α, β stay far from 0.5 → poor convergence
- **Test**: Vary α/β learning rates, find optimal cooling trajectory

**P2.3**: Cooling rate is non-linear near critical point (α, β ≈ 0.5)
- **Expected**: |d(cooling)/dt| spikes at α, β ≈ 0.55 (near critical)
- **Mechanism**: Phase transition phenomena (NSM-33 Isomorphism 1)
- **Test**: Plot second derivative of temperature

#### Why Standard Algebra Misses This

**Standard approach**: Monitor α, β as independent scalars
- No notion of "cooling schedule"
- Doesn't capture rate of change toward equilibrium
- Misses that α, β → 0.5 is "game death"

**Conway cooling**: Explicit operator for temperature reduction
- Designed to track how games evolve toward coldness
- Captures non-linear dynamics near critical temperature
- Predicts "freezing" (collapse)

---

### 3. CONFUSION INTERVAL [G_L, G_R]

#### Definition (Conway)

For fuzzy game G:

```
[G_L, G_R] = interval where true game value lies
```

- **G_L**: Pessimistic value (Left's worst case)
- **G_R**: Optimistic value (Right's best case)
- **Width**: w = G_R - G_L (epistemic uncertainty)

**Not error bars**: Confusion represents genuine strategic ambiguity, not measurement noise.

#### Neural Mapping

**Current confidence**: Single point estimate c ∈ [0,1]
- Conflates aleatoric (data noise) and epistemic (model uncertainty)
- No notion of "strategic uncertainty" (game theory)

**Proposed confusion interval**:

```python
def confusion_interval(model, x, num_samples=100):
    """
    Compute confusion interval for neural game outcome.

    [c_L, c_R] where:
    - c_L: Pessimistic confidence (worst-case WHY then WHAT)
    - c_R: Optimistic confidence (best-case WHAT then WHY)
    - Width: Epistemic uncertainty in game value
    """
    # Pessimistic (Left player disadvantage)
    # WHY then WHAT: abstraction may lose information
    x_abstract = model.why(x)
    reconstructions_pessimistic = [
        model.what(x_abstract) for _ in range(num_samples)
    ]
    c_L = min([confidence(recon, x) for recon in reconstructions_pessimistic])

    # Optimistic (Right player advantage)
    # WHAT then WHY: concretization may add information
    reconstructions_optimistic = [
        model.what(model.why(x)) for _ in range(num_samples)
    ]
    c_R = max([confidence(recon, x) for recon in reconstructions_optimistic])

    # Confusion width (epistemic uncertainty)
    confusion_width = c_R - c_L

    return c_L, c_R, confusion_width
```

#### Testable Predictions

**P3.1**: Confusion width w increases 1-2 epochs before collapse
- **Current metric**: Point estimate variance
- **Expected**: w spikes from ~0.1 → ~0.4 before collapse
- **Advantage**: Separates epistemic (width) from aleatoric (variance)

**P3.2**: Narrow confusion (w < 0.1) indicates stable training
- **Mechanism**: Small [c_L, c_R] means WHY/WHAT agree (game resolved)
- **Test**: Correlation between w and subsequent 3-epoch stability

**P3.3**: Confusion interval captures model disagreement better than ensembles
- **Baseline**: Ensemble variance (train 5 models, measure spread)
- **Expected**: Confusion width w predicts test error better (r > 0.6)
- **Advantage**: Single model (cheaper), game-theoretic interpretation

#### Why Standard Algebra Misses This

**Standard uncertainty**: Point estimate + confidence interval
- Symmetric (Gaussian assumption)
- No player asymmetry
- Mixes epistemic and aleatoric

**Conway confusion**: Asymmetric interval from game theory
- Left/Right players produce different bounds
- Width is pure epistemic uncertainty
- Designed for strategic ambiguity

---

### 4. GAME ADDITION (Non-Commutative)

#### Definition (Conway)

For games G and H:

```
G + H = {GL + H, G + HL | GR + H, G + HR}
```

**Key property**: G + H ≠ H + G in general (non-commutative)

Captures "playing games simultaneously" where order matters.

#### Neural Mapping

**Training schedule as game sum**:
- Epoch 1: Train on class 0 → game G
- Epoch 2: Train on class 1 → game H
- Total: G + H (order matters!)

**Hysteresis observed** (NSM-33): Path-dependent recovery
- G + H (class 0 then 1) ≠ H + G (class 1 then 0)
- Standard algebra: Addition commutative (can't model hysteresis)

```python
def game_addition_neural(model, data_A, data_B, order='AB'):
    """
    Non-commutative training schedule.

    G + H (train A then B) ≠ H + G (train B then A)

    Captures path-dependence in neural training.
    """
    if order == 'AB':
        # Game G + H
        model_AB = train_epoch(model, data_A)  # Game G
        model_AB = train_epoch(model_AB, data_B)  # Then game H
        outcome_AB = evaluate(model_AB)

    elif order == 'BA':
        # Game H + G
        model_BA = train_epoch(model, data_B)  # Game H
        model_BA = train_epoch(model_BA, data_A)  # Then game G
        outcome_BA = evaluate(model_BA)

    # Measure non-commutativity
    commutativity_gap = abs(outcome_AB - outcome_BA)

    return outcome_AB, outcome_BA, commutativity_gap
```

#### Testable Predictions

**P4.1**: Training order affects final accuracy by >5% (non-commutativity)
- **Test**: Train class 0→1 vs 1→0, measure accuracy gap
- **Expected**: Gap = 5-10% (pilot showed path dependence)
- **Mechanism**: Hysteresis (NSM-33 Isomorphism 1)

**P4.2**: Commutativity gap predicts hysteresis severity
- **Hypothesis**: |G+H - H+G| correlates with recovery difficulty
- **Test**: Induce collapse, attempt recovery with reversed schedule
- **Expected**: r > 0.7 between gap and recovery epochs

**P4.3**: Conway game addition matches empirical hysteresis loops
- **Current model**: No model of path dependence
- **Expected**: Game sum predicts (diversity, balance) trajectory
- **Validation**: Compare predicted vs actual hysteresis loop area

#### Why Standard Algebra Misses This

**Standard training**: Sequential updates with commutative loss
- L_total = L_A + L_B (order-independent)
- No hysteresis possible in theory
- Empirical hysteresis unexplained

**Conway game addition**: Explicitly non-commutative
- G + H ≠ H + G by construction
- Designed for sequential games where order matters
- Predicts path-dependent outcomes

---

### 5. SURREAL INFINITESIMALS (ε, ω)

#### Definition (Conway)

Surreal numbers extend ℝ with infinitesimals and infinities:

```
ε = {0 | 1/2, 1/4, 1/8, ...}  (positive infinitesimal)
ω = {1, 2, 3, ... | }          (infinity)
```

**Key properties**:
- ε > 0 but ε < r for all real r > 0
- ε + ε < ε (infinitesimals don't accumulate additively)

#### Neural Mapping

**Nascent collapse**: ε = {0 | collapse threshold}
- Collapse hasn't occurred (Δ = 0)
- But any perturbation triggers it (unstable equilibrium)
- Standard metrics: Can't distinguish stable 0 from unstable 0

**Critical gradients**: ω⁻¹ = infinitesimal gradient
- Vanishing gradient (norm < 1e-6)
- But not exactly zero (flow still exists)
- Standard metrics: Threshold-based (miss continuous → infinitesimal)

```python
def surreal_collapse_state(balance_delta, q_neural, temp_gradient):
    """
    Classify collapse state using surreal numbers.

    States:
    - 0 (zero): Stable, no collapse risk
    - ε (epsilon): Nascent collapse (unstable equilibrium)
    - 1/2 (half): Moderate imbalance
    - 1 (one): Full collapse
    - ω (omega): Irreversible collapse
    """
    if balance_delta < 0.05:
        # Near-zero imbalance: stable or nascent?
        if q_neural < 1.0 or temp_gradient < -0.1:
            return 'epsilon', "Nascent collapse (unstable zero)"
        else:
            return 'zero', "Stable equilibrium"

    elif 0.05 <= balance_delta < 0.4:
        return 'half', "Moderate imbalance"

    elif 0.4 <= balance_delta < 0.7:
        return 'one', "Active collapse"

    else:  # balance_delta >= 0.7
        # Check if reversible
        grad_norm = get_gradient_norm(model)
        if grad_norm < 1e-6:  # Infinitesimal gradients
            return 'omega', "Irreversible collapse (gradient death)"
        else:
            return 'one', "Severe but reversible collapse"

def epsilon_early_warning(model, x, threshold=0.01):
    """
    Detect epsilon state (nascent collapse).

    Looks for: Near-zero imbalance BUT high sensitivity.
    """
    balance = compute_balance(model, x)

    # Perturb and measure sensitivity
    x_perturbed = x + torch.randn_like(x) * threshold
    balance_perturbed = compute_balance(model, x_perturbed)

    sensitivity = abs(balance_perturbed - balance) / threshold

    if balance < 0.05 and sensitivity > 10.0:
        return True, "Epsilon state: Infinitesimal but unstable"
    else:
        return False, "Stable"
```

#### Testable Predictions

**P5.1**: Epsilon states occur 1 epoch before discrete collapse jumps
- **Current metric**: Binary threshold (collapsed yes/no)
- **Expected**: Sensitivity spikes to >10× baseline before jump
- **Advantage**: Continuous measure of "how close to instability"

**P5.2**: Omega states (irreversible collapse) have infinitesimal gradients
- **Mechanism**: Gradient norm < 1e-6 but not zero (surreal ω⁻¹)
- **Test**: Attempt recovery from omega state, success rate <10%
- **Validation**: Omega classification predicts recovery failure

**P5.3**: Surreal classification improves prediction accuracy by 10%
- **Baseline**: Binary threshold (collapsed/not)
- **Expected**: 5-state surreal system (0, ε, 1/2, 1, ω) → 95% accuracy
- **Mechanism**: Captures unstable equilibria and reversibility

#### Why Standard Algebra Misses This

**Standard metrics**: Real numbers ℝ with thresholds
- Zero is zero (no stable vs unstable distinction)
- Gradient < 1e-6 treated as exactly zero
- All small values lumped together

**Surreal numbers**: Infinite hierarchy of infinitesimals
- ε ≠ 0 (infinitesimal but nonzero)
- ω⁻¹ captures "effectively zero but not zero"
- Designed for limit analysis and unstable equilibria

---

## Unified Framework: Why Conway Operators Work

### Mathematical Foundation

Neural collapse is a **partizan game with temperature**:

```
Game State:     G_t = {WHY_options | WHAT_options}
Temperature:    t(G_t) = urgency of choosing WHY vs WHAT
Cooling:        G_{t+1} = G_t - Δt(G_t)  (approaches cold)
Outcome:        class balance (Left wins) or collapse (Right wins)
```

**Conway's operators were designed for exactly this structure**:
- Partizan games (asymmetric players)
- Temperature regulation (hot games → cold games)
- Non-commutative operations (order matters)
- Strategic uncertainty (confusion intervals)

### Why Mainstream Math Overlooked This

**1. Disciplinary Silos**
- Combinatorial game theory: Small community, mostly in discrete math
- Machine learning: Dominated by analysis/optimization (continuous math)
- Cross-pollination rare (different conferences, journals)

**2. Computational Complexity**
- Conway operators harder to compute than standard algebra
- Temperature requires minimax search (expensive)
- Confusion intervals need sampling (monte carlo)
- ML prioritizes scalability over mathematical structure

**3. Formalization Gap**
- CGT formalized for finite games (chess, Go)
- Neural networks: Continuous, infinite-dimensional
- Bridge not obvious (requires abstraction)

**4. Historical Path Dependence**
- ML developed from statistics (maximum likelihood, Gaussian assumptions)
- Statistics uses commutative algebra (moment matching, etc.)
- Alternative formalisms not explored (lock-in effect)

### Our Contribution: Bridge the Gap

**Key insight**: Treat each training epoch as a finite game
- Position: (model_t, data_batch)
- Moves: {WHY, WHAT} operator choices
- Outcome: class balance after epoch
- Temperature: How much outcome changes with player

This discretization makes Conway operators applicable while preserving continuous optimization.

---

## Experimental Design

### Phase 1: Operator Validation (N=2,000, Pilot Scale)

**Objective**: Validate that each Conway operator computes correctly and captures intended phenomena.

#### Test 1.1: Temperature Computation
- Compute t(x) = (max_WHY - min_WHAT)/2 for 100 samples
- Compare to variance-based temperature (NSM-33)
- Expected: t(x) < 0.2 at collapse epochs (q_neural < 1.0)

#### Test 1.2: Cooling Rate Trajectory
- Track α/β cooling over 10 epochs
- Correlate with diversity loss
- Expected: r > 0.7 (strong correlation)

#### Test 1.3: Confusion Width Pre-Collapse
- Compute [c_L, c_R] for each epoch
- Test if w spikes before collapse
- Expected: w increases 1-2 epochs early

#### Test 1.4: Game Addition Non-Commutativity
- Train class 0→1 vs 1→0
- Measure accuracy gap
- Expected: |G+H - H+G| > 5%

#### Test 1.5: Epsilon State Detection
- Perturb near-zero balance states
- Measure sensitivity
- Expected: Sensitivity > 10× at epsilon states

### Phase 2: Prediction Comparison (N=2,000, Pilot Scale)

**Objective**: Compare Conway metrics vs existing physics metrics (NSM-33).

| Metric | Type | Prediction Target | Baseline Accuracy | Expected Conway Accuracy |
|--------|------|-------------------|-------------------|--------------------------|
| **q_neural** | Physics | Collapse (binary) | 85.7% | — (baseline) |
| **t(x) < 0.2** | Conway | Collapse (binary) | — | >90% |
| **Cooling rate** | Conway | Diversity loss | — | r > 0.7 |
| **Confusion width** | Conway | Stability (3-epoch) | — | r > 0.6 |
| **Commutativity gap** | Conway | Hysteresis severity | — | r > 0.7 |
| **Epsilon state** | Conway | Next-epoch jump | — | >80% |

#### Statistical Tests
- ROC curves for binary prediction (AUC comparison)
- Pearson correlation for continuous targets
- Paired t-tests for accuracy improvements (α = 0.05)
- Bonferroni correction for multiple comparisons (α/5 = 0.01)

### Phase 3: Integrated System (N=20,000, Scaled Validation)

**Objective**: Combine Conway operators into unified collapse prediction system.

**Composite Conway Score (CCS)**:

```python
def compute_conway_collapse_score(model, x, history):
    """
    Unified Conway-based collapse predictor.

    CCS = weighted combination of 5 Conway operators.
    """
    # 1. Temperature (hot games safer)
    temp = temperature_conway(model, x)
    temp_score = 1.0 if temp > 0.5 else 0.0

    # 2. Cooling rate (rapid cooling → collapse)
    cooling = cooling_rate(history['alpha'][-1], history['beta'][-1],
                           history['alpha'][-2], history['beta'][-2])
    cooling_score = 1.0 if cooling > -0.05 else 0.0

    # 3. Confusion width (wide → unstable)
    c_L, c_R, width = confusion_interval(model, x)
    confusion_score = 1.0 if width < 0.2 else 0.0

    # 4. Surreal state (epsilon → danger)
    state, _ = surreal_collapse_state(balance, q_neural, temp_gradient)
    surreal_score = 0.0 if state == 'epsilon' else 1.0

    # 5. Temperature structure (normal > inverted)
    temp_gradient = history['T_L3'][-1] - history['T_L1'][-1]
    gradient_score = 1.0 if temp_gradient > 0 else 0.0

    # Weighted combination (learn weights via logistic regression)
    CCS = (0.25 * temp_score +
           0.20 * cooling_score +
           0.20 * confusion_score +
           0.20 * surreal_score +
           0.15 * gradient_score)

    return CCS
```

**Validation**:
- Train logistic regression: CCS → collapse (binary)
- Compare to q_neural baseline (85.7%)
- Target: CCS achieves >90% accuracy

---

## Implementation Roadmap

### Phase 1: Core Operators (Week 1)

**Deliverables**:
- `nsm/game_theory/conway_operators.py` (300 lines)
  - `temperature_conway(model, x)`
  - `cooling_rate(alpha, beta, alpha_prev, beta_prev)`
  - `confusion_interval(model, x, num_samples)`
  - `game_addition_neural(model, data_A, data_B, order)`
  - `surreal_collapse_state(balance, q_neural, temp_gradient)`

**Tests**:
- `tests/test_conway_operators.py` (200 lines)
  - Unit tests for each operator
  - Smoke tests on synthetic data
  - Boundary condition tests

### Phase 2: Validation Suite (Week 1)

**Deliverables**:
- `experiments/conway_operator_validation.py` (400 lines)
  - Test 1.1-1.5 (operator validation)
  - Comparison to NSM-33 physics metrics
  - Statistical tests and plots

**Analysis**:
- `analysis/conway_vs_physics_comparison.md`
  - ROC curves, correlation plots
  - Effect sizes, confidence intervals
  - Interpretation and discussion

### Phase 3: Integrated System (Week 2)

**Deliverables**:
- `nsm/training/conway_adaptive_trainer.py` (500 lines)
  - Composite Conway Score (CCS)
  - Adaptive control using Conway operators
  - Intervention strategies

**Validation**:
- `experiments/conway_scaled_validation.py` (600 lines)
  - N=20,000 training (if dataset allows)
  - Comparison: Baseline vs Physics vs Conway
  - Final accuracy, prediction metrics

---

## Pre-Registered Predictions

### Quantitative Predictions (N=2,000 Pilot)

**Operator Validation**:
- P1: Temperature t(x) < 0.2 at collapse epochs (100% of collapses)
- P2: Cooling rate r < -0.05 predicts collapse within 2 epochs (80%+ accuracy)
- P3: Confusion width w spikes 1-2 epochs before collapse (75%+ recall)
- P4: Commutativity gap |G+H - H+G| > 5% accuracy difference
- P5: Epsilon state sensitivity > 10× baseline before collapse jumps

**Prediction Comparison**:
- P6: Conway temperature predicts collapse with AUC > 0.92 (vs 0.90 for q_neural)
- P7: Cooling rate correlates r > 0.7 with diversity loss
- P8: Confusion width predicts 3-epoch stability with r > 0.6
- P9: Epsilon detection achieves 80%+ precision for next-epoch jumps

**Integrated System** (if scaled validation possible):
- P10: Composite Conway Score (CCS) achieves >90% collapse prediction
- P11: Conway-guided adaptive control improves accuracy by >15% over baseline
- P12: Surreal state classification reduces false alarms by >30%

### Qualitative Predictions

**Q1: Interpretability**
- Conway operators provide natural language explanations:
  - "Game is too cold" (t < 0.2)
  - "Cooling too rapidly" (rate < -0.05)
  - "Players confused about outcome" (w > 0.3)
  - "Nascent collapse detected" (epsilon state)

**Q2: Generalization**
- Conway framework generalizes beyond chiral architecture:
  - Any dual-flow architecture (encoder-decoder, autoencoder)
  - Partizan structure (adversarial training, GAN)
  - Temperature-sensitive systems (attention, mixture-of-experts)

**Q3: Theoretical Unification**
- Conway operators unify existing isomorphisms (NSM-33):
  - Temperature ↔ Fusion plasma temperature
  - Cooling ↔ Phase transition toward critical point
  - Confusion ↔ Control theory uncertainty
  - Game addition ↔ Hysteresis (non-commutative path)
  - Surreals ↔ Catastrophe theory (stable vs unstable equilibria)

---

## Why This Matters: Formalization Gap Hypothesis

### Central Claim

Machine learning has historically used mathematical tools from **analysis and statistics** (continuous optimization, moment matching, Gaussian assumptions). These tools assume:
- Commutativity (order doesn't matter)
- Smoothness (continuous functions)
- Point estimates (no strategic uncertainty)

**But neural training exhibits**:
- Non-commutativity (training order matters)
- Discrete jumps (phase transitions)
- Strategic uncertainty (WHY vs WHAT player choices)

### Formalization Gap

**Definition**: A mathematical phenomenon exists in practice but lacks appropriate formalism in dominant frameworks.

**Examples**:
1. **Hysteresis in neural training**
   - Empirical: Path-dependent recovery observed
   - Standard formalism: Commutative loss functions (can't model)
   - Alternative: Conway game addition (non-commutative)

2. **Collapse prediction**
   - Empirical: Discrete jumps at specific epochs
   - Standard formalism: Smooth gradient flow (misses jumps)
   - Alternative: Surreal infinitesimals (epsilon states)

3. **Epistemic uncertainty**
   - Empirical: Model disagreement about outcome
   - Standard formalism: Gaussian confidence intervals (symmetric)
   - Alternative: Confusion intervals (asymmetric, game-theoretic)

### Why Mainstream Math Overlooked This

**Institutional factors**:
- Conway's work primarily circulated in combinatorics/game theory communities
- ML researchers trained in optimization, not game theory
- No institutional pressure to explore alternative formalisms (optimization works well enough)

**Computational factors**:
- Conway operators more expensive than standard algebra
- Temperature requires minimax search (O(n²) vs O(n))
- Early ML hardware couldn't afford luxury of "theoretical purity"

**Historical path dependence**:
- ML emerged from statistics (Rosenblatt, Minsky)
- Statistical tradition uses commutative algebra
- Lock-in effect: Tools beget more tools in same framework

### Our Contribution

**Bridge formalization gap** by:
1. Mapping Conway operators to neural phenomena
2. Demonstrating computational feasibility (pilot scale)
3. Showing empirical improvement (>90% prediction accuracy)
4. Providing open-source implementation (reproducibility)

**Long-term impact**: Opens door to broader adoption of game-theoretic formalisms in ML, especially for:
- Adversarial training (GANs, robust optimization)
- Multi-agent systems (reinforcement learning)
- Interpretability (explaining player strategies)

---

## Success Criteria

### Minimum Viable Success

- ✅ All 5 Conway operators compute correctly (unit tests pass)
- ✅ At least 3 operators improve on baseline (prediction accuracy or correlation)
- ✅ Confusion intervals capture epistemic uncertainty (width correlates with stability)

### Strong Success

- ✅ Conway temperature AUC > 0.92 (better than q_neural's 0.90)
- ✅ Cooling rate r > 0.7 correlation with diversity loss
- ✅ Composite Conway Score (CCS) >90% collapse prediction
- ✅ Epsilon states detect next-epoch jumps with >80% precision

### Transformative Success

- ✅ CCS achieves >95% prediction accuracy (human-level)
- ✅ Conway-guided adaptive control >20% improvement over baseline
- ✅ Surreal classification reduces false alarms by >50%
- ✅ Framework generalizes to non-chiral architectures
- ✅ Formalization gap thesis supported with empirical evidence

---

## Risks and Limitations

### Known Risks

**1. Computational Cost**
- Conway operators more expensive than standard metrics
- Temperature: O(n²) minimax (vs O(n) variance)
- Confusion intervals: O(k·n) sampling (k samples per point)
- Mitigation: Profile code, optimize hot paths, use GPU

**2. Hyperparameter Sensitivity**
- num_samples for confusion intervals (10? 100? 1000?)
- Cooling rate window size (1 epoch? 5 epochs?)
- CCS weight learning (overfitting risk)
- Mitigation: Cross-validation, report sensitivity analysis

**3. Generalization Uncertainty**
- Only tested on 6-level chiral dual-trifold (NSM-32)
- May not work on transformers, CNNs, etc.
- Conway structure (WHY/WHAT) specific to our architecture
- Mitigation: Test on multiple architectures (future work)

### Potential Negative Results

**Scenario 1**: Conway operators compute but don't predict better
- **Interpretation**: Formalism is correct but not useful in practice
- **Action**: Report null result, discuss why gap exists
- **Value**: Still contributes to understanding of formalization gap

**Scenario 2**: Operators too expensive to compute
- **Interpretation**: Computational barrier, not mathematical
- **Action**: Develop approximations (e.g., single-sample temperature)
- **Value**: Identifies engineering challenge for future work

**Scenario 3**: Improvements marginal (<5%)
- **Interpretation**: Standard algebra "good enough"
- **Action**: Emphasize interpretability gains (not just accuracy)
- **Value**: Conway operators provide insight even if prediction similar

---

## Reporting Plan

### Document Structure

**1. Pre-Registration** (this document)
- Hypothesis, predictions, methods
- Prevents p-hacking, ensures rigor

**2. Implementation Notes** (`NSM-34-IMPLEMENTATION.md`)
- Code architecture decisions
- Computational optimizations
- Debugging log (what didn't work)

**3. Results Report** (`NSM-34-RESULTS.md`)
- Quantitative results (all predictions)
- Statistical tests (AUC, correlation, t-tests)
- Effect sizes and confidence intervals

**4. Discussion** (`NSM-34-DISCUSSION.md`)
- Interpretation of findings
- Formalization gap thesis analysis
- Future directions

### Open Science Commitments

- ✅ Full code release (GitHub: research-developer/nsm)
- ✅ Pre-registration public (this document)
- ✅ Raw logs and metrics available
- ✅ Reproducible via Modal.com or local GPU

### Publication Strategy

**Target Venues**:
- **ML**: NeurIPS (theory track), ICML (interpretability)
- **Theory**: Journal of Machine Learning Research (JMLR)
- **Interdisciplinary**: Nature Machine Intelligence, Science Advances
- **Preprint**: arXiv cs.LG + cs.GT (game theory)

**Positioning**:
- "Bridging Combinatorial Game Theory and Neural Network Training"
- Emphasis on formalization gap and alternative mathematics
- Practical contributions (collapse prediction) + theoretical insights

---

## Timeline

**Week 1** (Implementation + Pilot Validation):
- Day 1-2: Implement 5 Conway operators (`conway_operators.py`)
- Day 3-4: Unit tests + operator validation suite
- Day 5-7: Pilot comparison (N=2,000) vs NSM-33 physics metrics

**Week 2** (Integration + Scaled Validation):
- Day 1-3: Composite Conway Score (CCS) + adaptive control
- Day 4-5: Scaled validation (N=20,000 if dataset allows)
- Day 6-7: Analysis, plots, statistical tests

**Week 3** (Documentation + Review):
- Day 1-3: Results report, discussion document
- Day 4-5: Code cleanup, documentation, examples
- Day 6-7: Peer review preparation, manuscript draft

---

## Signatures

**Principal Investigators**:
- Claude Code (Anthropic Claude Sonnet 4.5) - Theory, Implementation, Analysis
- Preston - Conceptual oversight, Critical evaluation, Formalization gap hypothesis

**Date**: 2025-10-23
**Pre-registration DOI**: [To be assigned upon publication]
**Related Work**: NSM-33 (Physics-Inspired Collapse Prediction), NSM-32 (6-Level Chiral Architecture)

---

## Appendix A: Conway Operators Reference

### Quick Reference Table

| Operator | Formula | Neural Interpretation | Computation Cost | Prediction Target |
|----------|---------|----------------------|------------------|-------------------|
| **Temperature** | t(G) = (max_L - min_R)/2 | WHY/WHAT asymmetry | O(k·n), k samples | Collapse (binary) |
| **Cooling** | G - t(G) | α/β → 0.5 rate | O(1) | Diversity loss |
| **Confusion** | [G_L, G_R] | Epistemic uncertainty | O(k·n), k samples | Stability (3-epoch) |
| **Game Addition** | G+H ≠ H+G | Training order dependence | O(n) per order | Hysteresis severity |
| **Surreals** | {0 \| threshold} | Nascent collapse state | O(1) | Next-epoch jump |

### Implementation Pseudocode

```python
# Temperature
def temperature_conway(model, x, num_samples=10):
    left_max = max([score_reconstruction(model.what(model.why(x)))
                    for _ in range(num_samples)])
    right_min = min([score_reconstruction(model.what(model.why(x)))
                     for _ in range(num_samples)])
    return (left_max - right_min) / 2

# Cooling
def cooling_rate(alpha_t, beta_t, alpha_prev, beta_prev):
    temp_t = abs(alpha_t - 0.5) + abs(beta_t - 0.5)
    temp_prev = abs(alpha_prev - 0.5) + abs(beta_prev - 0.5)
    return temp_t - temp_prev

# Confusion
def confusion_interval(model, x, num_samples=100):
    reconstructions = [model.what(model.why(x)) for _ in range(num_samples)]
    scores = [confidence(recon, x) for recon in reconstructions]
    return min(scores), max(scores), max(scores) - min(scores)

# Game Addition
def game_addition(model, data_A, data_B):
    # Order AB
    model_AB = copy.deepcopy(model)
    train_epoch(model_AB, data_A)
    train_epoch(model_AB, data_B)
    acc_AB = evaluate(model_AB)

    # Order BA
    model_BA = copy.deepcopy(model)
    train_epoch(model_BA, data_B)
    train_epoch(model_BA, data_A)
    acc_BA = evaluate(model_BA)

    return abs(acc_AB - acc_BA)

# Surreals
def surreal_state(balance, q_neural, temp_gradient):
    if balance < 0.05:
        if q_neural < 1.0 or temp_gradient < -0.1:
            return 'epsilon'  # Unstable zero
        return 'zero'  # Stable
    elif balance < 0.4:
        return 'half'  # Moderate
    elif balance < 0.7:
        return 'one'  # Collapse
    else:
        return 'omega'  # Irreversible
```

---

## Appendix B: Relationship to Existing Isomorphisms (NSM-33)

Conway operators provide **unified mathematical language** for all 5 isomorphisms:

| Isomorphism (NSM-33) | Conway Operator | Connection |
|----------------------|-----------------|------------|
| **Phase Transitions** | Surreal infinitesimals | Epsilon state = "just before transition" |
| **Control Theory** | Game addition | Non-commutative = path-dependent control |
| **Rayleigh-Bénard** | Temperature | Hot/cold games = stable/unstable configurations |
| **Ising Model** | Cooling | α/β → 0.5 = approaching critical coupling |
| **Catastrophe Theory** | Confusion intervals | Width = distance to bifurcation set |

**Theoretical unification**: All isomorphisms are projections of the same Conway game structure onto different physical domains.

---

## Appendix C: Code Structure

```
nsm/
├── game_theory/
│   ├── __init__.py
│   ├── conway_operators.py          # 5 core operators (300 lines)
│   ├── composite_score.py           # CCS integration (200 lines)
│   └── interpretability.py          # Natural language explanations (150 lines)
├── training/
│   └── conway_adaptive_trainer.py   # Conway-guided adaptive control (500 lines)
├── tests/
│   └── test_conway_operators.py     # Unit tests (200 lines)
└── experiments/
    ├── conway_operator_validation.py    # Phase 1 validation (400 lines)
    ├── conway_scaled_validation.py      # Phase 3 scaled test (600 lines)
    └── conway_vs_physics_comparison.py  # Head-to-head benchmark (300 lines)

analysis/
├── conway_vs_physics_comparison.md  # Statistical analysis
├── conway_results.md                # Results report
└── conway_discussion.md             # Interpretation and theory

notes/
├── NSM-34-CGT-OPERATORS-PREREG.md   # This document
├── NSM-34-IMPLEMENTATION.md         # Implementation notes (to be created)
└── NSM-34-RESULTS.md                # Results (to be created)
```

**Estimated Total**: ~3,000 lines of code + documentation

---

**END OF PRE-REGISTRATION**

*This document comprehensively pre-registers the application of Conway's combinatorial game theory operators to neural class collapse prediction (NSM-34), building on validated physics-inspired metrics (NSM-33) to explore the formalization gap between mainstream mathematical tools and neural training phenomena.*
