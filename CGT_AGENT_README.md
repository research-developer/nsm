# CGT Operators Implementation - Agent Guide

**Worktree Location**: `/Users/preston/Projects/nsm-cgt`
**Branch**: `nsm-34-cgt-operators`
**Main Branch**: `/Users/preston/Projects/NSM` (branch: `main`)

---

## Mission

Implement Conway's Combinatorial Game Theory operators for neural collapse prediction (NSM-34).

**Target**: Composite Conway Score (CCS) achieving **>90% prediction accuracy** (vs 85.7% physics baseline from NSM-33)

---

## Essential Documents (Read These First)

### 1. Pre-Registration (Required Reading)
**Location**: `notes/NSM-34-CGT-OPERATORS-PREREG.md`
- Formal scientific pre-registration with all hypotheses
- 5 Conway operators mapped to neural phenomena
- 12 testable predictions with statistical plans
- Success criteria: Minimum (3/5 operators improve), Strong (>90%), Transformative (>95% + generalizes)

### 2. Implementation Guide (Your Blueprint)
**Location**: `notes/NSM-34-IMPLEMENTATION-GUIDE.md`
- Complete PyTorch code for all 5 operators (copy-paste ready)
- Training loop integration examples
- Unit test templates
- Performance profiling guidelines (target: <15% overhead)

### 3. Quick Reference (Lookup Table)
**Location**: `notes/NSM-34-QUICK-REFERENCE.md`
- One-page cheat sheet
- Decision tree: When to check which operator
- Interpretation guide: What values mean
- Common patterns: "Cold death spiral", "Epsilon precursor", "Confusion explosion"

### 4. Executive Summary (Context)
**Location**: `notes/NSM-34-EXECUTIVE-SUMMARY.md`
- High-level overview for understanding WHY we're doing this
- One-sentence summary: Conway operators capture phenomena standard algebra misses
- 3-tier success criteria

### 5. Formalization Gap Analysis (Theory)
**Location**: `notes/NSM-34-FORMALIZATION-GAP-ANALYSIS.md`
- WHY mainstream ML missed this
- Other potential mathematical gaps
- Theoretical foundation for the work

---

## Baseline Performance (NSM-33)

You're trying to beat these numbers:

| Metric | Baseline | Adaptive | Fixed Arch | Best |
|--------|----------|----------|------------|------|
| **Accuracy** | 48.16% | 53.68% | 57.82% | 57.82% |
| **Prediction Accuracy** | 33.3% (simple) | 85.7% (physics) | — | **85.7%** |
| **Interventions** | 0 | 5 | 0 | 5 |

**Your target**: CCS >90% prediction accuracy (beat 85.7%)

---

## Implementation Roadmap (3-4 weeks)

### Week 1: Core Implementation
**Deliverables**:
1. `nsm/training/cgt_metrics.py` (~500 lines)
   - Temperature t(G)
   - Cooling rate δt/δepoch
   - Confusion intervals [c_L, c_R]
   - Game addition (non-commutative)
   - Surreal number classification

2. `tests/test_cgt_metrics.py` (12+ unit tests)
   - Test each operator independently
   - Test Composite Conway Score (CCS)
   - Test non-commutativity (order matters)

3. `nsm/training/cgt_adaptive_trainer.py` (~300 lines)
   - Infinitesimal perturbation (ε-noise) for hysteresis reduction
   - Thermal annealing based on t(G)
   - Integration with existing AdaptivePhysicsTrainer

### Week 2: Validation Experiments
**Deliverables**:
1. `experiments/modal_cgt_validation.py`
   - Test all 12 predictions from pre-registration
   - Compare CCS vs q_neural vs simple heuristics
   - Track hysteresis reduction with ε-noise

2. Run experiments on Modal.com (N=2,000 pilot, then N=20,000 if successful)

3. `analysis/cgt_validation_results.md`
   - Which predictions validated (✅/❌)
   - Statistical tests (AUC-ROC, precision-recall, correlation)
   - Comparison to NSM-33 physics metrics

### Week 3: Integration & Comparison
**Deliverables**:
1. `nsm/training/unified_predictor.py`
   - Combines physics metrics (NSM-33) + CGT operators (NSM-34)
   - Ensemble predictor: weighted average or meta-learner
   - Test if combination >95% accuracy (transformative success)

2. Ablation studies:
   - Which operators contribute most?
   - Can we remove redundant metrics?
   - What's the minimal set for >90% accuracy?

3. `experiments/comparative_evaluation.py`
   - Physics only vs CGT only vs Combined
   - Statistical significance tests
   - Computational overhead analysis

### Week 4: Documentation & Cleanup
**Deliverables**:
1. Update pre-registration with results
2. Create NSM-34 results summary (like NSM-33-FINAL-SUMMARY.md)
3. Merge nsm-34-cgt-operators → main
4. Prepare publication materials

---

## Key Implementation Details

### The 5 Conway Operators (In Order of Priority)

#### 1. Temperature t(G) - HIGHEST PRIORITY
**Definition**:
```python
def temperature(x_why, x_what):
    """
    Temperature of the game G = (WHY, WHAT).
    Measures asymmetry between flows.
    """
    max_why = global_pool(x_why, 'max')  # Best WHY can do
    min_what = global_pool(x_what, 'min')  # Worst WHAT can do
    t = (max_why - min_what) / 2
    return t
```

**Interpretation**:
- t < 0.2: Cold (collapse imminent)
- t > 0.5: Hot (healthy diversity)
- t ≈ 0.35: Critical zone (monitor closely)

**Prediction**: t < 0.2 predicts collapse with >85% accuracy (beat q_neural)

#### 2. Cooling Rate δt/δepoch - HIGH PRIORITY
**Definition**:
```python
def cooling_rate(temp_history, window=3):
    """
    How fast is the game cooling down?
    """
    recent = temp_history[-window:]
    slope = (recent[-1] - recent[0]) / len(recent)
    return slope
```

**Interpretation**:
- δt/δe < -0.05: Rapid cooling (collapse next epoch)
- δt/δe ≈ 0: Stable
- δt/δe > 0: Heating (recovery)

**Prediction**: Cooling rate correlates with diversity loss (r > 0.7)

#### 3. Confusion Intervals [c_L, c_R] - MEDIUM PRIORITY
**Definition**:
```python
def confusion_interval(logits):
    """
    Uncertainty in prediction = width of confusion interval.
    """
    probs = softmax(logits, dim=-1)
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    c_L = sorted_probs[:, 1]  # Second-best class prob
    c_R = sorted_probs[:, 0]  # Best class prob
    width = c_R - c_L
    return c_L, c_R, width
```

**Interpretation**:
- width < 0.2: Overconfident (potential collapse)
- width > 0.8: Confused (unstable)
- width ≈ 0.5: Healthy uncertainty

**Prediction**: Confusion width spikes before collapse (early warning)

#### 4. Game Addition (Non-Commutative) - MEDIUM PRIORITY
**Definition**:
```python
def game_sum(path_A_to_B, path_B_to_A):
    """
    G + H ≠ H + G (order matters).
    Measures hysteresis via path asymmetry.
    """
    forward_loss = path_A_to_B['final_balance_delta']
    reverse_loss = path_B_to_A['final_balance_delta']
    asymmetry = abs(forward_loss - reverse_loss)
    return asymmetry
```

**Interpretation**:
- asymmetry > 0.1: Significant hysteresis
- asymmetry < 0.05: Reversible (no memory)

**Prediction**: Non-commutativity >5% for collapsed states (already validated in NSM-33)

#### 5. Surreal Numbers {0, ε, ½, 1, ω} - LOW PRIORITY
**Definition**:
```python
def classify_equilibrium(balance_delta, temp):
    """
    Classify system state using surreal numbers.
    """
    if balance_delta < 0.01 and temp > 0.5:
        return '0'  # True equilibrium (rare)
    elif balance_delta < 0.1 and temp > 0.3:
        return 'ε'  # Infinitesimal imbalance (precursor)
    elif 0.1 <= balance_delta < 0.4:
        return '½'  # Half-collapsed (metastable)
    elif balance_delta >= 0.4 and temp < 0.2:
        return '1'  # Full collapse
    else:
        return 'ω'  # Diverging (unstable)
```

**Interpretation**:
- 0: Healthy equilibrium
- ε: Early warning (infinitesimal imbalance)
- ½: Metastable (could go either way)
- 1: Collapsed
- ω: Diverging (emergency)

**Prediction**: Epsilon states predict jumps to 1 with >80% precision

### Composite Conway Score (CCS)
**Definition**:
```python
def composite_conway_score(t, cooling_rate, confusion_width, asymmetry, surreal_state):
    """
    Unified collapse predictor combining all 5 operators.
    """
    # Temperature component (40% weight)
    temp_score = 1.0 if t < 0.2 else (0.5 if t < 0.35 else 0.0)

    # Cooling component (25% weight)
    cooling_score = 1.0 if cooling_rate < -0.05 else 0.0

    # Confusion component (20% weight)
    confusion_score = 1.0 if confusion_width < 0.2 or confusion_width > 0.8 else 0.0

    # Hysteresis component (10% weight)
    hysteresis_score = 1.0 if asymmetry > 0.1 else 0.0

    # Surreal component (5% weight)
    surreal_score = 1.0 if surreal_state in ['1', 'ω'] else (0.5 if surreal_state == 'ε' else 0.0)

    # Weighted sum
    ccs = (0.40 * temp_score +
           0.25 * cooling_score +
           0.20 * confusion_score +
           0.10 * hysteresis_score +
           0.05 * surreal_score)

    return ccs  # Range [0, 1], >0.5 = collapse predicted
```

**Target**: CCS achieves AUC-ROC >0.90 (vs 0.857 for q_neural)

---

## Integration with Existing Code

### Use Physics Metrics as Baseline
```python
from nsm.training.physics_metrics import compute_all_physics_metrics
from nsm.training.cgt_metrics import compute_all_cgt_metrics

# In validation loop:
physics_metrics = compute_all_physics_metrics(model, class_accs, level_reps, epoch)
cgt_metrics = compute_all_cgt_metrics(model_output, targets, epoch)

# Compare
print(f"Physics q_neural: {physics_metrics['q_neural']:.3f}")
print(f"CGT temperature: {cgt_metrics['temperature']:.3f}")
print(f"CGT CCS: {cgt_metrics['ccs']:.3f}")
```

### Adaptive Training with CGT
```python
from nsm.training.cgt_adaptive_trainer import CGTAdaptiveTrainer

trainer = CGTAdaptiveTrainer(
    use_epsilon_noise=True,  # Reduce hysteresis
    thermal_annealing=True,   # Anneal based on t(G)
    monitor_cooling=True      # Alert on rapid cooling
)

# In training loop:
adaptation = trainer.adapt(cgt_metrics, epoch)
if adaptation['interventions']:
    print(f"CGT interventions: {adaptation['interventions']}")
```

---

## Dataset & Experimental Setup

### Use Expanded Dataset (N=24,000)
```python
from nsm.data.planning_dataset import PlanningTripleDataset

dataset = PlanningTripleDataset(
    root="data/planning_24k",
    split="train",
    num_problems=24000,
    problems_per_split=True,
    seed=42
)
```

### Pilot (N=2,000) First
Run small-scale validation before committing to full 24K experiments.

### Use Modal.com for GPU
Copy pattern from `experiments/modal_physics_validation.py`:
- A100 GPU
- 1-hour timeout
- Save results to `/tmp/cgt_results.json`

---

## Success Criteria (From Pre-Registration)

### Minimum Viable Success ✅
- 3/5 Conway operators show improvement over baseline
- CCS >75% prediction accuracy
- At least one operator provides unique signal (not redundant with physics)

### Strong Success ✅✅
- 4/5 Conway operators validated
- CCS >90% prediction accuracy (beat physics 85.7%)
- Hysteresis reduced by >30% with ε-noise
- Computational overhead <15%

### Transformative Success ✅✅✅
- 5/5 Conway operators validated
- CCS >95% prediction accuracy
- Unified predictor (physics + CGT) >98% accuracy
- Generalizes to other datasets/architectures
- Formalization gap thesis validated (other gaps found)

---

## Testing Strategy

### Unit Tests (tests/test_cgt_metrics.py)
```python
def test_temperature_collapse():
    """Temperature should be low (<0.2) during collapse."""
    # Simulate collapsed state
    x_why = torch.ones(100, 64) * 0.1  # Low diversity
    x_what = torch.ones(100, 64) * 0.9  # High uniformity
    t = temperature(x_why, x_what)
    assert t < 0.2, f"Expected cold temperature, got {t}"

def test_non_commutativity():
    """G + H ≠ H + G (order matters)."""
    path_AB = train_sequence(start='A', end='B')
    path_BA = train_sequence(start='B', end='A')
    asymmetry = game_sum(path_AB, path_BA)
    assert asymmetry > 0.05, "Should show hysteresis"
```

### Integration Tests (experiments/modal_cgt_validation.py)
```python
def validate_prediction_1_temperature():
    """P1: Temperature t(G) < 0.2 predicts collapse."""
    # Run training, track t(G) and collapse events
    # Compute AUC-ROC for t(G) as binary predictor
    # Compare to q_neural baseline (0.857)
    assert auc_roc > 0.85, f"Temperature AUC {auc_roc} below target"
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Computational Overhead
**Problem**: CGT metrics add latency
**Solution**:
- Compute every N epochs (not every step)
- Use vectorized operations (no Python loops)
- Target <15% overhead

### Pitfall 2: Redundancy with Physics
**Problem**: CGT just restates q_neural
**Solution**:
- Test orthogonality: correlation between t(G) and q_neural
- If r > 0.9, they're redundant
- Target: unique signal from at least 2/5 operators

### Pitfall 3: Overfitting to Planning Dataset
**Problem**: Works on planning but not KG/Causal
**Solution**:
- Cross-validate on multiple domains (Week 3)
- Test generalization as part of "transformative success"

### Pitfall 4: Poor Calibration
**Problem**: CCS predicts everything as collapse
**Solution**:
- Compute precision-recall curve
- Adjust thresholds in composite score
- Target balanced precision/recall

---

## Deliverables Checklist

### Code (Week 1-2)
- [ ] `nsm/training/cgt_metrics.py` (5 operators + CCS)
- [ ] `tests/test_cgt_metrics.py` (12+ tests, >90% coverage)
- [ ] `nsm/training/cgt_adaptive_trainer.py` (ε-noise + annealing)
- [ ] `experiments/modal_cgt_validation.py` (validation script)

### Results (Week 2-3)
- [ ] `analysis/cgt_validation_results.md` (statistical analysis)
- [ ] Plots: AUC-ROC curves, precision-recall, confusion matrices
- [ ] Comparison table: Physics vs CGT vs Combined

### Documentation (Week 3-4)
- [ ] `notes/NSM-34-RESULTS.md` (final summary)
- [ ] Update pre-registration with actual results
- [ ] Merge nsm-34-cgt-operators → main
- [ ] Create Linear comment with results

---

## Communication

### With Main Branch
- **Fetch updates**: `git fetch origin main`
- **Merge if needed**: `git merge origin/main`
- **Stay in sync**: Physics metrics may update during your work

### With Preston/Claude
- **Status updates**: Share progress at end of each week
- **Blockers**: If stuck, reference specific section of pre-registration
- **Questions**: Check Quick Reference first, then Implementation Guide

---

## Quick Start Command

```bash
cd /Users/preston/Projects/nsm-cgt

# Verify you're on the right branch
git branch  # Should show: * nsm-34-cgt-operators

# Install dependencies (if needed)
pip install torch torch-geometric

# Read the pre-registration
cat notes/NSM-34-CGT-OPERATORS-PREREG.md

# Read the implementation guide
cat notes/NSM-34-IMPLEMENTATION-GUIDE.md

# Start implementing
mkdir -p nsm/training
touch nsm/training/cgt_metrics.py

# Run tests
pytest tests/test_cgt_metrics.py -v
```

---

## Links to Key Documents

**Essential Reading** (in order):
1. `notes/NSM-34-CGT-OPERATORS-PREREG.md` - THE BLUEPRINT
2. `notes/NSM-34-IMPLEMENTATION-GUIDE.md` - CODE TEMPLATES
3. `notes/NSM-34-QUICK-REFERENCE.md` - CHEAT SHEET
4. `notes/NSM-34-EXECUTIVE-SUMMARY.md` - CONTEXT
5. `notes/NSM-34-FORMALIZATION-GAP-ANALYSIS.md` - THEORY

**Reference Code** (for patterns):
- `nsm/training/physics_metrics.py` - NSM-33 implementation
- `nsm/training/adaptive_physics_trainer.py` - Adaptive training pattern
- `experiments/modal_physics_validation.py` - Modal validation pattern

**Baseline Results** (beat these):
- `notes/NSM-33-FINAL-SUMMARY.md` - Full pilot results
- `analysis/phase_transition_results.md` - Phase transition validation

---

**Good luck! You're implementing cutting-edge mathematical framework that mainstream ML has never seen. This could be transformative.** 🚀

---

**Worktree**: `/Users/preston/Projects/nsm-cgt`
**Branch**: `nsm-34-cgt-operators`
**Merge back to**: `main` when complete
