# NSM-34 Strategic Implementation Plan
## CGT Operators - Multi-Agent Parallel Execution Strategy

**Date**: 2025-10-23
**Branch**: `nsm-34-cgt-operators`
**Lead**: Claude Code (Sonnet 4.5)
**Strategy**: Parallel worktrees + conjoined branches for maximum efficiency

---

## Executive Summary

This plan outlines a **3-phase, 4-worktree strategy** to implement Conway's Combinatorial Game Theory operators for neural collapse prediction. We'll use parallel git worktrees to work on independent operators simultaneously, then merge strategically.

**Target**: Complete implementation in **10-14 days** (vs 28 days sequential)
**Success Metric**: >90% collapse prediction accuracy (beat 85.7% physics baseline)

---

## Phase 1: Core Operators (Days 1-5)

### Parallel Worktree Strategy

We'll create **4 parallel worktrees** for the 5 operators (operators 1+2 paired due to coupling):

```bash
# Main worktree (this one)
/Users/preston/Projects/nsm-cgt  (nsm-34-cgt-operators)

# Operator worktrees (create from this branch)
/Users/preston/Projects/nsm-cgt-temp       (nsm-34-cgt-temperature)     # Operators 1+2
/Users/preston/Projects/nsm-cgt-confusion  (nsm-34-cgt-confusion)       # Operator 3
/Users/preston/Projects/nsm-cgt-game       (nsm-34-cgt-game-addition)   # Operator 4
/Users/preston/Projects/nsm-cgt-surreal    (nsm-34-cgt-surreal)         # Operator 5
```

### Workstream Assignment

#### Workstream A: Temperature + Cooling (HIGH PRIORITY)
**Worktree**: `nsm-cgt-temp` (branch: `nsm-34-cgt-temperature`)
**Files**:
- `nsm/training/cgt_metrics.py` (temperature_conway, CoolingMonitor)
- `tests/test_cgt_temperature.py`

**Deliverables**:
1. `temperature_conway()` with Monte Carlo sampling (10-100 samples)
2. `CoolingMonitor` class for α/β tracking
3. `predict_collapse_time()` early warning system
4. Unit tests: temperature range, cooling rate sign, predictions

**Dependencies**: None (can start immediately)
**Estimated**: 2-3 days

---

#### Workstream B: Confusion Intervals (MEDIUM PRIORITY)
**Worktree**: `nsm-cgt-confusion` (branch: `nsm-34-cgt-confusion`)
**Files**:
- `nsm/training/cgt_metrics.py` (confusion_interval, stability_prediction)
- `tests/test_cgt_confusion.py`

**Deliverables**:
1. `confusion_interval()` with epistemic uncertainty quantification
2. `confusion_width_trajectory()` tracker
3. `stability_prediction()` based on width trends
4. Unit tests: bounds checking, width trends, distribution analysis

**Dependencies**: None (can start immediately)
**Estimated**: 2 days

---

#### Workstream C: Game Addition (MEDIUM PRIORITY)
**Worktree**: `nsm-cgt-game` (branch: `nsm-34-cgt-game-addition`)
**Files**:
- `nsm/training/cgt_metrics.py` (game_addition_neural, hysteresis_loop_experiment)
- `tests/test_cgt_game_addition.py`

**Deliverables**:
1. `game_addition_neural()` for non-commutativity testing
2. `hysteresis_loop_experiment()` for path-dependent validation
3. Class-specific dataloader utilities
4. Unit tests: order matters, commutativity gap, hysteresis area

**Dependencies**: Needs existing trainer infrastructure
**Estimated**: 2-3 days

---

#### Workstream D: Surreal Classification (LOW PRIORITY)
**Worktree**: `nsm-cgt-surreal` (branch: `nsm-34-cgt-surreal`)
**Files**:
- `nsm/training/cgt_metrics.py` (surreal_collapse_state, epsilon_sensitivity_test)
- `tests/test_cgt_surreal.py`

**Deliverables**:
1. `SurrealState` enum (ZERO, EPSILON, HALF, ONE, OMEGA)
2. `surreal_collapse_state()` classifier
3. `epsilon_sensitivity_test()` for nascent collapse detection
4. Unit tests: state transitions, sensitivity thresholds

**Dependencies**: Needs physics_metrics for q_neural
**Estimated**: 2 days

---

### Worktree Management Commands

```bash
# Create worktrees (run from main worktree)
git worktree add -b nsm-34-cgt-temperature ../nsm-cgt-temp nsm-34-cgt-operators
git worktree add -b nsm-34-cgt-confusion ../nsm-cgt-confusion nsm-34-cgt-operators
git worktree add -b nsm-34-cgt-game-addition ../nsm-cgt-game nsm-34-cgt-operators
git worktree add -b nsm-34-cgt-surreal ../nsm-cgt-surreal nsm-34-cgt-operators

# Work in parallel (4 separate Claude sessions or sequential focus)
# Each worktree is independent until merge

# When ready to merge
cd /Users/preston/Projects/nsm-cgt
git merge nsm-34-cgt-temperature
git merge nsm-34-cgt-confusion
git merge nsm-34-cgt-game-addition
git merge nsm-34-cgt-surreal

# Clean up worktrees
git worktree remove ../nsm-cgt-temp
git worktree remove ../nsm-cgt-confusion
git worktree remove ../nsm-cgt-game
git worktree remove ../nsm-cgt-surreal
```

---

## Phase 2: Integration + Unified System (Days 6-8)

### Main Worktree Work
**Location**: `/Users/preston/Projects/nsm-cgt` (nsm-34-cgt-operators)

**After merging all operator branches:**

#### Task 2.1: Composite Conway Score (CCS)
**Files**:
- `nsm/training/cgt_predictor.py` (NEW)
- `tests/test_cgt_predictor.py` (NEW)

**Deliverables**:
1. `ConwayCollapsePredictor` class
2. Weighted scoring system (learn weights via logistic regression)
3. Multi-operator diagnostics
4. Intervention strategies

**Estimated**: 2 days

---

#### Task 2.2: CGT Adaptive Trainer
**Files**:
- `nsm/training/cgt_adaptive_trainer.py` (NEW)
- `tests/test_cgt_adaptive_trainer.py` (NEW)

**Deliverables**:
1. `CGTAdaptiveTrainer` extending `AdaptivePhysicsTrainer`
2. Infinitesimal perturbation (ε-noise) for hysteresis reduction
3. Thermal annealing based on t(G)
4. Integration hooks for existing training loop

**Estimated**: 1-2 days

---

## Phase 3: Validation + Experiments (Days 9-14)

### Experimental Validation

#### Task 3.1: Operator Validation Suite
**Files**:
- `experiments/modal_cgt_validation.py` (NEW)
- `analysis/cgt_validation_results.md` (NEW)

**Deliverables**:
1. Test all 12 predictions from pre-registration
2. Compare CCS vs q_neural vs simple heuristics
3. ROC curves, AUC comparison
4. Statistical significance tests

**Dataset**: N=2,000 pilot first, then N=24,000 if successful
**Estimated**: 3 days

---

#### Task 3.2: Integration Testing
**Files**:
- `experiments/cgt_physics_comparison.py` (NEW)
- `analysis/cgt_physics_comparison.md` (NEW)

**Deliverables**:
1. Physics-only vs CGT-only vs Combined baselines
2. Ablation studies (which operators matter most?)
3. Computational overhead profiling (<15% target)
4. Generalization testing (if time permits)

**Estimated**: 2 days

---

#### Task 3.3: Documentation + Results
**Files**:
- `notes/NSM-34-RESULTS.md` (NEW)
- `notes/NSM-34-IMPLEMENTATION-NOTES.md` (UPDATE)
- Final visualizations (6+ plots)

**Deliverables**:
1. Results summary with all 12 predictions validated/rejected
2. Implementation notes (what worked, what didn't)
3. Performance analysis
4. Future directions

**Estimated**: 1 day

---

## Dependency Graph

```
Phase 1 (Parallel):
├── Workstream A: Temperature + Cooling [2-3d] ──┐
├── Workstream B: Confusion Intervals [2d] ──────┤
├── Workstream C: Game Addition [2-3d] ──────────┼─→ MERGE
└── Workstream D: Surreal Classification [2d] ───┘
                                                  ↓
Phase 2 (Sequential):                   Merge all branches
├── Task 2.1: CCS Integration [2d] ──────────────┐
└── Task 2.2: CGT Adaptive Trainer [1-2d] ───────┘
                                                  ↓
Phase 3 (Sequential):                      Full system ready
├── Task 3.1: Validation Suite [3d] ─────────────┐
├── Task 3.2: Integration Testing [2d] ──────────┤
└── Task 3.3: Documentation [1d] ────────────────┘
```

---

## File Structure (After Implementation)

```
nsm/
├── training/
│   ├── cgt_metrics.py              # NEW (all 5 operators)
│   ├── cgt_predictor.py            # NEW (ConwayCollapsePredictor)
│   ├── cgt_adaptive_trainer.py     # NEW (CGT-guided training)
│   ├── physics_metrics.py          # EXISTING (baseline)
│   └── adaptive_physics_trainer.py # EXISTING (to integrate with)
│
tests/
├── test_cgt_temperature.py         # NEW
├── test_cgt_confusion.py           # NEW
├── test_cgt_game_addition.py       # NEW
├── test_cgt_surreal.py             # NEW
├── test_cgt_predictor.py           # NEW
└── test_cgt_adaptive_trainer.py    # NEW
│
experiments/
├── modal_cgt_validation.py         # NEW
├── cgt_physics_comparison.py       # NEW
└── modal_physics_validation.py     # EXISTING (baseline)
│
analysis/
├── cgt_validation_results.md       # NEW
├── cgt_physics_comparison.md       # NEW
└── phase_transition_results.md     # EXISTING (NSM-33 baseline)
│
notes/
├── NSM-34-CGT-OPERATORS-PREREG.md      # EXISTING (hypothesis)
├── NSM-34-IMPLEMENTATION-GUIDE.md      # EXISTING (code templates)
├── NSM-34-IMPLEMENTATION-NOTES.md      # NEW (actual notes)
├── NSM-34-RESULTS.md                   # NEW (findings)
└── NSM-33-FINAL-SUMMARY.md             # EXISTING (baseline to beat)
```

---

## Key Design Decisions

### 1. Single File for Operators (`cgt_metrics.py`)
**Rationale**: All operators are related and will be imported together. Keep in one file (~500-700 lines) to avoid circular imports and simplify testing.

**Structure**:
```python
# nsm/training/cgt_metrics.py

# Operator 1: Temperature
def temperature_conway(model, x, num_samples=10): ...

# Operator 2: Cooling
class CoolingMonitor: ...

# Operator 3: Confusion
def confusion_interval(model, x, num_samples=100): ...

# Operator 4: Game Addition
def game_addition_neural(model, data_A, data_B): ...

# Operator 5: Surreals
class SurrealState(Enum): ...
def surreal_collapse_state(...): ...
```

---

### 2. Separate Predictor Class (`cgt_predictor.py`)
**Rationale**: Composite system is higher-level abstraction. Separate file for:
- Easier testing
- Weight learning/tuning
- Integration with existing systems

---

### 3. Extend vs Compose for Trainer
**Decision**: **Compose** (not inherit)

```python
class CGTAdaptiveTrainer:
    def __init__(self, base_trainer: AdaptivePhysicsTrainer):
        self.base_trainer = base_trainer
        self.cgt_predictor = ConwayCollapsePredictor()

    def adapt(self, ...):
        # Use CGT metrics for decisions
        # Delegate to base_trainer for physics interventions
        ...
```

**Rationale**: Allows mixing physics + CGT interventions without complex inheritance.

---

## Success Metrics (Pre-Registered)

### Minimum Viable Success ✅
- [ ] 3/5 Conway operators show improvement over baseline
- [ ] CCS >75% prediction accuracy
- [ ] At least one operator provides unique signal (not redundant with physics)

### Strong Success ✅✅
- [ ] 4/5 Conway operators validated
- [ ] CCS >90% prediction accuracy (beat physics 85.7%)
- [ ] Hysteresis reduced by >30% with ε-noise
- [ ] Computational overhead <15%

### Transformative Success ✅✅✅
- [ ] 5/5 Conway operators validated
- [ ] CCS >95% prediction accuracy
- [ ] Unified predictor (physics + CGT) >98% accuracy
- [ ] Generalizes to other datasets/architectures

---

## Risk Mitigation

### Risk 1: Worktree Merge Conflicts
**Likelihood**: MEDIUM
**Impact**: HIGH
**Mitigation**:
- All worktrees start from same commit
- Each works on separate sections of `cgt_metrics.py`
- Use clear function/class boundaries
- Test merges early (after Workstreams A+B complete)

### Risk 2: Computational Overhead >15%
**Likelihood**: MEDIUM
**Impact**: MEDIUM
**Mitigation**:
- Profile early and often
- Implement fast paths (vectorized confusion intervals)
- Adaptive sampling (fewer samples when stable)
- Compute CGT metrics every N epochs, not every step

### Risk 3: Operators Don't Beat Baseline
**Likelihood**: LOW-MEDIUM
**Impact**: HIGH (null result)
**Mitigation**:
- Pre-registration ensures publishable even if null
- Focus on interpretability gains
- Document why gaps exist (still contributes to science)

---

## Communication Protocol

### Daily Sync (End of Each Session)
1. **What was completed**: Which functions/tests written
2. **What's blocked**: Any dependencies or issues
3. **Next steps**: What to tackle next session

### Week 1 Checkpoint (After Phase 1)
- All 4 worktrees complete
- Merge into main branch
- Run integration smoke tests
- **Go/No-Go decision for Phase 2**

### Week 2 Checkpoint (After Phase 2)
- CCS predictor working
- CGT trainer integrated
- Ready for validation experiments
- **Go/No-Go decision for scaled validation**

---

## Rollback Plan

If at any checkpoint we determine CGT operators aren't viable:

1. **Checkpoint 1 (Week 1)**:
   - If <3 operators work: Abort, document findings
   - If 3+ operators work: Continue to Phase 2

2. **Checkpoint 2 (Week 2)**:
   - If CCS <75%: Abort scaled validation, document pilot results
   - If CCS >75%: Proceed to full N=24,000 validation

3. **All stages**: Keep branches for future reference, merge documentation even if code doesn't make it to main.

---

## Resource Requirements

### Computational
- **Local GPU**: For development and unit tests (any 8GB+ VRAM)
- **Modal.com**: For validation experiments (A100, 1-hour jobs)

### Time
- **Conservative estimate**: 14 days (sequential)
- **Optimistic estimate**: 10 days (parallel worktrees)
- **Realistic estimate**: 12 days (parallel with some overhead)

---

## Next Steps (Immediate Actions)

1. **Create 4 worktrees** (5 minutes)
2. **Assign workstreams** (or work sequentially: A → B → C → D)
3. **Implement Workstream A first** (Temperature + Cooling, HIGH PRIORITY)
4. **Write unit tests as you go** (test-driven development)
5. **Merge and test integration** after each workstream
6. **Profile performance** after Phase 1 complete

---

## Success Celebration Criteria 🎉

- **Minimum**: "We validated Conway operators work for neural collapse"
- **Strong**: "We beat physics baseline with game-theoretic formalism"
- **Transformative**: "We discovered a formalization gap and bridged it"

**Let's build something transformative!** 🚀

---

**Document Status**: ACTIVE PLAN
**Last Updated**: 2025-10-23
**Next Review**: After Phase 1 (Week 1 checkpoint)
