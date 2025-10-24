# NSM-31: Chiral Architecture - Parallel Exploration Strategy

**Created**: October 21, 2025
**Issue**: NSM-31 - Chiral Dual-Trifold Architecture
**Status**: Active - 3 Parallel Branches

---

## Overview

Testing 3 different approaches to hinge exchange mechanism in parallel to identify the most effective method for preventing class collapse via simultaneous bidirectional flows.

**Core Hypothesis**: Simultaneous bidirectional flows with L2 exchange can prevent class collapse by forcing diversity **during the forward pass**, not after predictions are made.

---

## Parallel Branches

### Branch 1: `chiral-attention` (Cross-Attention Exchange)
**Location**: `/Users/preston/Projects/nsm-chiral-attention`
**Approach**: Standard cross-attention mechanism

**Mechanism**:
```python
# Upper queries lower's knowledge
upper_from_lower = MultiheadAttention(
    query=x_upper,
    key=x_lower,
    value=x_lower
)

# Lower queries upper's knowledge
lower_from_upper = MultiheadAttention(
    query=x_lower,
    key=x_upper,
    value=x_upper
)

# Fuse with residuals
x_upper_refined = fusion([x_upper, upper_from_lower])
x_lower_refined = fusion([x_lower, lower_from_upper])
```

**Pros**:
- ✅ Standard mechanism, well-understood
- ✅ Learnable interaction patterns
- ✅ Attention weights interpretable
- ✅ PyTorch implementation available (nn.MultiheadAttention)

**Cons**:
- ❌ Higher computational cost (O(n²) for self-attention)
- ❌ More parameters to tune (num_heads, dropout)
- ❌ May be overkill for simple exchange

**Expected Outcome**: Best interpretability and flexibility, moderate complexity

---

### Branch 2: `chiral-gating` (Learnable Gating Mechanism)
**Location**: `/Users/preston/Projects/nsm-chiral-gating`
**Approach**: Learnable gates control information flow

**Mechanism**:
```python
# Compute gates
gate_upper = sigmoid(W_gate_upper @ concat([x_upper, x_lower]))
gate_lower = sigmoid(W_gate_lower @ concat([x_lower, x_upper]))

# Gated exchange
x_upper_refined = (1 - gate_upper) * x_upper + gate_upper * transform(x_lower)
x_lower_refined = (1 - gate_lower) * x_lower + gate_lower * transform(x_upper)
```

**Pros**:
- ✅ Simpler than attention (O(n) complexity)
- ✅ Fewer parameters
- ✅ Similar to GRU/LSTM gating (proven mechanism)
- ✅ Fast training

**Cons**:
- ❌ Less expressive than attention
- ❌ Gates may collapse to extremes (0 or 1)
- ❌ Less interpretable interaction patterns

**Expected Outcome**: Best efficiency, moderate expressiveness

---

### Branch 3: `chiral-fusion` (Direct Weighted Fusion)
**Location**: `/Users/preston/Projects/nsm-chiral-fusion`
**Approach**: Simple learnable weighted sum (baseline)

**Mechanism**:
```python
# Learnable weights
alpha = learnable_param([1, dim])
beta = learnable_param([1, dim])

# Direct fusion
x_upper_refined = alpha * x_upper + (1 - alpha) * transform(x_lower)
x_lower_refined = beta * x_lower + (1 - beta) * transform(x_upper)
```

**Pros**:
- ✅ Simplest approach
- ✅ Minimal parameters
- ✅ Fast training and inference
- ✅ Easy to debug

**Cons**:
- ❌ Least expressive
- ❌ Fixed mixing ratio (no position-dependent interaction)
- ❌ May not provide enough diversity enforcement

**Expected Outcome**: Baseline for comparison, may be sufficient if problem is simple

---

## Testing Protocol

### Identical Configuration (Fair Comparison)

All 3 branches use **identical settings** except for hinge exchange mechanism:

**Dataset**: Planning (2,858 samples, 50/50 class balance)

**Architecture**:
- Minimal 3-level chiral (L1 ↔ L2 ↔ L3)
- Single hinge at L2
- Node features: 64
- Hidden dim: 128

**Training**:
- Epochs: 10 (early stopping patience=20)
- Batch size: 64
- Learning rate: 1e-4
- Optimizer: Adam
- Loss: Task loss + 0.01 * cycle_loss

**Hardware**:
- Modal GPU (A100-40GB)
- ~30 minutes per variant
- ~$2 per variant

**Total Cost**: ~$6 for all 3 variants

---

## Success Criteria

### Primary Metrics (Must Pass)

1. **Accuracy** ≥ 50% (random baseline)
   - Current baseline: 43.3% (FAILED)
   - Target: Beat random guessing

2. **Class Balance Delta** < 50%
   - Current baseline: 95.3% (SEVERE collapse)
   - Dual-pass: 72-100% (WORSE)
   - Target: Significant improvement

### Secondary Metrics (Nice to Have)

3. **Reconstruction Error** < 20%
   - Cycle consistency: ||WHY(WHAT(x)) - x||² / ||x||²
   - Current: ~0.79-0.86 (poor)

4. **Training Stability**
   - Monotonic loss decrease (smoothed)
   - No gradient explosion/vanishing

5. **Interpretability**
   - Can visualize exchange patterns
   - Attention/gate weights make sense

---

## Evaluation Timeline

### Phase 1: Individual Testing (October 22, 2025)
**Duration**: 1.5 hours per variant (implementation + testing)
**Total**: 4.5 hours

**Tasks per branch**:
1. Implement hinge exchange variant (1 hour)
2. Run validation script (30 min)
3. Analyze results (save to `/tmp/{variant}_results.json`)

**Parallel execution**:
- All 3 developers can work simultaneously on separate branches
- Or single developer implements sequentially

### Phase 2: Comparison (October 22, 2025)
**Duration**: 1 hour
**Owner**: Lead developer

**Tasks**:
1. Collect results from all 3 branches
2. Compare metrics (accuracy, class balance, cycle loss)
3. Statistical significance testing (95% CI)
4. Visualize attention/gate patterns (if interpretable)
5. Select winner

### Phase 3: Winner Integration (October 22-23, 2025)
**Duration**: 2 hours
**Owner**: Lead developer

**Tasks**:
1. Merge winning branch to `phase1.5-3level`
2. Clean up code (remove TODOs, add documentation)
3. Update NSM-31 Linear issue
4. Prepare for Stage 2 (full 6-level implementation)

---

## Decision Criteria

### Quantitative Scoring

| Metric | Weight | Threshold | Points |
|--------|--------|-----------|--------|
| Accuracy | 40% | ≥50% | 0-40 pts (linear scale 43-70%) |
| Class Balance | 30% | Δ<50% | 0-30 pts (0 = 100% collapse, 30 = balanced) |
| Cycle Loss | 20% | <0.5 | 0-20 pts (linear scale 0.2-0.9) |
| Interpretability | 10% | Qualitative | 0-10 pts (subjective) |

**Total**: 100 points possible

**Selection Rule**:
- Variant with highest score wins
- If tie (within 5 points), choose simpler implementation
- If all fail (<50 points), re-examine hypothesis

### Qualitative Factors

- **Simplicity**: Fewer parameters, easier to debug
- **Extensibility**: Can scale to 6-level architecture
- **Robustness**: Stable training, no hyperparameter sensitivity
- **Novelty**: Publishable if successful

---

## Risk Mitigation

### Risk 1: All 3 Variants Fail
**Probability**: Medium (30%)
**Impact**: High (invalidates chiral hypothesis)

**Mitigation**:
1. Quick abort ($6 total cost, 4.5 hours)
2. Fallback: Re-examine dataset quality
3. Fallback: Test standard GCN baseline
4. Fallback: Add explicit class balancing loss

**Decision Point**: If all score <50 points, pivot to alternative approach

### Risk 2: Multiple Variants Succeed
**Probability**: Low (20%)
**Impact**: Low (good problem to have)

**Mitigation**:
1. Select simplest (fusion > gating > attention)
2. Run ablation study (remove hinge, compare)
3. Test on other domains (Causal, KG)

### Risk 3: Winner Doesn't Scale to 6-Level
**Probability**: Low (10%)
**Impact**: Medium (need to re-implement)

**Mitigation**:
1. Design with extensibility in mind
2. Test with 2 hinges before full 3-hinge implementation
3. Staged rollout (3-level → 4-level → 6-level)

---

## Branch Management

### Workflow

**Do NOT push branches to remote** - keep local only for exploration

**After winner selected**:
1. Merge winning branch to `phase1.5-3level`:
   ```bash
   git checkout phase1.5-3level
   git merge chiral-{winner} --no-ff
   ```

2. Delete losing branches and worktrees:
   ```bash
   git worktree remove /Users/preston/Projects/nsm-chiral-{loser}
   git branch -D chiral-{loser}
   ```

3. Archive results:
   ```bash
   mv /tmp/*_results.json experiments/chiral_exploration/
   ```

### Commit Messages (Per Branch)

**Format**: `Implement {variant} hinge exchange for minimal chiral`

**Example**:
```
Implement attention-based hinge exchange for minimal chiral

Use bidirectional cross-attention at L2 hinge:
- 8 attention heads
- 0.1 dropout
- Residual fusion

Results: 52% accuracy, 38% class balance delta (IMPROVEMENT)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Expected Outcomes

### Pessimistic (30% probability)
- All variants fail (accuracy <50%, collapse >80%)
- Cost: $6, 4.5 hours
- Outcome: Chiral hypothesis invalidated, pivot to alternative
- Value: Learned what doesn't work

### Realistic (50% probability)
- One variant succeeds (accuracy 50-60%, collapse 30-50%)
- Cost: $6, 4.5 hours + 2 hours integration
- Outcome: Select winner, proceed to 6-level
- Value: Validated approach, clear path forward

### Optimistic (20% probability)
- Multiple variants succeed (accuracy >60%, collapse <30%)
- Cost: $6, 4.5 hours + 2 hours integration + 2 hours ablation
- Outcome: Select simplest winner, publish comparison
- Value: Strong validation, publishable results

---

## Integration Checklist

After winner selected:

### Code
- [ ] Merge winning branch to `phase1.5-3level`
- [ ] Remove TODOs from `nsm/models/chiral.py`
- [ ] Add comprehensive docstrings
- [ ] Write unit tests for hinge exchange
- [ ] Update `nsm/models/__init__.py` to export `MinimalChiralModel`

### Documentation
- [ ] Update NSM-31 Linear issue with results
- [ ] Create `notes/CHIRAL_EXPLORATION_RESULTS.md`
- [ ] Update `experiments/training_log.jsonl`
- [ ] Document decision in `notes/NSM_PHASE1.5_DECISION_LOG.md`

### Validation
- [ ] Test winner on Causal domain
- [ ] Test winner on KG domain
- [ ] Compare to baseline (dual-pass, single-pass)
- [ ] Run ablation (remove hinge, test impact)

### Preparation for Stage 2
- [ ] Design 6-level architecture with 3 hinges
- [ ] Implement normalization inversion
- [ ] Create full validation script
- [ ] Estimate GPU cost for full training

---

## Resources

### Implementation References
- `nsm/models/chiral.py`: Base classes (ChiralHingeExchange, MinimalChiralModel)
- `experiments/modal_chiral_validation.py`: Validation script template
- `notes/CHIRAL_ARCHITECTURE.md`: 3-level design specification
- `notes/FULL_CHIRAL_6LEVEL.md`: 6-level architecture (future)

### Background
- `notes/DUAL_PASS_VALIDATION_RESULTS.md`: Why sequential doesn't work
- `notes/NSM_PHASE1.5_DECISION_LOG.md`: Decision history
- NSM-31 Linear issue: Project tracking

---

**Status**: Ready for parallel implementation
**Next Step**: Implement hinge exchange in all 3 branches
**Deadline**: October 22, 2025 (end of day)
**Budget**: $6 GPU, 6.5 hours dev time
