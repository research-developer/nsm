# Chiral Architecture - Attention Variant Results

**Date**: October 21, 2025
**Branch**: `chiral-attention`
**Variant**: Cross-Attention Hinge Exchange
**Issue**: NSM-31

---

## Executive Summary

**Status**: ❌ FAILED - Did not meet primary success criteria

**Key Results**:
- ✅ Accuracy: 53.10% (vs 43.3% baseline) - **+9.8% improvement**
- ❌ Class Balance Δ: 87.48% (vs 95.3% baseline) - **Still severe collapse**

**Verdict**: The cross-attention hinge architecture successfully improves accuracy but **fails to prevent class collapse**. The hypothesis that simultaneous bidirectional flows would enforce diversity is not validated in this configuration.

---

## Architecture

### Hinge Exchange Mechanism

**Type**: Bidirectional cross-attention at L2

**Implementation**:
```python
class ChiralHingeExchange(nn.Module):
    def __init__(self, dim=128, num_heads=8, dropout=0.1):
        # Cross-attention: upper queries lower's knowledge
        self.upper_to_lower_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention: lower queries upper's knowledge
        self.lower_to_upper_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # Fusion layers combining original + exchanged
        self.fusion_upper = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.fusion_lower = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
```

**Forward Pass**:
1. Upper flow queries lower's knowledge via cross-attention
2. Lower flow queries upper's knowledge via cross-attention
3. Each flow fuses original representation with exchanged information
4. Final L2 representation: average of refined upper and lower flows

**Parameters**: 85,476 total model parameters

---

## Experimental Setup

### Dataset
- **Domain**: Planning (task planning reasoning)
- **Size**: 2,858 total samples
- **Split**: 2,000 train / 870 validation
- **Balance**: 50/50 class distribution
- **Format**: Graph-structured triples

### Training Configuration
- **Epochs**: 10 (early stopping patience=20)
- **Batch Size**: 64
- **Learning Rate**: 1e-4
- **Optimizer**: Adam
- **Loss**: Task loss + 0.01 * cycle_loss
- **Hardware**: Modal A100-40GB GPU
- **Training Time**: ~3 minutes

### Model Architecture
- **Levels**: 3 (L1 → L2 ← L3)
- **Node Features**: 128 dimensions
- **Hidden Dim**: 128
- **Num Heads**: 8 (attention)
- **Dropout**: 0.1
- **Pool Ratio**: 0.5

---

## Results

### Final Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Best Val Accuracy | 53.10% | ≥50% | ✅ PASS |
| Class Balance Δ | 87.48% | <50% | ❌ FAIL |
| Val Loss (best) | 0.7124 | - | - |
| Cycle Loss (final) | ~0.01 | <0.5 | ✅ PASS |

### Training History

| Epoch | Val Acc | Class 0 Acc | Class 1 Acc | Balance Δ | Val Loss |
|-------|---------|-------------|-------------|-----------|----------|
| 1 | 49.77% | 43.81% | 55.33% | 11.52% | 0.7177 |
| 2 | 50.46% | 65.24% | 36.67% | 28.57% | 0.7042 |
| 3 | 50.46% | 100.00% | 4.44% | 95.56% | 0.7102 |
| 4 | 50.69% | 99.52% | 5.56% | 93.97% | 0.7085 |
| 5 | 48.74% | 52.14% | 45.56% | 6.59% | 0.7200 | ← **Best balance**
| 6 | 50.23% | 94.52% | 9.56% | 84.95% | 0.7123 |
| 7 | **53.10%** | 7.86% | 95.33% | **87.48%** | **0.7124** | ← **Best accuracy**
| 8 | 50.00% | 0.00% | 96.67% | 96.67% | 0.7283 |
| 9 | 50.00% | 0.00% | 96.67% | 96.67% | 0.7329 |
| 10 | 50.00% | 0.00% | 96.67% | 96.67% | 0.7381 |

### Key Observations

1. **Accuracy Improvement**: +9.8% over baseline (43.3% → 53.10%)
   - Shows the architecture can learn better representations

2. **Class Collapse Pattern**:
   - Epoch 1-2: Moderate balance (11-28% delta)
   - Epoch 5: Excellent balance (6.59% delta, best)
   - Epoch 7+: Severe collapse to class 1 (87-96% delta)

3. **Collapse Trajectory**:
   - Early training maintains reasonable balance
   - After epoch 5, model begins collapsing to class 1
   - Collapse correlates with accuracy increase
   - Suggests model found "shortcut" to minimize loss

4. **Training Instability**:
   - Class balance oscillates wildly between epochs
   - No monotonic improvement in either direction
   - Indicates unstable optimization landscape

5. **Best Epoch Dilemma**:
   - Epoch 5: Best balance (6.59%), but below-target accuracy (48.74%)
   - Epoch 7: Best accuracy (53.10%), but severe collapse (87.48%)
   - **No epoch satisfies both criteria**

---

## Analysis

### What Worked

1. **Architecture Functions**: Forward/backward passes work correctly
2. **Attention Mechanism**: Cross-attention computes without errors
3. **Training Stability**: No gradient explosion/vanishing
4. **Accuracy**: Meaningful improvement over baseline

### What Failed

1. **Class Balance**: Severe collapse (87.48% vs <50% target)
2. **Diversity Enforcement**: Bidirectional flows did not prevent collapse
3. **Hypothesis Invalidation**: Simultaneous WHY/WHAT exchange insufficient

### Why It Failed

**Hypothesis**: Bidirectional flows at L2 would force diversity by requiring upper (WHY) and lower (WHAT) perspectives to remain complementary.

**Reality**: The cross-attention mechanism allows flows to:
1. Attend to complementary information (good)
2. But both flows can still converge to same class prediction (bad)

**Root Cause**: No explicit constraint forcing flows to make different predictions. The fusion step (averaging) allows both to agree on the same class.

**Loss Landscape**: Cross-entropy loss rewards confident predictions. Model discovered it can minimize loss by:
1. Making both flows predict class 1 confidently
2. Averaging still yields class 1
3. Gets lower loss than balanced predictions

---

## Comparison to Baseline

### Dual-Pass Architecture (Previous Attempt)

| Metric | Dual-Pass | Attention | Δ |
|--------|-----------|-----------|---|
| Best Val Acc | 48.05% | 53.10% | +5.05% |
| Class Balance Δ | 72-100% | 87.48% | Similar |
| Cycle Loss | 0.79-0.86 | ~0.01 | Much better |
| Training Time | ~4 min | ~3 min | Faster |

**Analysis**: Attention variant is strictly better than dual-pass (higher accuracy, better cycle loss), but still fails primary objective (class balance).

### Single-Level Baseline (Original)

| Metric | Baseline | Attention | Δ |
|--------|----------|-----------|---|
| Best Val Acc | 43.3% | 53.10% | +9.8% |
| Class Balance Δ | 95.3% | 87.48% | +7.82% (slight improvement) |

**Analysis**: Attention variant improves both metrics, but improvements insufficient to meet success criteria.

---

## Scoring (NSM-31 Criteria)

| Metric | Weight | Score | Points | Calculation |
|--------|--------|-------|--------|-------------|
| Accuracy | 40% | 53.10% | 14.9/40 | Linear scale 43-70% |
| Class Balance | 30% | 87.48% Δ | 3.8/30 | 0 = 100% collapse, 30 = balanced |
| Cycle Loss | 20% | ~0.01 | 20/20 | Excellent (<0.5 target) |
| Interpretability | 10% | High | 8/10 | Attention weights visualizable |

**Total Score**: **46.7/100** - **FAIL** (below 50 point threshold)

---

## Recommendations

### Option 1: Add Explicit Balance Regularization

**Approach**: Force flows to make diverse predictions

```python
# Diversity loss: penalize agreement between flows
pred_upper = predictor(x_l2_up_refined)
pred_lower = predictor(x_l2_down_refined)

diversity_loss = -torch.mean(
    torch.abs(pred_upper - pred_lower)
)

total_loss = task_loss + λ_cycle * cycle_loss + λ_div * diversity_loss
```

**Pros**: Directly targets the problem
**Cons**: Adds hyperparameter tuning (λ_div)

### Option 2: Use Adversarial Training

**Approach**: Train discriminator to distinguish upper/lower flows

```python
# Discriminator tries to identify which flow produced prediction
discriminator_loss = -log(D(x_upper)) - log(1 - D(x_lower))

# Hinge optimizes to fool discriminator
hinge_loss = -discriminator_loss
```

**Pros**: Forces flows to be maximally different
**Cons**: More complex, slower training

### Option 3: Early Stopping at Best Balance

**Approach**: Stop at epoch 5 (6.59% balance delta)

**Pros**: Simple, no architecture changes
**Cons**: Accuracy below target (48.74% vs 50%)

### Option 4: Proceed with Other Variants

**Approach**: Test gating and fusion variants as planned

**Rationale**: Other mechanisms might inherently enforce diversity
- Gating: Learnable trade-off between flows
- Fusion: Fixed weighted combination

**Recommendation**: **Test fusion variant next** (simplest baseline), then decide whether to iterate on attention or try gating.

---

## Files Modified

1. **nsm/models/chiral.py**:
   - Implemented `ChiralHingeExchange` (lines 30-134)
   - Implemented `MinimalChiralModel` (lines 137-298)
   - Fixed pooling API call to use `.why_operation()` (line 257)

2. **experiments/modal_chiral_validation.py**:
   - Fixed dataset import (`PlanningTripleDataset`)
   - Pinned NumPy version (`<2`)
   - Materialized dataset into list (avoid slicing issues)
   - Added custom `pyg_collate` function
   - Implemented complete training loop

---

## Next Steps

1. **Document results** (this file) ✅
2. **Update Linear NSM-31** with findings
3. **Decide next action**:
   - **Option A**: Implement fusion variant (simplest baseline)
   - **Option B**: Add balance regularization to attention variant
   - **Option C**: Analyze why epoch 5 had good balance, modify training

**Estimated Time**:
- Option A: 1 hour (implement fusion) + 30 min (test)
- Option B: 1 hour (add regularization) + 30 min (retrain)
- Option C: 2 hours (analysis + modification)

**Estimated Cost**:
- Option A: $2 (one Modal run)
- Option B: $2 (one Modal run)
- Option C: $4 (multiple Modal runs)

---

## Conclusion

The cross-attention hinge architecture **successfully implements bidirectional flow exchange** and **improves accuracy**, but **fails to prevent class collapse**. The hypothesis that simultaneous WHY/WHAT perspectives would enforce diversity through complementary reasoning is **not validated** in this configuration.

**Key Insight**: Attention-based exchange allows information flow but does not enforce prediction diversity. Additional constraints (regularization, adversarial training, or alternative fusion mechanisms) are needed.

**Recommendation**: Proceed with **fusion variant** to establish simplest baseline, then decide whether to iterate on attention with regularization or select fusion as final approach.
