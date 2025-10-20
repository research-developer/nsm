# NSM-31: Training Stability Fixes

## Problem Analysis

All three domains showing poor performance across the board:
- **Planning**: 43.5% accuracy (class collapse - always predicts class 1)
- **Causal**: 52.9% accuracy (barely above random)
- **KG**: 46.0% accuracy (below random)
- **Cycle Loss**: 0.78-0.98 (target <0.2) - reconstruction failing

### Root Causes Identified

#### 1. ✅ Dataset Balance (NOT the issue)
- Planning: 50.0% / 50.0% (perfect)
- KG: 50.0% / 50.0% (perfect)
- Causal: 57.9% / 42.1% (acceptable)

#### 2. ✅ PyG Extensions (NOT the issue)
- torch-scatter/torch-sparse show warnings but SAGPooling works
- Fallback implementations in pure PyTorch are functional
- Pooling operations verified working

#### 3. ❌ Cycle Loss Dominance (MAIN ISSUE)
- Cycle loss weight: 0.1
- Cycle loss magnitude: ~0.98
- Contribution to gradient: 0.1 × 0.98 = 0.098
- Task loss (cross-entropy): ~0.7
- **Problem**: Cycle loss gradient is competing with task gradient!

#### 4. ❌ No Class Weights
- Binary classification without class weighting
- Model can minimize loss by always predicting majority class
- No mechanism to prevent collapse

#### 5. ❌ Learning Rate Too High
- Current: 1e-3
- High LR + complex model → unstable training
- Cycle loss not converging

## Proposed Fixes

### Fix 1: Progressive Cycle Loss Warmup

Instead of fixed weight, use warmup schedule:

```python
def get_cycle_loss_weight(epoch, max_epochs=100, initial=0.0, final=0.05):
    """
    Progressive warmup for cycle loss weight.

    Epochs 0-20: Linear ramp 0.0 → 0.05
    Epochs 20+: Fixed at 0.05
    """
    if epoch < 20:
        return initial + (final - initial) * (epoch / 20)
    return final
```

**Rationale**: Let model learn task first, then enforce cycle consistency

### Fix 2: Class-Weighted Loss

Add class weights to combat collapse:

```python
# In training script
from torch.nn import CrossEntropyLoss

# Count class distribution
class_counts = torch.bincount(all_labels)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()  # Normalize

criterion = CrossEntropyLoss(weight=class_weights)
```

**Rationale**: Forces model to learn both classes equally

### Fix 3: Reduce Learning Rate

Change from 1e-3 → 5e-4 with cosine annealing:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
```

**Rationale**: More stable training, better convergence

### Fix 4: Increase Gradient Monitoring

Add gradient norm logging to detect vanishing/exploding:

```python
def log_gradient_norms(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
```

**Rationale**: Identify if gradient clipping is helping or hurting

### Fix 5: Adaptive Cycle Loss Weight

Use validation reconstruction error to adjust weight:

```python
def adjust_cycle_weight(val_cycle_loss, current_weight, target=0.2):
    """
    Increase weight if reconstruction is good (< target).
    Decrease weight if reconstruction is poor (> 2*target).
    """
    if val_cycle_loss < target:
        return min(current_weight * 1.1, 0.2)  # Increase (cap at 0.2)
    elif val_cycle_loss > 2 * target:
        return max(current_weight * 0.9, 0.01)  # Decrease (floor at 0.01)
    return current_weight
```

**Rationale**: Self-tuning based on reconstruction quality

## Implementation Plan

### Phase 1: Quick Fixes (Immediate)

1. **Reduce cycle loss weight**: 0.1 → 0.01
2. **Reduce learning rate**: 1e-3 → 5e-4
3. **Add class weights** to CrossEntropyLoss
4. **Run 20-epoch validation** on all three domains

**Expected Results**:
- Accuracy > 60% (better than random)
- Cycle loss < 0.5 (improving trend)
- No class collapse (both classes predicted)

### Phase 2: Progressive Improvements (After Phase 1)

5. **Implement cycle loss warmup** (0.0 → 0.05 over 20 epochs)
6. **Add cosine LR scheduler**
7. **Implement gradient logging**
8. **Run 100-epoch training**

**Expected Results**:
- Accuracy > 75%
- Cycle loss < 0.3
- Stable training (no plateaus)

### Phase 3: Adaptive Tuning (After Phase 2)

9. **Implement adaptive cycle weight**
10. **Tune pool ratio** per domain
11. **Hyperparameter search** (learning rate, weight decay)

**Expected Results**:
- Accuracy > 85%
- Cycle loss < 0.2 (target achieved)
- Transferable to 3-level architecture (NSM-30)

## Validation Metrics

Track these metrics to validate fixes:

### Classification Metrics
- **Overall Accuracy**: > 60% (Phase 1), > 75% (Phase 2), > 85% (Phase 3)
- **Per-Class Accuracy**: Both > 50% (no collapse)
- **F1 Score**: > 0.7 (Phase 2+)

### Reconstruction Metrics
- **Cycle Loss**: < 0.5 (Phase 1), < 0.3 (Phase 2), < 0.2 (Phase 3)
- **Gradient Norm**: 0.1 - 10.0 (stable range)

### Training Stability
- **Loss Curve**: Monotonic decrease (smoothed over 10 epochs)
- **No Early Stopping**: Reaches at least 50 epochs before patience trigger
- **Learning Rate**: Scheduler reduces smoothly, not prematurely

## Command to Run Phase 1 Fixes

```bash
# Planning
cd /Users/preston/Projects/nsm-planning
python experiments/train_planning.py \
  --epochs 20 \
  --batch-size 32 \
  --num-plans 2858 \
  --cycle-loss-weight 0.01 \
  --lr 5e-4 \
  --seed 42

# Causal
cd /Users/preston/Projects/nsm-causal
python experiments/train_causal.py \
  --epochs 20 \
  --batch-size 32 \
  --num-scenarios 1000 \
  --cycle-loss-weight 0.01 \
  --lr 5e-4 \
  --seed 42

# KG
cd /Users/preston/Projects/nsm-kg
python experiments/train_kg.py \
  --epochs 20 \
  --batch-size 32 \
  --num-entities 100 \
  --num-triples 500 \
  --cycle-loss-weight 0.01 \
  --lr 5e-4 \
  --seed 42
```

## Next Steps

1. **Implement Phase 1 fixes** in trainer.py (class weights)
2. **Run 20-epoch validation** with new hyperparameters
3. **Analyze results** and proceed to Phase 2 if successful
4. **Document findings** in NSM-10-CROSS-DOMAIN-COMPARISON.md
5. **Only proceed to NSM-30** (3-level) after 2-level is solid

## References

- NSM-20: Phase 1 Foundation Implementation
- NSM-10: Dataset Exploration (Causal, KG, Planning)
- NSM-30: 3-Level Architecture (blocked until this is resolved)
- CLAUDE.md: Architecture principles and constraints
