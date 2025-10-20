# NSM-10: CRITICAL FINDINGS - Dataset Exploration Red Flags

**Status**: 🚨 **TRAINING INVALIDATED** - All three datasets have degenerate label distributions
**Date**: 2025-10-20
**Issue**: All synthetic datasets generate 100% positive examples (class 1 only)

---

## Executive Summary

Investigation of the KG domain's suspicious 100% validation accuracy revealed a **catastrophic bug affecting all three exploration branches**: All datasets only generate positive examples (label=1), making the classification task trivial.

**Impact**:
- Current training results are **meaningless** - models just learned to always predict class 1
- All reported accuracies (KG: 100%, Planning: 85%, Causal: 53%) are **not real performance**
- Need to stop current runs and fix datasets before retraining

---

## Red Flag #1: 100% Positive Labels (CONFIRMED)

### Knowledge Graph Dataset
- **Label range**: [0.51, 1.0] (all >0.5 threshold)
- **Distribution**: Class 0: 0%, Class 1: 100%
- **Root cause**: All `SemanticTriple.confidence` values generated in high ranges:
  - `random.uniform(0.7, 0.95)` (creative triples)
  - `random.uniform(0.8, 1.0)` (biographical triples)
  - `random.uniform(0.85, 0.99)` (type triples)
  - `random.uniform(0.9, 1.0)` (subclass triples)
- **Collate bug**: Converts continuous labels to binary via `int(label > 0.5)` → always 1
- **File**: `/Users/preston/Projects/nsm-kg/nsm/data/knowledge_graph_dataset.py:451-492`

### Causal Dataset
- **Distribution**: Class 0: 0%, Class 1: 100%
- **Total graphs**: 669
- **Same bug**: Only generates positive examples
- **File**: `/Users/preston/Projects/nsm-causal/nsm/data/causal_dataset.py`

### Planning Dataset
- **Distribution**: Class 0: 0%, Class 1: 100%
- **Total graphs**: 2,836
- **Same bug**: Only generates positive examples
- **File**: `/Users/preston/Projects/nsm-planning/nsm/data/planning_dataset.py`

---

## Red Flag #2: No Baselines (CONFIRMED)

Current training logs show **zero baseline comparisons**:
- No random baseline (should be 50% for balanced, 100% for all-positive!)
- No standard GCN baseline
- No comparison with simpler models

**What we thought were good results**:
- KG: 100% accuracy → Actually just "always predict 1" (random would get 100%!)
- Planning: 85% accuracy → Meaningless without seeing random baseline
- Causal: 53% accuracy → Worse than random (50%) but random would also get 100% here!

---

## Red Flag #3: Dataset Sizes (CONFIRMED)

Actual sizes discovered:

| Domain   | Train | Val  | Total | Notes                          |
|----------|-------|------|-------|--------------------------------|
| Causal   | 278   | 70   | 348   | **TINY** - explains instability|
| Planning | 22,370| 5,593| 27,963| **LARGE** - most robust        |
| KG       | 400   | 100  | 500   | **MEDIUM**                     |

**Issues**:
- Causal dataset is far too small (278 examples) for 100-epoch training
- Planning dataset is 80x larger than Causal - unfair comparison
- No consistent dataset size across domains

---

## Current Training Status

**All three runs are INVALID and should be stopped**:

1. **Causal** (PID 54487): Epoch ~60/100, 53% accuracy
   - Worse than 50% random, but random would actually get 100% here!

2. **Planning** (PID 58793): Epoch ~30/100, 85% accuracy
   - Looks good but meaningless - just learning "always predict 1"

3. **KG** (PID 67175): Epoch ~20/100, 100% accuracy
   - Perfect score because task is trivial (all labels are 1)

---

## Root Cause Analysis

### Why All Datasets Have This Bug

All three datasets inherit from `BaseSemanticTripleDataset` and implement `generate_labels()`, but they all make the same mistake:

1. **Synthetic generation bias**: Only create "valid" triples (high confidence)
2. **No negative sampling**: Never generate invalid/false triples
3. **Threshold conversion**: Collate functions convert continuous confidence → binary via `>0.5`

### KG-Specific Implementation

```python
# knowledge_graph_dataset.py:576-591
def generate_labels(self, idx: int) -> torch.Tensor:
    """Generate link prediction labels."""
    triple = self.triples[idx]
    # Use confidence as continuous label
    return torch.tensor([triple.confidence], dtype=torch.float32)
```

All generated triples have `confidence ∈ [0.7, 1.0]`, so after threshold:
```python
# train_kg.py:124 (collate_fn)
labels = torch.tensor([int(item[1].item() > 0.5) for item in batch_list], dtype=torch.long)
# Result: ALL labels become 1
```

---

## Required Fixes

### Priority 1: Fix Label Generation

**For ALL datasets**, implement proper negative sampling:

1. **KG Link Prediction**:
   - Generate corrupted triples (replace head/tail with random entity)
   - Label: 1 for true triples, 0 for corrupted triples
   - Target distribution: 50/50 balanced

2. **Planning Goal Achievement**:
   - Generate both achievable and unachievable plans
   - Label: 1 for achievable, 0 for missing preconditions
   - Target distribution: 50/50 balanced

3. **Causal Treatment Effectiveness**:
   - Generate both effective and ineffective treatments
   - Label: 1 for effective, 0 for no effect
   - Target distribution: 50/50 balanced

### Priority 2: Add Baselines

For each domain, add:
1. **Random baseline**: Should be 50% for balanced datasets
2. **Standard GCN baseline**: GCN without NSM hierarchical operations
3. **MLP baseline**: Simple feedforward on graph features

### Priority 3: Standardize Dataset Sizes

Use consistent sizes across domains:
- Train: 1,000 examples (minimum)
- Val: 250 examples
- Test: 250 examples
- Total: 1,500 examples per domain

---

## Recommended Actions

### Immediate (Now)

1. **STOP all current training runs** - results are invalid
2. **Document this finding** in Linear NSM-10 issue
3. **Create hotfix branches** for each dataset

### Short-term (Today)

1. **Fix dataset generation** with proper negative sampling
2. **Add balanced label checking** in dataset `__init__`:
   ```python
   labels = [self[i][1] for i in range(len(self))]
   class_0_pct = labels.count(0) / len(labels)
   assert 0.4 <= class_0_pct <= 0.6, f"Imbalanced: {class_0_pct:.1%} class 0"
   ```
3. **Implement baseline models** for comparison

### Medium-term (This Week)

1. **Rerun all training** with fixed datasets
2. **Compare against baselines** (random, GCN, MLP)
3. **Analyze which domain has most hierarchical structure**
4. **Write NSM-10 completion report** with fair comparison

---

## Lessons Learned

1. **Always check label distribution** before training
2. **Always compare against baselines** (especially random!)
3. **Validate synthetic data generation** with statistical tests
4. **Unit test dataset properties** (balance, diversity, coverage)

---

## Files Affected

### Need Immediate Fixes
- `/Users/preston/Projects/nsm-kg/nsm/data/knowledge_graph_dataset.py`
- `/Users/preston/Projects/nsm-causal/nsm/data/causal_dataset.py`
- `/Users/preston/Projects/nsm-planning/nsm/data/planning_dataset.py`

### Need Baseline Implementations
- `/Users/preston/Projects/nsm-kg/experiments/baseline_gcn.py` (new)
- `/Users/preston/Projects/nsm-causal/experiments/baseline_gcn.py` (new)
- `/Users/preston/Projects/nsm-planning/experiments/baseline_gcn.py` (new)

### Need Updates
- All `train_*.py` scripts to report random baseline
- All collate functions (already correct, just need balanced data)

---

## Next Steps

See NSM-10 Linear issue for implementation plan and timeline.
