# NSM-32: Full 6-Level Chiral Dual-Trifold Architecture - Design Document

**Date**: October 21, 2025
**Status**: Design Complete - Ready for Implementation
**Linear Issue**: [NSM-32](https://linear.app/imajn/issue/NSM-32)

---

## Quick Reference

This document provides a high-level overview. **Full detailed specifications are in Linear NSM-32 comments (9 comprehensive sections).**

---

## Architecture Overview

```
Upper Trifold (WHY):  L1 → L2 → L3  (concrete → abstract, bottom-up)
                       ↕    ↕    ↕
                    Hinge1 Hinge2 Hinge3  (Fusion-based exchange)
                       ↕    ↕    ↕
Lower Trifold (WHAT): L6 → L5 → L4  (abstract → concrete, top-down)
```

**6 Levels**:
- L1: Environment/Perception (most concrete, ~1000 nodes)
- L2: Actions/Behavior (~500 nodes)
- L3: Capabilities/Skills (~250 nodes)
- L4: Plans/Beliefs (~250 nodes)
- L5: Goals/Identity (~500 nodes)
- L6: Purpose/Mission (most abstract, learned prior)

**3 Fusion Hinges** (proven mechanism from NSM-31):
1. L1 ↔ L6: Environment ↔ Mission
2. L2 ↔ L5: Behavior ↔ Identity
3. L3 ↔ L4: Capability ↔ Beliefs

---

## Key Design Decisions

### ✅ Validated from Phase 1.5

1. **Fusion > Attention**: Simple weighted fusion beats complex cross-attention
   - Phase 1.5 results: Fusion 51.26% acc, Attention 53.10% acc but collapsed
   - Fusion achieved 29.60% balance delta (PASSED), Attention 87.48% (FAILED)

2. **Learnable Mixing Weights**: Per-dimension α and β parameters
   - Provides implicit regularization preventing class collapse
   - Simpler than attention (48% fewer parameters)

3. **Stable Training**: Fusion showed smooth convergence vs wild oscillations

### 🆕 New for 6-Level

1. **Size Alignment at Hinges**: L1↔L6 and L3↔L4 have mismatched node counts
   - Solution: Adaptive pooling + interpolation
   - Broadcast smaller to match larger when needed

2. **Scale Normalization**: Normalize features to [0,1] before exchange
   - Prevents gradient explosion from scale mismatches
   - Denormalize after exchange to restore original scale

3. **Multi-Level Predictions**: 3 prediction heads + ensemble
   - Auxiliary training signals from L1, L2, L3
   - Final prediction: average of all 3 heads

4. **Triple Cycle Consistency**: 3 reconstruction losses
   - Upper trifold: L1 → L3 → L1
   - Lower trifold: L6 → L4 → L6
   - Cross-trifold: L1 ↔ L6 consistency

---

## Success Criteria

**Primary** (Must Pass):
- ✅ Accuracy ≥ 55% on Planning domain (vs 3-level: 51.26%)
- ✅ Class Balance Δ < 40% (vs 3-level: 29.60%)
- ✅ All 3 hinges contribute (ablation test)

**Secondary**:
- Cycle consistency < 0.3 (tighter than 3-level ~0.91)
- Training stability (monotonic loss decrease)
- Interpretable level hierarchy

---

## Implementation Roadmap

**Week 1**: Core Architecture
- Days 1-2: Implement FullChiralModel with size alignment
- Day 3: Composite loss function + training loop
- Day 4: Debug and unit tests
- Day 5: Initial Modal GPU validation

**Week 2**: Validation & Ablation
- Days 1-2: Full validation on Planning domain
- Day 3: Multi-domain (Causal, KG)
- Days 4-5: Ablation studies

**Week 3**: Analysis & Optimization
- Days 1-2: Result analysis
- Days 3-4: Hyperparameter tuning (if needed)
- Day 5: Final validation + documentation

**Estimated Cost**: $12-15 GPU

---

## Risk Mitigation

**Critical Risks**:
1. **Size Mismatch at Hinges** (60% prob, High impact)
   - Mitigation: Adaptive pooling + interpolation fallback

2. **Gradient Vanishing/Explosion** (40% prob, High impact)
   - Mitigation: Gradient clipping, residual connections, layer norm

3. **Class Collapse** (30% prob, Medium impact)
   - Mitigation: Diversity loss, class weighting, focal loss

**Contingency**: If complete failure, revert to proven 3-level fusion

---

## Technical Specifications

**Parameters**: ~180K (vs 3-level: 44K, attention: 85K)

**Key Components**:
- 6 R-GCN layers (message passing)
- 2 pooling operators (upper trifold)
- 2 unpooling operators (lower trifold)
- 3 fusion hinges (size-aligned, scale-normalized)
- 4 prediction heads (L1, L2, L3, ensemble)
- 2 reconstruction layers (cycle consistency)

**Loss Function**:
```
L_total = L_task_final + 0.3·L_task_aux +
          0.01·(L_cycle_upper + L_cycle_lower + L_cycle_cross) +
          [optional: 0.05·L_diversity]
```

---

## References

**Linear Issue**: NSM-32 with 9 detailed design comments:
1. Architectural Design Overview
2. Fusion Hinge Exchange Mechanism
3. Normalization Inversion: Scale Matching Between Trifolds
4. Forward Pass Execution Flow
5. Training Strategy & Loss Function
6. Validation & Ablation Studies
7. Implementation Roadmap & Technical Specifications
8. Risk Analysis & Mitigation Strategies
9. Complete Architecture Summary & Quick Reference

**Related Documents**:
- `notes/CHIRAL_ARCHITECTURE.md` - 3-level minimal design
- `notes/FULL_CHIRAL_6LEVEL.md` - Original 6-level specification
- `notes/CHIRAL_VARIANT_COMPARISON.md` - Why fusion won over attention
- NSM-31 - Phase 1.5 validation results

**Code References**:
- `nsm/models/chiral.py` - MinimalChiralModel (3-level, working)
- `experiments/modal_chiral_validation.py` - Validation infrastructure

---

## Next Steps

1. ✅ Design complete (this document + Linear NSM-32)
2. ⏳ Review by another agent (if needed)
3. ⏳ Implement FullChiralModel
4. ⏳ Validate on Planning domain
5. ⏳ Multi-domain validation + ablation
6. ⏳ Analysis and documentation

**Status**: Ready for implementation to begin.

**Implementation Owner**: TBD

**Estimated Completion**: 3 weeks from start
