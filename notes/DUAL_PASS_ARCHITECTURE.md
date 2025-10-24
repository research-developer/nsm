# Dual-Pass Bidirectional NSM Architecture

**Created**: 2025-10-21
**Purpose**: Address class collapse in 3-level NSM by using bidirectional dual-pass inference
**Status**: Design Document

---

## Problem Statement

Current 100-epoch training results show severe class collapse:
- **Planning**: 97% class 1 predictions (43% accuracy)
- **Causal**: 100% class 0 predictions (57% accuracy)
- **KG**: 79% class 1 bias (53% accuracy)

All domains fail to maintain balanced class predictions despite 50/50 training data.

---

## Root Cause Hypothesis

The **single round-trip architecture** (L1→L2→L3→L2→L1) has fundamental issues:

1. **Information Bottleneck**:
   - L3 has ~13-50% fewer nodes than L1 (pool_ratio = 0.5 applied twice)
   - Task-relevant discriminative features may be lost during abstraction
   - Reconstruction loss forces model to preserve structure, not task info

2. **Single Prediction Point**:
   - Only predicts from L3 (most abstract level)
   - No way to leverage concrete-level discriminative features
   - Abstract representation might not capture class-specific details

3. **Cycle Loss Dominance**:
   - Cycle loss weight = 0.01, but losses are ~0.72-0.79
   - Effective weight: 0.01 × 0.75 ≈ 0.0075 vs task loss ~0.69
   - Reconstruction objective competing with classification objective

---

## Proposed Solution: Dual-Pass Bidirectional Architecture

### Architecture Overview

```
Input: Graph G at L1 (concrete level)

PASS 1 - Bottom-Up Abstraction:
┌─────────────────────────────────────────────────────┐
│ L1 → WHY(L1→L2) → L2 → WHY(L2→L3) → L3 (abstract) │
│                                      ↓              │
│                           prediction_abstract      │
└─────────────────────────────────────────────────────┘

PASS 2 - Top-Down Refinement:
┌─────────────────────────────────────────────────────┐
│ L3 → WHAT(L3→L2) → L2' → WHAT(L2'→L1) → L1'       │
│                                      ↓              │
│                           prediction_concrete       │
└─────────────────────────────────────────────────────┘

FUSION:
┌─────────────────────────────────────────────────────┐
│ final_prediction = α·prediction_abstract +          │
│                    β·prediction_concrete             │
│                                                      │
│ where α + β = 1, learned via attention or fixed     │
└─────────────────────────────────────────────────────┘
```

### Key Differences from Current Architecture

| Aspect | Current (Single Pass) | Proposed (Dual Pass) |
|--------|----------------------|----------------------|
| **Predictions** | 1 (from L3 only) | 2 (from L3 and L1') |
| **Information flow** | L1→L3→L1 (round-trip) | L1→L3 + L3→L1 (bidirectional) |
| **Gradient paths** | 1 path through hierarchy | 2 independent paths |
| **Abstraction bias** | Single (depends on start) | Balanced (both directions) |
| **Cycle loss** | L1 vs L1_reconstructed | Optional (can use or remove) |

---

## Implementation Design

### 1. Modified Forward Pass

```python
def forward_dual_pass(
    self,
    x: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    edge_attr: Optional[Tensor] = None,
    batch: Optional[Tensor] = None
) -> Dict[str, Any]:
    """Dual-pass bidirectional forward.

    Returns:
        Dict with:
        - logits: Fused task predictions
        - logits_abstract: Predictions from bottom-up pass
        - logits_concrete: Predictions from top-down pass
        - cycle_loss: Optional reconstruction loss
        - fusion_weights: Learned α, β weights
    """

    # ===== PASS 1: Bottom-Up (L1 → L3) =====
    # L1 → L2
    x_l2, edge_index_l2, edge_attr_l2, perm_l2, score_l2 = \
        self.layer_1_2.why_operation(x, edge_index, edge_type, edge_attr, batch)

    batch_l2 = batch[perm_l2] if batch is not None else None
    edge_type_l2 = torch.zeros(edge_index_l2.size(1), dtype=torch.long, device=x.device)

    # L2 → L3
    x_l3, edge_index_l3, edge_attr_l3, perm_l3, score_l3 = \
        self.layer_2_3.why_operation(x_l2, edge_index_l2, edge_type_l2, edge_attr_l2, batch_l2)

    batch_l3 = batch_l2[perm_l3] if batch_l2 is not None else None

    # Predict from L3 (abstract representation)
    x_graph_abstract = self._global_pool(x_l3, batch_l3)
    logits_abstract = self.predictor_abstract(x_graph_abstract)  # New predictor head


    # ===== PASS 2: Top-Down (L3 → L1) =====
    # L3 → L2' (reconstructed)
    x_l2_recon = self.layer_2_3.what_operation(
        x_l3, perm_l3, batch_l2, original_num_nodes=x_l2.size(0)
    )

    # L2' → L1' (reconstructed)
    x_l1_recon = self.layer_1_2.what_operation(
        x_l2_recon, perm_l2, batch, original_num_nodes=x.size(0)
    )

    # Predict from L1' (concrete representation)
    x_graph_concrete = self._global_pool(x_l1_recon, batch)
    logits_concrete = self.predictor_concrete(x_graph_concrete)  # New predictor head


    # ===== FUSION =====
    # Option A: Fixed weights
    alpha = 0.5  # Equal weight to abstract and concrete
    beta = 0.5

    # Option B: Learned weights (attention)
    # fusion_input = torch.cat([x_graph_abstract, x_graph_concrete], dim=-1)
    # weights = self.fusion_attention(fusion_input)  # → [batch, 2]
    # alpha, beta = weights[:, 0:1], weights[:, 1:2]

    logits_fused = alpha * logits_abstract + beta * logits_concrete


    # ===== CYCLE LOSS (Optional) =====
    cycle_loss_l1 = F.mse_loss(x_l1_recon, x)
    cycle_loss_l2 = F.mse_loss(x_l2_recon, x_l2)
    cycle_loss = 0.7 * cycle_loss_l1 + 0.3 * cycle_loss_l2


    return {
        'logits': logits_fused,
        'logits_abstract': logits_abstract,
        'logits_concrete': logits_concrete,
        'cycle_loss': cycle_loss,
        'x_l1_recon': x_l1_recon,
        'x_l2': x_l2,
        'x_l3': x_l3,
        'fusion_weights': (alpha, beta)
    }
```

### 2. New Model Components

```python
class DualPassNSMModel(nn.Module):
    """NSM with dual-pass bidirectional inference."""

    def __init__(self, node_features, num_relations, num_classes, ...):
        super().__init__()

        # Existing hierarchical layers
        self.layer_1_2 = SymmetricHierarchicalLayer(...)
        self.layer_2_3 = SymmetricHierarchicalLayer(...)

        # NEW: Separate prediction heads for abstract and concrete
        self.predictor_abstract = nn.Sequential(
            nn.Linear(node_features, node_features // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(node_features // 2, num_classes)
        )

        self.predictor_concrete = nn.Sequential(
            nn.Linear(node_features, node_features // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(node_features // 2, num_classes)
        )

        # NEW: Optional learned fusion (attention-based)
        self.fusion_attention = nn.Sequential(
            nn.Linear(node_features * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
```

### 3. Modified Loss Function

```python
def compute_dual_pass_loss(output, labels, cycle_weight=0.01):
    """Combined loss for dual-pass architecture."""

    # Task losses for both predictions
    task_loss_abstract = F.cross_entropy(output['logits_abstract'], labels)
    task_loss_concrete = F.cross_entropy(output['logits_concrete'], labels)
    task_loss_fused = F.cross_entropy(output['logits'], labels)

    # Combined task loss (all three contribute)
    task_loss = (
        0.5 * task_loss_fused +      # Primary: fused prediction
        0.25 * task_loss_abstract +  # Auxiliary: abstract prediction
        0.25 * task_loss_concrete    # Auxiliary: concrete prediction
    )

    # Cycle loss (optional, can reduce or remove)
    cycle_loss = output['cycle_loss']

    # Total loss
    total_loss = task_loss + cycle_weight * cycle_loss

    return {
        'total_loss': total_loss,
        'task_loss': task_loss,
        'task_loss_abstract': task_loss_abstract,
        'task_loss_concrete': task_loss_concrete,
        'task_loss_fused': task_loss_fused,
        'cycle_loss': cycle_loss
    }
```

---

## Expected Benefits

### 1. **Reduced Class Collapse**
- **Two prediction points** with different biases
- Abstract predictor: Sees high-level patterns, might favor one class
- Concrete predictor: Sees detailed features, might favor other class
- Fusion balances both perspectives

### 2. **Better Gradient Flow**
- **Three task loss terms** (abstract + concrete + fused)
- Multiple gradient paths through hierarchy
- Less dependence on cycle loss for learning

### 3. **Complementary Information**
- **Abstract pass**: Captures global structure, relationships
- **Concrete pass**: Preserves local details, edge features
- **Fusion**: Combines strengths of both

### 4. **Interpretability**
- Can analyze which pass contributes more to predictions
- Fusion weights reveal if task needs abstraction or details
- Separate predictions help debug collapse issues

---

## Testing Strategy

### Phase 1: Unit Tests (Validate Architecture)

```python
def test_dual_pass_shapes():
    """Verify output shapes match expected."""
    model = DualPassNSMModel(...)
    output = model(x, edge_index, edge_type, batch=batch)

    assert output['logits'].shape == (batch_size, num_classes)
    assert output['logits_abstract'].shape == (batch_size, num_classes)
    assert output['logits_concrete'].shape == (batch_size, num_classes)
    assert output['cycle_loss'].ndim == 0  # Scalar

def test_dual_pass_gradients():
    """Verify gradients flow to all parameters."""
    model = DualPassNSMModel(...)
    output = model(x, edge_index, edge_type, batch=batch)
    loss = compute_dual_pass_loss(output, labels)
    loss['total_loss'].backward()

    # Check all components receive gradients
    assert model.layer_1_2.rgcn.weight.grad is not None
    assert model.predictor_abstract[0].weight.grad is not None
    assert model.predictor_concrete[0].weight.grad is not None

def test_dual_pass_fusion_weights():
    """Verify fusion weights sum to 1."""
    model = DualPassNSMModel(...)
    output = model(x, edge_index, edge_type, batch=batch)
    alpha, beta = output['fusion_weights']

    assert torch.allclose(alpha + beta, torch.ones_like(alpha))
    assert (alpha >= 0).all() and (alpha <= 1).all()
```

### Phase 2: Training Comparison

Run **side-by-side comparison** on GPU:

| Variant | Description | Hypothesis |
|---------|-------------|------------|
| **Baseline** | Current single-pass (control) | Class collapse baseline |
| **Dual-Pass (Equal)** | Dual-pass with α=β=0.5 | Balanced fusion reduces collapse |
| **Dual-Pass (Learned)** | Dual-pass with attention fusion | Adaptive weights find best balance |
| **Dual-Pass (No Cycle)** | Dual-pass with cycle_weight=0 | Remove reconstruction constraint |

**Quick validation** (10 epochs each):
```bash
modal run experiments/modal_dual_pass.py::validate_variants
```

**Full training** (100 epochs, winner only):
```bash
modal run experiments/modal_dual_pass.py::train_best_variant
```

### Phase 3: Metrics to Track

```python
metrics = {
    # Primary: Class balance
    'accuracy_class_0': ...,
    'accuracy_class_1': ...,
    'class_balance_delta': abs(acc_0 - acc_1),  # Target: < 0.1

    # Secondary: Prediction analysis
    'fusion_weight_mean': alpha.mean(),  # Which pass dominates?
    'abstract_accuracy': ...,  # Individual pass performance
    'concrete_accuracy': ...,

    # Tertiary: Ensemble effect
    'ensemble_gain': acc_fused - max(acc_abstract, acc_concrete),
    'prediction_diversity': disagreement_rate(pred_abstract, pred_concrete)
}
```

**Success Criteria**:
- ✅ Class balance delta < 0.1 (both classes within 10%)
- ✅ Overall accuracy > 60% (meaningful learning)
- ✅ Ensemble gain > 0 (fusion helps)
- ✅ Prediction diversity > 0.2 (passes capture different aspects)

---

## Implementation Plan

### Step 1: Create New Model File
**File**: `nsm/models/hierarchical_dual_pass.py`
- Copy `NSMModel` → `DualPassNSMModel`
- Add dual prediction heads
- Implement `forward_dual_pass()`
- Add fusion mechanism

**Estimated time**: 2 hours

### Step 2: Update Loss Computation
**File**: `nsm/training/trainer.py`
- Add `compute_dual_pass_loss()` function
- Modify trainer to handle dual-pass outputs
- Track new metrics (abstract/concrete accuracy)

**Estimated time**: 1 hour

### Step 3: Create Unit Tests
**File**: `tests/test_dual_pass.py`
- Shape validation
- Gradient flow verification
- Fusion weight constraints

**Estimated time**: 1 hour

### Step 4: Create Modal Training Script
**File**: `experiments/modal_dual_pass.py`
- Variant comparison function
- Side-by-side training
- Results aggregation

**Estimated time**: 30 minutes

### Step 5: Run Validation
```bash
# Quick 10-epoch validation on all variants
modal run experiments/modal_dual_pass.py::validate_variants

# Analyze results, pick best variant
python analyze_dual_pass_results.py

# Full 100-epoch training on winner
modal run experiments/modal_dual_pass.py::train_best_variant
```

**Total estimated time**: ~5 hours development + ~30 minutes GPU time

---

## Alternative Designs Considered

### Option B: Iterative Refinement (2 Full Round-Trips)
```
Pass 1: L1 → L2 → L3 → L2 → L1 → prediction_1
Pass 2: L1 → L2 → L3 → L2 → L1 → prediction_2 (refined)
```
**Rejected**: Doubles computation, more complex gradients, unclear benefit

### Option C: Multi-Scale Predictions (All Levels)
```
Predictions from L1, L2, AND L3
Fusion of all three
```
**Rejected**: Too many hyperparameters, overfitting risk, complexity

### Option D: Attention-Based Routing
```
Learnable routing between levels
Skip connections across hierarchy
```
**Rejected**: Breaks symmetric WHY/WHAT structure, harder to interpret

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Increased complexity** | Harder to debug | Extensive unit tests, clear logging |
| **Doubled parameters** | Overfitting on small datasets | Weight sharing between predictors |
| **Gradient conflicts** | Instability | Careful loss weighting, gradient clipping |
| **No improvement** | Wasted effort | Quick 10-epoch validation first |

---

## Success Metrics

**Minimum Viable Success**:
- ✅ No class collapse (both classes >40% accuracy each)
- ✅ Overall accuracy >55% (better than current best)
- ✅ Training stability (no divergence)

**Target Success**:
- ✅ Both classes 45-55% accuracy (balanced)
- ✅ Overall accuracy >70% (meaningful learning)
- ✅ Cycle loss <0.5 (better reconstruction)
- ✅ Fusion weights interpretable (clear preference)

**Stretch Success**:
- ✅ Outperforms 2-level baseline
- ✅ Transferable across all 3 domains
- ✅ Insights for future architecture improvements

---

## Next Steps

1. **Review this design** with stakeholders
2. **Implement Phase 1** (new model class)
3. **Run quick validation** (10 epochs on Planning domain)
4. **Iterate if needed** (adjust fusion, loss weights)
5. **Full training** on all domains if promising

---

**Generated**: 2025-10-21
**Author**: Claude Code + Preston
**Status**: Ready for Implementation
