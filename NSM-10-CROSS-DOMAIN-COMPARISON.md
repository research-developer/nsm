# NSM-10 Cross-Domain Comparison

**Date**: 2025-10-20
**Linear Issue**: [NSM-10](https://linear.app/imajn/issue/NSM-10)
**Status**: ✅ All three dataset domains implemented

---

## Executive Summary

Successfully implemented three parallel dataset exploration branches for NSM Phase 1:

1. **Planning Domain** (`dataset-planning` branch)
2. **Knowledge Graph Domain** (`dataset-knowledge-graph` branch)
3. **Causal Reasoning Domain** (`dataset-causal` branch)

All three implementations:
- Extend `BaseSemanticTripleDataset` from NSM-18
- Include comprehensive test suites (21-27 tests each)
- Provide domain-specific evaluation metrics
- Include interactive example scripts
- Are ready for comparative evaluation

**Total Implementation**: ~5,350 lines of production code + tests across three branches

---

## Implementation Overview

| Aspect | Planning | Knowledge Graph | Causal |
|--------|----------|----------------|--------|
| **Branch** | `dataset-planning` | `dataset-knowledge-graph` | `dataset-causal` |
| **Worktree** | `../nsm-planning` | `../nsm-kg` | `../nsm-causal` |
| **Dataset Class** | `PlanningTripleDataset` | `KnowledgeGraphTripleDataset` | `CausalTripleDataset` |
| **Code (lines)** | 1,741 | 2,000 | 1,856 |
| **Tests** | 25 tests | 21 tests | 27 tests |
| **Test Coverage** | Not specified | 98% | 94% |
| **Commits** | 5 commits | 5 commits | 5 commits |

---

## Dataset Characteristics

### Scale and Structure

| Property | Planning | Knowledge Graph | Causal |
|----------|----------|----------------|--------|
| **Total Scenarios** | 1,000 problems | 20,000 triples | 2,000 scenarios |
| **Entities** | ~315 | ~5,000 | ~1,400 |
| **Predicates** | 16 types | 50+ types | 20 types |
| **Total Triples** | ~515 (20 problems) | 20,000 | ~14,000 |
| **L1/L2 Distribution** | 69%/31% | 87%/13% | 50%/50% |

### Confidence Distribution

| Domain | Confidence Range | Average | Variance | Interpretation |
|--------|-----------------|---------|----------|----------------|
| **Planning** | 0.7 - 1.0 | 0.903 | Low | High certainty in procedural knowledge |
| **Knowledge Graph** | 0.5 - 1.0 | 0.77 | High | Partial observability, uncertain facts |
| **Causal** | Variable | Not specified | Very High | Strength of causal effects |

### Hierarchical Structure

| Domain | Level 1 (Concrete) | Level 2 (Abstract) | Hierarchy Naturalness |
|--------|-------------------|-------------------|---------------------|
| **Planning** | Actions, Environment | Goals, Capabilities | ⭐⭐⭐⭐⭐ Very Natural |
| **Knowledge Graph** | Facts, Instances | Types, Categories | ⭐⭐⭐ Moderate |
| **Causal** | Observations, Events | Mechanisms, Responses | ⭐⭐⭐⭐ Natural |

---

## Domain-Specific Features

### Planning Domain

**Triple Examples**:
```python
# Level 1
("robot", "move_to", "location_A", 0.9, level=1)
("robot", "pick_up", "block_1", 0.85, level=1)
("location_A", "contains", "block_1", 0.95, level=1)

# Level 2
("robot", "achieve", "stack_ABC", 0.7, level=2)
("stack_ABC", "requires", "pick_up", 0.8, level=2)
```

**Domain Properties**:
- ✅ Hierarchical goal decomposition
- ✅ Temporal ordering with prerequisites
- ✅ Clear ground truth (valid/invalid sequences)
- ✅ Testable goal achievement

**Evaluation Metrics**:
- Goal achievement rate
- Invalid sequence detection (Acc, Prec, Rec, F1)
- Temporal ordering accuracy
- Capability coverage
- Decomposition accuracy

**Expected Strengths**:
- Strong hierarchical structure (WHY/WHAT alignment)
- Clear procedural reasoning
- Interpretable failures

**Expected Challenges**:
- Low predicate diversity (16 types)
- May be too structured (overly deterministic)

---

### Knowledge Graph Domain

**Triple Examples**:
```python
# Level 1
("Albert_Einstein", "born_in", "Ulm", 0.99, level=1)
("Albert_Einstein", "won", "Nobel_Prize_1921", 0.99, level=1)
("Ulm", "located_in", "Germany", 1.0, level=1)

# Level 2
("Albert_Einstein", "instance_of", "Physicist", 0.95, level=2)
("Physicist", "typically_wins", "Scientific_Awards", 0.7, level=2)
```

**Domain Properties**:
- ✅ Entity-centric reasoning
- ✅ Rich predicate vocabulary (50+ types)
- ✅ Type hierarchies (instance_of, subclass_of)
- ✅ Partial observability

**Evaluation Metrics**:
- Link prediction (Hits@K, MRR, Mean Rank)
- Analogical reasoning (A:B :: C:?)
- Type consistency (Prec, Rec, F1)
- Multi-hop reasoning
- Calibration (ECE, MCE)

**Expected Strengths**:
- Rich relational diversity (tests R-GCN)
- Multi-hop reasoning paths
- Entity-centric interpretability

**Expected Challenges**:
- Weaker hierarchical structure
- Random generation may create noise
- May need deeper hierarchies

---

### Causal Domain

**Triple Examples**:
```python
# Level 1
("aspirin", "taken_by", "patient_42", 0.9, level=1)
("patient_42", "has_symptom", "headache", 0.95, level=1)
("patient_42", "symptom_reduced", "headache", 0.8, level=1)

# Level 2
("aspirin", "causes", "pain_reduction", 0.85, level=2)
("pain_reduction", "treats", "headache", 0.9, level=2)
```

**Domain Properties**:
- ✅ Causal relationships (treatment → effect)
- ✅ Counterfactual reasoning support
- ✅ Confounder modeling
- ✅ Confidence as effect size

**Evaluation Metrics**:
- Counterfactual accuracy
- Confounder detection (F1)
- Effect size estimation (MAE)
- Causal vs correlation distinction
- Intervention prediction accuracy
- Calibration (ECE)

**Expected Strengths**:
- Tests causal reasoning (frontier AI capability)
- Counterfactual queries
- Confidence has epistemic meaning
- WHY/WHAT maps to cause/effect

**Expected Challenges**:
- Confounders add complexity
- May require specialized inference
- Balance treatment assignment randomness

---

## Comparative Analysis

### Predicate Diversity

```
Planning:        16 types  [█░░░░░░░░░] Low
Causal:          20 types  [██░░░░░░░░] Medium
Knowledge Graph: 50+ types [█████░░░░░] High
```

**Insight**: KG will stress-test R-GCN basis decomposition most effectively

---

### Temporal Structure

```
Planning:        Strong    [█████████░] Explicit prerequisites
Causal:          Medium    [█████░░░░░] Treatment → outcome
Knowledge Graph: Weak      [██░░░░░░░░] Mostly atemporal facts
```

**Insight**: Planning tests temporal reasoning, KG tests static knowledge

---

### Hierarchy Naturalness

```
Planning:        Very High [██████████] Goals → actions natural
Causal:          High      [████████░░] Mechanisms → observations
Knowledge Graph: Moderate  [██████░░░░] Types → instances
```

**Insight**: Planning aligns best with BDI-HTN-HRL framework

---

### Confidence Variance

```
Planning:        Low       [███░░░░░░░] 0.7-1.0 range
Knowledge Graph: High      [████████░░] 0.5-1.0 range
Causal:          Very High [██████████] Variable effect sizes
```

**Insight**: Causal tests confidence propagation most rigorously

---

## Expected Stress Tests

| Component | Planning Tests | KG Tests | Causal Tests |
|-----------|---------------|----------|--------------|
| **R-GCN** | 16 edge types | 50+ edge types ⭐ | 20 edge types |
| **WHY/WHAT** | Goal decomposition ⭐ | Type abstraction | Cause → observation |
| **Confidence** | Stable values | Partial observability ⭐ | Effect sizes ⭐ |
| **Multi-hop** | Prerequisite chains | Entity paths ⭐ | Causal chains |
| **Interpretability** | Action sequences ⭐ | Entity relations | Counterfactuals ⭐ |

⭐ = Domain expected to provide strongest test

---

## Evaluation Protocol

### Shared Metrics (All Domains)

```python
def evaluate_domain(model, dataset_name):
    return {
        'reconstruction_error': test_cycle_consistency(model, dataset),
        'task_accuracy': test_prediction_accuracy(model, dataset),
        'confidence_calibration': compute_ECE(model, dataset),
        'multi_hop_accuracy': test_reasoning_chains(model, dataset, k=5),
        'training_convergence': epochs_to_convergence(model, dataset),
        'interpretability': qualitative_assessment(model, dataset)
    }
```

### Domain-Specific Tests

**Planning**:
- ✓ Goal achievement rate
- ✓ Invalid sequence detection
- ✓ Temporal ordering correctness
- ✓ Capability coverage
- ✓ Decomposition accuracy

**Knowledge Graph**:
- ✓ Link prediction (Hits@10, MRR)
- ✓ Analogical reasoning
- ✓ Type consistency
- ✓ Multi-hop reasoning

**Causal**:
- ✓ Counterfactual accuracy
- ✓ Confounder detection
- ✓ Effect size estimation
- ✓ Causal vs correlational distinction
- ✓ Intervention prediction

---

## Decision Criteria

As specified in CLAUDE.md NSM-10:

| Criterion | Weight | Planning | Knowledge Graph | Causal |
|-----------|--------|----------|----------------|--------|
| **Task Accuracy** | 40% | Goal achievement | Link prediction | Counterfactual accuracy |
| **Calibration** | 20% | ECE | ECE | ECE |
| **Multi-hop** | 20% | Prerequisite chains | Entity paths | Causal chains |
| **Interpretability** | 20% | Action sequences | Entity relations | Counterfactuals |

**Success Criteria**:
- **Minimum**: ≥80% accuracy on 2/3 domains
- **Ideal**: ≥80% on all three with same hyperparameters
- **Failure**: Only works on one domain → architecture too specialized

---

## Implementation Status

### ✅ Planning Domain (`dataset-planning`)

**Files Created**:
- `nsm/data/planning_dataset.py` (515 lines)
- `nsm/evaluation/planning_metrics.py` (458 lines)
- `tests/data/test_planning_dataset.py` (493 lines)
- `examples/planning_example.py` (255 lines)

**Test Results**: 25/25 passing ✓

**Commits**: 5 logical commits

---

### ✅ Knowledge Graph Domain (`dataset-knowledge-graph`)

**Files Created**:
- `nsm/data/knowledge_graph_dataset.py` (682 lines)
- `nsm/evaluation/kg_metrics.py` (449 lines)
- `tests/data/test_kg_dataset.py` (394 lines)
- `examples/knowledge_graph_example.py` (216 lines)
- `KG_IMPLEMENTATION_SUMMARY.md` (245 lines)

**Test Results**: 21/21 passing ✓ (98% coverage)

**Commits**: 5 logical commits

---

### ✅ Causal Domain (`dataset-causal`)

**Files Created**:
- `nsm/data/causal_dataset.py` (513 lines)
- `nsm/evaluation/causal_metrics.py` (502 lines)
- `tests/data/test_causal_dataset.py` (488 lines)
- `examples/causal_example.py` (353 lines)
- `CAUSAL_DATASET_SUMMARY.md` (not specified)

**Test Results**: 27/27 passing ✓ (94% coverage)

**Commits**: 5 logical commits

---

## Integration Verification

All three datasets verified compatible with:

| NSM Component | Planning | KG | Causal |
|---------------|----------|-----|--------|
| **NSM-18** (BaseSemanticTripleDataset) | ✅ | ✅ | ✅ |
| **NSM-18** (GraphConstructor) | ✅ | ✅ | ✅ |
| **NSM-18** (TripleVocabulary) | ✅ | ✅ | ✅ |
| **NSM-17** (R-GCN edge types) | ✅ | ✅ | ✅ |
| **NSM-12** (Confidence propagation) | ✅ | ✅ | ✅ |
| **NSM-14** (Training loop ready) | ✅ | ✅ | ✅ |

---

## Next Steps

### 1. Run Evaluation Suite (Week 2-3)

```bash
# In each worktree
cd ../nsm-planning
python -m tests.evaluation_suite --dataset planning --output results/planning.json

cd ../nsm-kg
python -m tests.evaluation_suite --dataset knowledge_graph --output results/kg.json

cd ../nsm-causal
python -m tests.evaluation_suite --dataset causal --output results/causal.json
```

### 2. Comparative Analysis (Week 4)

```bash
# Back in main repo
python compare_results.py results/*.json
```

### 3. Document Findings

Create `results/NSM-10-EVALUATION-REPORT.md` with:
- Performance comparison table
- Domain-specific failure modes
- Architectural recommendations
- Multi-domain training strategy

### 4. Update Parent Issues

- Update NSM-10 with evaluation results
- Update NSM-20 with findings
- Inform design decisions for Phase 2

---

## Recommendations

### Expected Outcomes

**Most Likely**: All three domains work with minor hyperparameter tuning
- NSM architecture is domain-general by design
- BDI-HTN-HRL framework supports all three reasoning types

**If Planning >> Others**:
- Architecture may be over-specialized for procedural reasoning
- Consider adding attention mechanisms for relational reasoning

**If KG >> Others**:
- May need stronger temporal modeling (planning)
- May need causal inference layers (causal)

**If Causal >> Others**:
- Good sign for frontier AI capabilities
- Consider Pearl's do-calculus integration

### Multi-Domain Training

**Recommendation**: Train on all three domains sequentially or jointly

```python
class MultiDomainNSM:
    datasets = {
        'planning': PlanningTripleDataset(),
        'knowledge': KnowledgeGraphTripleDataset(),
        'causal': CausalTripleDataset()
    }

    # Multi-task learning with domain-specific heads
    # OR sequential transfer learning
    # OR ensemble approach
```

---

## Conclusion

✅ **All three dataset domains successfully implemented**

**Total Effort**:
- ~5,350 lines of code across three branches
- 73 comprehensive tests (25 + 21 + 27)
- Domain-specific evaluation metrics
- Interactive examples for each domain

**Status**: Ready for empirical evaluation and comparison

**Next**: Run evaluation suite, compare results, update NSM-20 with findings

---

## References

- **Linear Issue**: [NSM-10](https://linear.app/imajn/issue/NSM-10)
- **Parent Issue**: [NSM-20](https://linear.app/imajn/issue/NSM-20)
- **Planning Summary**: `../nsm-planning/` (25 tests)
- **KG Summary**: `../nsm-kg/KG_IMPLEMENTATION_SUMMARY.md` (21 tests)
- **Causal Summary**: `../nsm-causal/CAUSAL_DATASET_SUMMARY.md` (27 tests)

---

**Generated**: 2025-10-20
**Claude Code**: claude.ai/code

🤖 Generated with [Claude Code](https://claude.com/claude-code)
