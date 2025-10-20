# NSM-10 Dataset Evaluation Plan & Pruning Strategy

**Date**: 2025-10-20  
**Status**: Datasets Complete, Models Pending  
**Decision Gate**: End of Week 3 (after training results available)

---

## 🎯 Executive Summary

**Current State**: All three dataset branches (Planning, KG, Causal) fully implemented with 73 tests passing.

**Critical Blocker**: **No model components exist yet** (NSM-17, 16, 15) - cannot train or evaluate.

**Recommendation**: 
1. ✅ **Keep all three domains** (complementary validation, low marginal cost)
2. ⏸️ **Defer evaluation until Week 3** (after models implemented)
3. 🎯 **Focus on NSM-17 (R-GCN) immediately** (blocks all progress)
4. 📊 **Make pruning decision with empirical data** (not assumptions)

---

## 📅 Evaluation Timeline (Gated Approach)

### **NOW → Day 7 (Week 2): Foundation Implementation**

**GATE 1: Can we train?**

```
Required Components:
├─ NSM-17 (R-GCN): Message passing ← CRITICAL PATH
├─ NSM-16 (Coupling): Invertible transforms
├─ NSM-15 (Confidence): Base semirings
└─ NSM-14 (Training): Training loop

Status: 0/4 complete
Timeline: 7-10 days
Pass Criteria: All four components working, can run train.py
```

**Actions**:
- [ ] Implement NSM-17 in main branch (5-7 days)
- [ ] Implement NSM-16 in parallel (3-5 days)
- [ ] Implement NSM-15 in parallel (4-6 days)
- [ ] Integrate into NSM-14 training loop

**GATE 1 Pass**: Working model that can train on datasets  
**GATE 1 Fail**: Debug components, extend timeline

---

### **Day 8 (End Week 2): Dataset Quality Check**

**GATE 2: Are datasets high quality?**

```bash
# Run on each worktree
cd /Users/preston/Projects/nsm-planning
python scripts/check_dataset_quality.py > results/planning_stats.txt

cd /Users/preston/Projects/nsm-kg
python scripts/check_dataset_quality.py > results/kg_stats.txt

cd /Users/preston/Projects/nsm-causal
python scripts/check_dataset_quality.py > results/causal_stats.txt
```

**Metrics to Check**:

| Metric | Healthy Range | Action if Outside |
|--------|---------------|-------------------|
| L1/L2 Ratio | 0.3 - 0.7 | Regenerate with different parameters |
| Confidence Mean | 0.6 - 0.9 | Adjust confidence generation |
| Confidence Std | 0.1 - 0.3 | Increase variance if too narrow |
| Isolated Components | <5% | Fix graph generation logic |
| Predicate Entropy | >2.0 | Add predicate diversity |
| Avg Node Degree | 3 - 10 | Adjust edge generation |

**GATE 2 Pass**: All datasets meet quality criteria  
**GATE 2 Fail**: Fix dataset generation, regenerate

---

### **Day 9-10 (Week 3): Baseline Training**

**GATE 3: What's the baseline?**

```python
# Simple GNN (no NSM architecture)
class BaselineGNN(nn.Module):
    def __init__(self, hidden_dim=64):
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

# Train for 50 epochs on each dataset
for domain in ['planning', 'kg', 'causal']:
    baseline_results[domain] = train_baseline(domain)
```

**Target Baseline Performance**:
- **Too easy** (>90%): Dataset may have ceiling effect
- **Good** (60-80%): Room for NSM to improve
- **Too hard** (<50%): May be noisy or poorly defined

**GATE 3 Pass**: Baseline 60-80% on all domains  
**GATE 3 Adjust**: If outside range, tune dataset difficulty

---

### **Day 11-13 (Week 3): Full NSM Training**

**GATE 4: Does NSM work?**

```bash
# Full architecture with WHY/WHAT, confidence, coupling
cd /Users/preston/Projects/nsm-planning
python train_nsm.py --config configs/planning.yaml --epochs 50
# Save results/planning_nsm.json

cd /Users/preston/Projects/nsm-kg
python train_nsm.py --config configs/kg.yaml --epochs 50
# Save results/kg_nsm.json

cd /Users/preston/Projects/nsm-causal
python train_nsm.py --config configs/causal.yaml --epochs 50
# Save results/causal_nsm.json
```

**Success Criteria** (from NSM-10):

| Metric | Threshold | Planning | KG | Causal |
|--------|-----------|----------|-----|--------|
| Task Accuracy | >80% | ? | ? | ? |
| Reconstruction Error | <20% | ? | ? | ? |
| Calibration (ECE) | <0.1 | ? | ? | ? |
| 5-Hop Accuracy | >70% | ? | ? | ? |
| vs Baseline | +5-10% | ? | ? | ? |

**GATE 4 Outcome**:
- **3/3 pass**: Excellent! Keep all, proceed to explorations
- **2/3 pass**: Good! Keep 2, investigate 3rd before pruning
- **1/3 pass**: Concerning. Debug before explorations
- **0/3 pass**: Critical architecture issues

---

### **Day 14 (Week 3 End): Decision Point**

**PRUNING DECISION FRAMEWORK**

```python
def should_prune_domain(results):
    """Data-driven pruning decision."""
    
    # Quantitative criteria
    accuracy_threshold = results['accuracy'] < 0.75
    significantly_worse = (results['accuracy'] < 
                          max_accuracy - 0.15)
    
    # Qualitative criteria
    architecture_mismatch = (
        # Root cause analysis from debugging
        # Not just hyperparameter tuning issue
        # Would require substantial architecture changes
    )
    
    return (accuracy_threshold and 
            significantly_worse and 
            architecture_mismatch)

# Example outcomes
if all_pass:
    decision = "KEEP ALL THREE - proceed to NSM-12, NSM-11"
    
elif two_pass:
    decision = "KEEP TWO - investigate third, defer pruning 1 week"
    
elif one_pass:
    decision = "ARCHITECTURE ISSUES - debug before explorations"
```

**Document decision** in:
- NSM-10 (this issue)
- NSM-20 (main development)
- Create NSM-10-EVALUATION-REPORT.md

---

## 🔬 Evaluation Checklist (Execute in Order)

### ☐ Phase 1: Pre-Training Validation (Can Do Now)

**Dataset Statistics** (1 day):
```bash
# Check each worktree
for dir in nsm-planning nsm-kg nsm-causal; do
    cd /Users/preston/Projects/$dir
    python -c "
from nsm.data.* import *
dataset = ...Dataset(num_graphs=100)
# Print statistics
print(f'Total triples: {len(dataset.all_triples)}')
print(f'L1/L2 ratio: {l1_count / l2_count}')
print(f'Confidence mean/std: {mean}, {std}')
print(f'Graph connectivity: {stats}')
"
done
```

**Visual Inspection** (1 hour):
```bash
# Run examples, spot check quality
cd /Users/preston/Projects/nsm-planning
python examples/planning_example.py
# Manually review 5-10 generated scenarios
# Verify semantic coherence
```

---

### ☐ Phase 2: Baseline Training (After NSM-17 Complete)

**Simple GNN Baseline** (2-3 days):

```bash
# Create baseline trainer
python scripts/train_baseline.py \
  --dataset planning \
  --epochs 50 \
  --output results/planning_baseline.json

# Repeat for KG and Causal
```

**Establish benchmarks**:
- What accuracy is achievable without NSM architecture?
- How fast does training converge?
- What's the difficulty level?

---

### ☐ Phase 3: Full NSM Training (After NSM-14 Complete)

**Complete Architecture** (3-4 days):

```bash
# Train with full WHY/WHAT, confidence, coupling
python scripts/train_nsm.py \
  --dataset planning \
  --epochs 50 \
  --lambda_recon 0.1 \
  --output results/planning_nsm.json

# Repeat for KG and Causal
```

**Collect comprehensive metrics**:
- Task accuracy
- Reconstruction error (NSM-specific)
- Confidence calibration
- Multi-hop reasoning
- Training curves
- Gradient norms

---

### ☐ Phase 4: Comparative Analysis (Week 3 End)

**Generate comparison tables**:

```bash
cd /Users/preston/Projects/NSM
python scripts/compare_domains.py \
  nsm-planning/results/planning_nsm.json \
  nsm-kg/results/kg_nsm.json \
  nsm-causal/results/causal_nsm.json \
  --output NSM-10-EVALUATION-REPORT.md
```

**Statistical significance testing**:
- Are accuracy differences significant? (t-test, p<0.05)
- Which domain has lowest variance? (most stable)
- Any domain-specific failure patterns?

---

### ☐ Phase 5: Pruning Decision (Week 3 End)

**Document decision** with:

```markdown
## NSM-10 Pruning Decision

### Results Summary
| Domain | Accuracy | Reconstruction | Calibration | Decision |
|--------|----------|----------------|-------------|----------|
| Planning | 87% | 15% | 0.08 | ✅ KEEP |
| KG | 73% | 18% | 0.11 | ⚠️ INVESTIGATE |
| Causal | 84% | 22% | 0.09 | ✅ KEEP |

### Decision Rationale
[Why each domain kept/pruned]

### Root Cause Analysis
[For any failed domains, explain why]

### Next Steps
[Multi-domain training strategy OR architecture fixes]
```

---

## 💡 Key Insights

### Parallel Implementation Was Correct Decision

**Benefits realized**:
- ✅ All three domains ready simultaneously
- ✅ Each provides unique validation
- ✅ Can evaluate multi-domain generalization
- ✅ Investment: ~5,350 lines, 73 tests (manageable)

**Cost was reasonable**: 
- ~3-5 days per domain
- ~15 days total (would've been 15 days for one anyway due to learning curve)
- High-quality implementations (94-98% test coverage)

---

### Model Components are Critical Path

**Cannot evaluate until**:
- R-GCN exists (message passing)
- Training loop exists (optimization)
- WHY/WHAT exists (reconstruction)

**Current blocker**: NSM-17 (R-GCN)

**Timeline impact**: 
- Week 2: Implement models
- Week 3: Evaluate datasets
- Week 4: Make decisions

---

### Evaluation Should Be Conservative

**Don't prune hastily**:
- Failures reveal architectural insights
- May be fixable with tuning
- Scientific value in reporting challenges
- Multi-domain training may help struggling domains

**Prune only if**:
- Clear architecture-domain mismatch
- Debugging effort exceeds benefit
- Other domains strongly outperform

---

## 🎯 Success Definition

**Minimum Acceptable**: 2/3 domains >80% accuracy
- Demonstrates domain-generality
- Validates core architecture
- Publishable result

**Ideal**: 3/3 domains >80% accuracy
- Strong validation
- Architecture truly general
- High-impact publication

**Red Flag**: 0-1 domains >80% accuracy
- Fundamental architecture issues
- Delay Phase 2 (scaling to 6 levels)
- Debug and stabilize first

---

## 📊 Current Status Summary

| Component | Status | Blocking Evaluation? |
|-----------|--------|---------------------|
| Planning Dataset | ✅ Complete (25 tests) | No |
| KG Dataset | ✅ Complete (21 tests) | No |
| Causal Dataset | ✅ Complete (27 tests) | No |
| R-GCN (NSM-17) | ❌ Not started | **YES - CRITICAL** |
| Coupling (NSM-16) | ❌ Not started | **YES** |
| Confidence (NSM-15) | ❌ Not started | **YES** |
| Training (NSM-14) | ❌ Not started | **YES** |

**Evaluation possible**: Week 3 (after models complete)  
**Pruning decision**: End Week 3 (after empirical results)  
**Current focus**: Implement NSM-17 immediately

---

**Generated**: 2025-10-20  
**For**: NSM-10 Dataset Domain Exploration Evaluation

