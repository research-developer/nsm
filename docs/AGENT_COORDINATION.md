# NSM-25: Agent Coordination Strategy

This document outlines the parallel agent coordination strategy for full-scale training across three domain branches.

## Overview

NSM-25 coordinates parallel execution of 100-epoch training runs across three git worktree branches:
- `dataset-causal` (NSM-24): Causal Reasoning Domain
- `dataset-planning` (NSM-22): Planning & Goal Achievement Domain
- `dataset-knowledge-graph` (NSM-23): Knowledge Graph Completion Domain

## Agent Assignment

### Agent 1: Causal Domain (NSM-24)
**Branch**: `dataset-causal` at `/Users/preston/Projects/nsm-causal`

**Primary Task**: Run 100-epoch training on causal reasoning domain

**Responsibilities**:
1. Execute `bash experiments/run_full_training.sh --use-tensorboard`
2. Monitor training convergence and metrics
3. Run analysis notebook when training completes
4. Report results back to NSM-25

**Target Metrics**:
- Reconstruction error: <25%
- Intervention accuracy: ≥75%
- Counterfactual accuracy: ≥60%

**Configuration**:
```bash
Domain: Causal (NSM-24)
Relations: 20
Bases: 5 (75% compression)
Pool Ratio: 0.5
Dataset: 1,000 scenarios (800 train / 200 val)
```

**Success Criteria**:
- Training completes without errors
- All metrics exceed targets
- Results saved to `results/causal_100epoch/`

---

### Agent 2: Planning Domain (NSM-22)
**Branch**: `dataset-planning` at `/Users/preston/Projects/nsm-planning`

**Primary Task**: Run 100-epoch training on planning domain

**Responsibilities**:
1. Execute `bash experiments/run_full_training.sh --use-tensorboard`
2. Monitor goal achievement and temporal ordering
3. Run analysis notebook when training completes
4. Report results back to NSM-25

**Target Metrics**:
- Reconstruction error: <20%
- Goal achievement: ≥85%
- Temporal ordering: ≥80%
- Plan validity: ≥75%

**Configuration**:
```bash
Domain: Planning (NSM-22)
Relations: 16
Bases: 8 (50% compression)
Pool Ratio: 0.5
Dataset: 1,000 plans (800 train / 200 val)
```

**Success Criteria**:
- Training completes without errors
- All metrics exceed targets
- Results saved to `results/planning_100epoch/`

---

### Agent 3: Knowledge Graph Domain (NSM-23)
**Branch**: `dataset-knowledge-graph` at `/Users/preston/Projects/nsm-kg`

**Primary Task**: Run 100-epoch training on knowledge graph domain

**Responsibilities**:
1. Execute `bash experiments/run_full_training.sh --use-tensorboard`
2. Monitor link prediction and analogical reasoning
3. Run analysis notebook when training completes
4. Report results back to NSM-25

**Target Metrics**:
- Reconstruction error: <30%
- Hits@10: ≥70%
- MRR: ≥0.50
- Analogical reasoning: ≥60%

**Configuration**:
```bash
Domain: Knowledge Graph (NSM-23)
Relations: 66
Bases: 12 (81.8% compression)
Pool Ratio: 0.13
Dataset: 1,000 subgraphs (800 train / 200 val)
```

**Success Criteria**:
- Training completes without errors
- All metrics exceed targets
- Results saved to `results/kg_100epoch/`

---

## Coordination Protocol

### Phase 1: Setup (Week 1)
**Status**: ✅ COMPLETE

All agents should verify:
- [x] Experiment runner scripts exist and are executable
- [x] Results directories created
- [x] Analysis notebook templates available
- [x] Git worktrees properly configured

### Phase 2: Parallel Training (Week 2-3)
**Status**: 🔄 READY TO START

Each agent should:
1. Navigate to their assigned worktree
2. Activate conda environment: `conda activate nsm`
3. Run training script: `bash experiments/run_full_training.sh --use-tensorboard`
4. Monitor progress via TensorBoard
5. Report completion status to NSM-25

**Monitoring**:
- Check `results/<domain>_100epoch/*.log` for progress
- Use `tensorboard --logdir checkpoints/<domain>_full/` for visualization
- Expected runtime: ~4-6 hours per domain (CPU), ~1-2 hours (GPU)

**Checkpoints**:
- Automatic checkpointing every 10 epochs
- Best model saved based on validation loss
- Early stopping after 20 epochs without improvement

### Phase 3: Analysis (Week 3)
**Status**: ⏳ PENDING (after training)

Each agent should:
1. Copy `notebooks/domain_analysis_template.ipynb` to domain branch
2. Set `DOMAIN` variable to their assigned domain
3. Run all cells to generate analysis
4. Save figures and summary to results directory
5. Report key findings to NSM-25

### Phase 4: Cross-Domain Comparison (Week 4)
**Status**: ⏳ PENDING (after all agents complete)

Coordination agent should:
1. Collect all results JSON files
2. Run `python scripts/compare_domain_results.py`
3. Generate cross-domain comparison report
4. Identify best-performing domain for Phase 2

### Phase 5: Hyperparameter Tuning (Week 4-5)
**Status**: ⏳ PENDING (after comparison)

Based on Phase 4 results:
1. Select top-performing domain for intensive tuning
2. Run grid search on critical hyperparameters
3. Validate optimal configuration
4. Document findings for Phase 2

---

## Communication Protocol

### Reporting Format

Each agent should report using this template:

```markdown
## NSM-25 Progress Report: <Domain>

**Agent**: <Agent ID>
**Domain**: <Domain Name>
**Status**: <In Progress / Complete / Failed>
**Timestamp**: <ISO 8601>

### Training Status
- Epochs Completed: X/100
- Best Validation Loss: X.XXXX
- Current Metrics:
  - <Metric 1>: X.XX%
  - <Metric 2>: X.XX%
  - ...

### Issues Encountered
<List any errors, warnings, or unexpected behavior>

### Next Steps
<What the agent will do next>

### Attachments
- Log: `results/<domain>_100epoch/<experiment>.log`
- Results: `results/<domain>_100epoch/<experiment>_results.json`
- Figures: `results/<domain>_100epoch/*.png`
```

### Sync Points

Agents should synchronize at:
1. **Training Start**: All agents confirm ready to begin
2. **50% Complete**: Midpoint check-in on metrics
3. **Training End**: All agents report completion
4. **Analysis Complete**: All agents share findings

---

## Directory Structure

Each worktree should have:

```
nsm-<domain>/
├── experiments/
│   ├── train_<domain>.py              # Training script
│   └── run_full_training.sh           # Runner script ✅
├── results/
│   └── <domain>_100epoch/             # Results directory ✅
│       ├── <experiment>_results.json
│       ├── <experiment>_best_model.pt
│       ├── <experiment>.log
│       └── <domain>_*.png             # Analysis figures
├── checkpoints/
│   └── <domain>_full/
│       └── <experiment>/
│           ├── best_model.pt
│           ├── checkpoint_epoch_*.pt
│           └── results.json
└── notebooks/
    └── <domain>_analysis.ipynb        # Copied from template
```

---

## Resource Management

### Compute Resources
- **CPU**: 4 cores per agent (12 cores total)
- **Memory**: 16GB per agent (48GB total)
- **GPU**: Optional (shared if available)
- **Disk**: ~10GB per domain (30GB total)

### Time Estimates
- Training (CPU): 4-6 hours per domain
- Training (GPU): 1-2 hours per domain
- Analysis: 30 minutes per domain
- Total: ~1 day (parallel) vs ~3 days (sequential)

### Parallelization Strategy
Run all three agents **concurrently** to maximize throughput:
- Same infrastructure (NSM foundation models)
- Independent datasets (no conflicts)
- Separate git worktrees (isolated state)
- Different checkpoint directories (no overwrites)

---

## Failure Handling

### Training Failures
If training fails:
1. Check log file for error messages
2. Verify conda environment activated
3. Check dataset cache exists
4. Try reducing batch size (32 → 16)
5. Report issue to NSM-25 with full traceback

### Convergence Issues
If not converging:
1. Check gradient norms (should be >1e-6)
2. Verify cycle loss decreasing
3. Try reducing learning rate (1e-3 → 5e-4)
4. Check for NaN/Inf in losses

### Resource Issues
If out of memory:
1. Reduce batch size (32 → 16 → 8)
2. Enable gradient checkpointing
3. Reduce num_workers (4 → 2 → 0)

---

## Success Criteria

### Individual Agent Success
- ✅ Training completes 100 epochs or early stops
- ✅ Best validation loss saved
- ✅ All target metrics exceeded
- ✅ Analysis notebook executed
- ✅ Results reported to NSM-25

### Overall NSM-25 Success
- ✅ All 3 domains complete successfully
- ✅ Cross-domain comparison generated
- ✅ Best domain identified for Phase 2
- ✅ Hyperparameter recommendations documented
- ✅ Phase 2 planning initiated

---

## References

- **Linear Issue**: [NSM-25](https://linear.app/imajn/issue/NSM-25)
- **Parent Issue**: [NSM-20](https://linear.app/imajn/issue/NSM-20)
- **Domain Issues**: NSM-24 (Causal), NSM-22 (Planning), NSM-23 (KG)
- **Foundation**: Phase 1 (2-level hierarchy)

---

**Last Updated**: Auto-generated by NSM-25 setup
**Status**: Ready for parallel execution
