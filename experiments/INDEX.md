# NSM Experiments - Complete Index

Master index for all experimental training infrastructure.

## Quick Navigation

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[README.md](README.md)** | Overview & integration | Start here for big picture |
| **[MODAL_NOTEBOOK_GUIDE.md](MODAL_NOTEBOOK_GUIDE.md)** | Complete user guide | Primary reference for using notebook |
| **[NOTEBOOK_QUICK_REFERENCE.md](NOTEBOOK_QUICK_REFERENCE.md)** | One-page cheat sheet | Quick lookup while working |
| **[NOTEBOOK_WORKFLOW.md](NOTEBOOK_WORKFLOW.md)** | Visual workflow diagrams | Understand data/execution flow |
| **[NOTEBOOK_TEST_CHECKLIST.md](NOTEBOOK_TEST_CHECKLIST.md)** | 30-test validation suite | Before production deployment |
| **[NOTEBOOK_DEPLOYMENT.md](NOTEBOOK_DEPLOYMENT.md)** | Deployment summary | Final launch preparation |

## File Organization

```
experiments/
├── Core Implementation
│   ├── nsm_training_notebook.py          Modal app (launches Jupyter)
│   └── NSM_Training_Dashboard.ipynb      Interactive training notebook
│
├── Primary Documentation
│   ├── README.md                         Overview & integration
│   ├── MODAL_NOTEBOOK_GUIDE.md          Complete user guide ⭐
│   └── NOTEBOOK_QUICK_REFERENCE.md       One-page cheat sheet
│
├── Supplementary Documentation
│   ├── NOTEBOOK_WORKFLOW.md              Visual workflows & diagrams
│   ├── NOTEBOOK_TEST_CHECKLIST.md        Testing & validation
│   ├── NOTEBOOK_DEPLOYMENT.md            Deployment summary
│   └── INDEX.md                          This file
│
├── Production Scripts (Previous Work)
│   ├── modal_train_production.py         Batch training script
│   ├── modal_train.py                    Original Modal script
│   ├── MODAL_QUICKSTART.md               Production guide
│   ├── MODAL_OPTIMIZATION_REPORT.md      Performance tuning
│   └── VALIDATION_RESULTS_SUMMARY.md     Phase 1.5 results
│
└── Generated Outputs
    └── training_log.jsonl                Training logs
```

## Document Hierarchy

```
START HERE
    ↓
┌─────────────────────────────────────────┐
│ README.md                               │  Overview
│ "What is this project?"                 │  5 min read
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ MODAL_NOTEBOOK_GUIDE.md                 │  Primary Guide
│ "How do I use the notebook?"            │  30 min read
│                                         │  ⭐ MAIN REFERENCE
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ NOTEBOOK_QUICK_REFERENCE.md             │  Quick Lookup
│ "How do I do X quickly?"                │  2 min scan
└────────────┬────────────────────────────┘
             │
             ├─────────────────────────────┐
             │                             │
             ↓                             ↓
┌──────────────────────────┐  ┌──────────────────────────┐
│ NOTEBOOK_WORKFLOW.md     │  │ NOTEBOOK_TEST_CHECKLIST.md│
│ "How does it work?"      │  │ "Is it ready?"            │
│ Visual diagrams          │  │ Validation tests          │
└──────────────────────────┘  └──────────────────────────┘
             │                             │
             └─────────────┬───────────────┘
                           ↓
             ┌──────────────────────────┐
             │ NOTEBOOK_DEPLOYMENT.md   │  Launch Summary
             │ "Final checklist"        │  Pre-launch review
             └──────────────────────────┘
```

## User Personas & Recommended Reading

### First-Time User
**Goal**: Get started quickly
**Path**:
1. README.md (5 min) - Understand what this is
2. MODAL_NOTEBOOK_GUIDE.md → Quick Start section (10 min)
3. Launch notebook and run Cell 1
4. Keep NOTEBOOK_QUICK_REFERENCE.md open while working

### Developer
**Goal**: Modify and extend notebook
**Path**:
1. README.md (5 min) - Architecture overview
2. NOTEBOOK_WORKFLOW.md (15 min) - Understand data flow
3. NSM_Training_Dashboard.ipynb - Read through all cells
4. MODAL_NOTEBOOK_GUIDE.md → Advanced Usage (20 min)

### Validator/Tester
**Goal**: Validate notebook is production-ready
**Path**:
1. README.md (5 min) - What to expect
2. NOTEBOOK_TEST_CHECKLIST.md (full) - Run all 30 tests
3. NOTEBOOK_DEPLOYMENT.md - Review deployment criteria
4. Report findings

### Operations/MLOps
**Goal**: Deploy and maintain in production
**Path**:
1. README.md (5 min) - System overview
2. NOTEBOOK_DEPLOYMENT.md (15 min) - Deployment guide
3. MODAL_NOTEBOOK_GUIDE.md → Troubleshooting (15 min)
4. Set up monitoring and cost tracking

### Researcher/Experimenter
**Goal**: Run experiments and analyze results
**Path**:
1. MODAL_NOTEBOOK_GUIDE.md (full) - Learn all features
2. NSM_Training_Dashboard.ipynb - Explore all cells
3. NOTEBOOK_QUICK_REFERENCE.md - Keep handy
4. VALIDATION_RESULTS_SUMMARY.md - Compare to baseline

## Quick Command Reference

### Launch Notebook
```bash
modal run experiments/nsm_training_notebook.py
```

### List Checkpoints
```bash
modal volume ls nsm-checkpoints
```

### Download Results
```bash
modal volume get nsm-checkpoints causal ./results/causal
```

### Check Logs
```bash
modal app logs nsm-notebook
```

### Debug Container
```bash
modal container list
modal container exec <id> bash
```

## Common Tasks → Document Mapping

| Task | Primary Document | Section |
|------|------------------|---------|
| First launch | MODAL_NOTEBOOK_GUIDE.md | Quick Start |
| Configure training | MODAL_NOTEBOOK_GUIDE.md | Training Configuration |
| Fix GPU not available | NOTEBOOK_QUICK_REFERENCE.md | Troubleshooting |
| Change domains | MODAL_NOTEBOOK_GUIDE.md | Change Domain Mid-Session |
| Download checkpoints | NOTEBOOK_QUICK_REFERENCE.md | Checkpoint Management |
| Compare domains | MODAL_NOTEBOOK_GUIDE.md | Cross-Domain Comparison |
| Understand workflow | NOTEBOOK_WORKFLOW.md | (All sections) |
| Run tests | NOTEBOOK_TEST_CHECKLIST.md | (All tests) |
| Deploy to prod | NOTEBOOK_DEPLOYMENT.md | Launch Instructions |
| Optimize performance | MODAL_NOTEBOOK_GUIDE.md | Performance Optimization |
| Debug errors | MODAL_NOTEBOOK_GUIDE.md | Troubleshooting |
| Add features | README.md | Contributing |

## External Resources

### Modal Platform
- **Docs**: https://modal.com/docs
- **CLI Reference**: https://modal.com/docs/reference/cli
- **GPU Guide**: https://modal.com/docs/guide/gpu
- **Volumes**: https://modal.com/docs/guide/volumes

### PyTorch Ecosystem
- **PyTorch**: https://pytorch.org/docs/stable/index.html
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io
- **PyG Examples**: https://github.com/pyg-team/pytorch_geometric/tree/master/examples

### Jupyter
- **JupyterLab**: https://jupyterlab.readthedocs.io
- **Keyboard Shortcuts**: https://jupyterlab.readthedocs.io/en/stable/user/interface.html

### NSM Project
- **Main README**: /Users/preston/Projects/NSM/README.md
- **Architecture Guide**: /Users/preston/Projects/NSM/CLAUDE.md
- **Phase 1.5 Results**: /Users/preston/Projects/NSM/NSM-10-CROSS-DOMAIN-COMPARISON.md

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Oct 21, 2025 | Initial release |
| | | - Interactive notebook with 11 cells |
| | | - Full documentation suite |
| | | - 30-test validation checklist |
| | | - Cross-domain comparison |

## Support & Troubleshooting

### Issue Resolution Path

```
Problem Encountered
        │
        ↓
1. Check NOTEBOOK_QUICK_REFERENCE.md (Troubleshooting section)
        │
        ↓ Not solved?
        │
2. Review MODAL_NOTEBOOK_GUIDE.md (Troubleshooting & Tips)
        │
        ↓ Not solved?
        │
3. Check error traceback against known issues
        │
        ↓ Not solved?
        │
4. Enable debug logging:
   MODAL_LOGLEVEL=DEBUG modal run experiments/nsm_training_notebook.py
        │
        ↓ Not solved?
        │
5. Exec into container for investigation:
   modal container exec <id> bash
        │
        ↓ Not solved?
        │
6. Consult Modal docs or support
```

### Known Issues & Workarounds

See MODAL_NOTEBOOK_GUIDE.md → Troubleshooting section for:
- GPU not available → Restart kernel
- Out of memory → Reduce batch size
- Import errors → Check sys.path
- Training hangs → Reduce num_workers
- Volume issues → Manual commit

## Statistics

**Total Files**: 8 core files + 5 legacy/output files
**Total Lines**: ~4,800 lines of code and documentation
**Documentation**: ~50KB across 6 primary documents
**Test Coverage**: 30 tests across all functionality

**Notebook Cells**: 11
- Setup & config: 2
- Data & model: 2
- Training: 1
- Analysis: 4
- Utilities: 2

## License & Attribution

Part of the Neural Symbolic Model (NSM) project.
See main project LICENSE for details.

**Created by**: Claude Code (Anthropic)
**Date**: October 21, 2025
**Purpose**: Phase 1.5 interactive training infrastructure

---

## Quick Start (30 seconds)

```bash
# 1. Launch
modal run experiments/nsm_training_notebook.py

# 2. Open URL in browser

# 3. Load NSM_Training_Dashboard.ipynb

# 4. Run Cell 1-5

# 5. Watch training!
```

**Need help?** Open MODAL_NOTEBOOK_GUIDE.md

**Quick reference?** Open NOTEBOOK_QUICK_REFERENCE.md

**Visual guide?** Open NOTEBOOK_WORKFLOW.md

---

**Last Updated**: October 21, 2025
