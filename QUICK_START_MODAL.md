# Quick Start: Modal GPU Training (NSM)

**TL;DR**: All fixes applied. Ready to run validation on Modal GPU.

---

## What Was Fixed

1. **Tensor shape mismatch** in KG domain metrics → ✅ Fixed and tested
2. **Image import paths** (`/root/nsm` → `/root`) → ✅ Fixed
3. **GPU sizing** (bare `A100` → `A100-40GB`) → ✅ Fixed
4. **Error handling** (sequential → parallel with recovery) → ✅ Fixed

---

## Run Validation Now

```bash
# Quick 10-epoch validation (all three domains in parallel)
modal run experiments/modal_train.py::validate_3level

# Expected output:
# 🧪 Running 3-level validation (10 epochs)...
# ⏳ Waiting for validation jobs...
#
# PLANNING: ✅ SUCCESS
#   Accuracy: XX.XX%
#   Cycle loss: X.XXXX
#   ✅ No collapse (C0: XX.XX%, C1: XX.XX%)
#
# CAUSAL: ✅ SUCCESS
#   ...
#
# KG: ✅ SUCCESS
#   ...
```

**Time**: ~10-15 minutes per domain (parallel execution)

---

## Run Full Training

```bash
# Train all domains for 100 epochs
modal run experiments/modal_train.py::train_all_domains
```

**Time**: ~60-90 minutes per domain

---

## What to Check

### During Training

Monitor Modal dashboard:
- GPU utilization (target >80%)
- Training logs (epoch progress bars)
- No "CUDA out of memory" errors

### After Completion

Check results:
```bash
# Download checkpoints from Modal volume
modal volume get nsm-checkpoints /checkpoints ./local_checkpoints

# Review results
cat local_checkpoints/planning/modal_results.json
cat local_checkpoints/causal/modal_results.json
cat local_checkpoints/kg/modal_results.json
```

Expected metrics:
- `final_metrics.accuracy`: >50% (random baseline is 50% for binary)
- `final_metrics.accuracy_class_0`: >0% (detects class collapse)
- `final_metrics.accuracy_class_1`: >0% (detects class collapse)
- `final_metrics.cycle_loss`: <0.5 (target <0.2)

---

## Troubleshooting

### If KG Still Fails

Check error message. If it's still tensor shape related:
```bash
# Test metrics locally first
python tests/test_metrics_fix.py

# Should output:
# 🎉 All tests passed! Metrics fix is working correctly.
```

### If Training Hangs

- Check Modal dashboard for container logs
- Possible causes:
  - Data loading bottleneck (reduce `num_workers`)
  - GPU OOM (reduce `batch_size`)
  - Timeout (increase `timeout` parameter)

### If One Domain Fails

**This is OK!** The new error handling continues other domains.

Check which domain failed:
- Planning: Likely procedural reasoning issue
- Causal: Likely counterfactual complexity
- KG: Likely relational diversity challenge

Review error in output:
```
KG: ❌ FAILED
  Error: <error message here>
```

---

## Cost Estimates

**Validation (10 epochs, 3 domains in parallel)**:
- Time: ~15 minutes
- Cost: ~$0.50 (A100-40GB @ $2/hr)

**Full Training (100 epochs, 3 domains in parallel)**:
- Time: ~90 minutes
- Cost: ~$3.00

**Savings from strict GPU sizing**: ~50% (vs auto-upgrade to 80GB)

---

## Next Steps After Validation

1. **If all domains succeed**:
   - Run full training (100 epochs)
   - Compare domain performance (see NSM-10-CROSS-DOMAIN-COMPARISON.md)

2. **If some domains fail**:
   - Review error messages
   - Adjust hyperparameters per domain
   - Consider domain-specific model configs

3. **Production deployment**:
   - Enable memory snapshots (3-5x faster cold starts)
   - Add checkpoint resumption (saves retry waste)
   - Tune DataLoader workers (improve GPU utilization)

---

## Key Files

- **Modal script**: `/Users/preston/Projects/NSM/experiments/modal_train.py`
- **Metrics fix**: `/Users/preston/Projects/NSM/nsm/training/trainer.py` (line 557)
- **Best practices**: `/Users/preston/Projects/NSM/MODAL_BEST_PRACTICES.md`
- **Test suite**: `/Users/preston/Projects/NSM/tests/test_metrics_fix.py`

---

## Getting Help

**Modal Issues**:
- Check logs: `modal container logs <container-id>`
- Interactive debug: `modal run -i experiments/modal_train.py::validate_3level`
- Exec into container: `modal container exec <container-id> bash`

**NSM Issues**:
- Review CLAUDE.md for architecture
- Check NSM-20 Linear issue for implementation details
- Review NSM-10-CROSS-DOMAIN-COMPARISON.md for domain insights

---

**Ready?** Run this now:

```bash
modal run experiments/modal_train.py::validate_3level
```

🚀 Good luck!

---

**Generated**: 2025-10-21
**Status**: ✅ Ready for GPU validation

🤖 Generated with Claude Code
