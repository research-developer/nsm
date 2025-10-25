# Modal Deployment Guide - CGT Operators Validation

**Project**: NSM-34 Conway Combinatorial Game Theory Operators
**Status**: Ready for cloud deployment
**GPU**: A100-40GB recommended

---

## Quick Start

### 1. Install Modal

```bash
pip install modal
modal setup  # Follow authentication prompts
```

### 2. Run Validation Experiments

```bash
# Validate all operators in parallel (~30 min)
modal run experiments/modal_cgt_validation.py::validate_all_operators

# Or run individual operators
modal run experiments/modal_cgt_validation.py::validate_temperature  # ~15 min
modal run experiments/modal_cgt_validation.py::validate_cooling      # ~15 min

# View results
modal run experiments/modal_cgt_validation.py::show_results
```

---

## What Gets Validated

### Operator 1: Conway Temperature

**Tests:**
- ✅ Temperature computation on 50 test batches
- ✅ Statistical analysis (mean, std, range)
- ✅ Comparison to physics baseline (q_neural)
- ✅ Stability prediction agreement

**Pre-Registered Predictions:**
- **P1.2**: Temperature < 0.2 predicts collapse (threshold check)
- **P1.1**: Temperature decreases during collapse (awaits training data)

**Expected Output:**
```json
{
  "operator": "temperature",
  "statistics": {
    "mean_temperature": 0.45,
    "std_temperature": 0.12,
    "min_temperature": 0.25,
    "max_temperature": 0.68
  },
  "baseline_comparison": {
    "q_neural": 1.23,
    "q_neural_stable": true,
    "cgt_stable": true,
    "agreement": true
  }
}
```

### Operator 2: Cooling Monitor

**Tests:**
- ✅ Cooling rate computation over 20 training epochs
- ✅ Temperature trajectory (α, β → 0.5)
- ✅ Collapse time prediction
- ✅ Rapid cooling event detection (rate < -0.05)

**Pre-Registered Predictions:**
- **P2.1**: Rapid cooling (< -0.05) predicts collapse within 2 epochs

**Expected Output:**
```json
{
  "operator": "cooling",
  "statistics": {
    "initial_temperature": 0.80,
    "final_temperature": 0.05,
    "mean_cooling_rate": -0.0375,
    "rapid_cooling_events": 3
  },
  "predictions_tested": {
    "P2.1": "rapid_cooling_detected: 3 events"
  }
}
```

---

## Modal Best Practices Implemented

### ✅ 1. Correct Import Paths
```python
# Uses /root as remote path (not /root/nsm)
.add_local_dir(PROJECT_ROOT / "nsm", remote_path="/root")

# Modal adds /root to PYTHONPATH → import nsm.training.cgt_metrics works
```

### ✅ 2. Strict GPU Sizing
```python
gpu="A100-40GB"  # Explicit 40GB (no surprise 80GB upgrades = 2x cost)
```

### ✅ 3. Memory Snapshots
```python
enable_memory_snapshot=True  # 3-5x faster cold starts
```

### ✅ 4. Parallel Job Execution
```python
# Launch jobs in parallel
jobs = {
    'temperature': validator.validate_temperature_operator.spawn(...),
    'cooling': validator.validate_cooling_operator.spawn(...)
}

# Handle errors independently
for name, job in jobs.items():
    try:
        result = job.get(timeout=1800)
        results[name] = {'status': 'success', 'data': result}
    except Exception as e:
        results[name] = {'status': 'failed', 'error': str(e)}
        # Continue instead of crashing
```

### ✅ 5. Volume Commits
```python
@modal.exit()
def cleanup(self):
    """Always runs on exit (success, failure, OR preemption)."""
    print("💾 Final volume commit...")
    volume.commit()
```

### ✅ 6. Optimized DataLoaders
```python
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,           # Match reserved CPUs
    pin_memory=True,         # Faster GPU transfer
    persistent_workers=True, # Reuse workers
    prefetch_factor=2        # Prefetch batches
)
```

### ✅ 7. Retries with Backoff
```python
retries=modal.Retries(
    max_retries=2,
    backoff_coefficient=2.0,
    initial_delay=60.0
)
```

---

## Cost Estimation

### Per Run Costs (A100-40GB)

| Experiment | Duration | Cost | Notes |
|------------|----------|------|-------|
| **Temperature validation** | ~15 min | ~$0.20 | 50 batches, 20 samples each |
| **Cooling validation** | ~15 min | ~$0.20 | 20 epochs mini-training |
| **Both in parallel** | ~20 min | ~$0.40 | Parallel = max(15, 15) + overhead |

**Optimization tips:**
- Use `enable_memory_snapshot=True` (free 3-5x startup speedup)
- Strict `gpu="A100-40GB"` (avoid 80GB surprise = -50% cost)
- Results cached in volume (re-run = instant, no GPU)

---

## Development Workflow

### 1. Local Testing First

```bash
# Run tests locally before Modal deployment
pytest tests/test_cgt_temperature.py -v

# Verify imports work
python -c "from nsm.training.cgt_metrics import temperature_conway; print('✅ Import works')"
```

### 2. Deploy to Modal

```bash
# Interactive mode for debugging
modal run -i experiments/modal_cgt_validation.py::validate_temperature

# Production mode
modal run experiments/modal_cgt_validation.py::validate_all_operators
```

### 3. Monitor Progress

```bash
# List running containers
modal container list

# Attach to running container
modal container exec <container-id> bash

# View logs in real-time
modal app logs nsm-cgt-validation
```

### 4. Retrieve Results

```bash
# View results via Modal function
modal run experiments/modal_cgt_validation.py::show_results

# Or download volume locally
modal volume get nsm-cgt-checkpoints /results ./local_results/
```

---

## Customization

### Adjust Validation Parameters

```python
# More thorough temperature validation
modal run experiments/modal_cgt_validation.py::validate_temperature \
    --num-samples 100 \
    --num-test-batches 200

# Longer cooling validation
modal run experiments/modal_cgt_validation.py::validate_cooling \
    --num-epochs 50
```

### Change GPU Type

```python
# Edit modal_cgt_validation.py
gpu="L40S"  # Cheaper for development
# or
gpu="A100-80GB"  # If you need more VRAM
```

### Add New Operators

When implementing Operators 3, 4, 5, add to the same file:

```python
@modal.method()
def validate_confusion_operator(self, ...):
    """Validate Operator 3: Confusion Intervals"""
    ...

# Then update validate_all_operators()
jobs['confusion'] = validator.validate_confusion_operator.spawn(...)
```

---

## Troubleshooting

### Issue: Import Error

```bash
ModuleNotFoundError: No module named 'nsm'
```

**Fix**: Verify remote path is `/root` (not `/root/nsm`)

```python
# CORRECT
.add_local_dir(PROJECT_ROOT / "nsm", remote_path="/root")

# WRONG
.add_local_dir(PROJECT_ROOT / "nsm", remote_path="/root/nsm")
```

### Issue: CUDA Out of Memory

```bash
RuntimeError: CUDA out of memory
```

**Fix**: Reduce batch size or use A100-80GB

```python
# In validate_*_operator methods
batch_size=16  # Down from 32
```

### Issue: Timeout

```bash
TimeoutError: Function exceeded timeout of 3600 seconds
```

**Fix**: Increase timeout or reduce work

```python
@app.cls(
    timeout=7200,  # 2 hours instead of 1
    ...
)
```

### Issue: Volume Not Persisting

```bash
# Results disappear after run
```

**Fix**: Ensure explicit commits

```python
# After writing results
volume.commit()

# And in @modal.exit() hook
```

---

## Next Steps

### After Validation

1. **Analyze Results**
   ```bash
   modal run experiments/modal_cgt_validation.py::show_results
   ```

2. **Compare to Baseline**
   - Check if temperature predictions align with q_neural
   - Verify cooling rates correlate with collapse events

3. **Iterate**
   - Adjust thresholds (0.2 for temperature, -0.05 for cooling)
   - Test on different architectures
   - Run full N=24,000 validation

### Implement Remaining Operators

Use this as a template for:
- **Operator 3**: Confusion intervals (MEDIUM PRIORITY)
- **Operator 4**: Game addition (MEDIUM PRIORITY)
- **Operator 5**: Surreal classification (LOW PRIORITY)

### Integration

Once all 5 operators validated:
- Build Composite Conway Score (CCS)
- Run comparative experiments (Physics vs CGT vs Combined)
- Target: >90% collapse prediction accuracy

---

## References

- **Modal Docs**: https://modal.com/docs
- **Modal Best Practices**: [MODAL_BEST_PRACTICES.md](MODAL_BEST_PRACTICES.md)
- **CGT Operators Pre-Reg**: [notes/NSM-34-CGT-OPERATORS-PREREG.md](notes/NSM-34-CGT-OPERATORS-PREREG.md)
- **Implementation Guide**: [notes/NSM-34-IMPLEMENTATION-GUIDE.md](notes/NSM-34-IMPLEMENTATION-GUIDE.md)

---

**Status**: Production-ready
**Last Updated**: 2025-10-23
**Estimated Cost**: ~$0.40 per full validation run

🤖 Generated with [Claude Code](https://claude.com/claude-code)
