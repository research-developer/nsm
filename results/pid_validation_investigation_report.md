# PID Validation Experiment Investigation Report

**Date**: 2025-10-23
**App ID (Failed)**: ap-xdnHob5pqwd5v7WkUzoLLk
**App ID (Successful)**: ap-UVgGtfGeapaDyVQpYNX0NJ
**Investigator**: Claude Code

---

## Executive Summary

The PID validation experiment initially failed with **zero logs and zero tasks**, indicating it never started execution. Investigation revealed **two critical bugs** in the Modal script that prevented it from running:

1. **Missing entrypoint**: No `@app.local_entrypoint()` decorator to trigger execution
2. **Import order bug**: NSM modules imported at module-level before `sys.path` was configured

Both issues were fixed, and the experiment completed successfully, comparing PID control strategies for adaptive physics-based training.

---

## Root Cause Analysis

### Issue 1: Missing Local Entrypoint

**Problem**: The script defined `@app.function()` for `validate_pid_control()` but had no `@app.local_entrypoint()` to invoke it.

**Evidence**:
```python
# Original code (line 663):
# Removed: This is now a Modal function, not a local script
```

**Impact**: When running `modal run experiments/modal_pid_validation.py`, Modal had nothing to execute. The app registered but immediately stopped with 0 tasks.

**Fix**:
```python
@app.local_entrypoint()
def main():
    """Launch PID validation experiment."""
    print("Launching PID controller validation on Modal...")
    summary = validate_pid_control.remote()
    # Display results...
```

### Issue 2: Module-Level NSM Imports

**Problem**: NSM modules were imported at the top of the script (lines 33-34) before `sys.path` was configured:

```python
# Original code (lines 33-34 - MODULE LEVEL):
from nsm.training.adaptive_physics_trainer import AdaptivePhysicsConfig, AdaptivePhysicsTrainer
from nsm.training.pid_controller import PIDController
```

However, `sys.path.insert(0, "/root/NSM")` was inside the function (line 608), which runs **after** module-level imports.

**Error Message**:
```
File "/root/modal_pid_validation.py", line 33, in <module>
    from nsm.training.adaptive_physics_trainer import AdaptivePhysicsConfig, AdaptivePhysicsTrainer
ModuleNotFoundError: No module named 'nsm'
```

**Why This Happens**:
1. Modal copies project to `/root/NSM` via `.add_local_dir(PROJECT_ROOT, "/root/NSM", ...)`
2. `/root/NSM` is **not** on default PYTHONPATH
3. Module-level imports execute when Python loads the script, **before** any function code runs
4. The `sys.path.insert(0, "/root/NSM")` inside `validate_pid_control()` never gets a chance to execute

**Fix**:
```python
# Top of file: Remove NSM imports, add explanatory comment
# NOTE: nsm imports are moved inside the Modal function to ensure
# sys.path is set up before importing.

# Inside validate_pid_control():
@app.function(...)
def validate_pid_control():
    import sys
    sys.path.insert(0, "/root/NSM")

    # Import nsm modules AFTER sys.path is configured
    from nsm.training.adaptive_physics_trainer import AdaptivePhysicsConfig, AdaptivePhysicsTrainer
    from nsm.training.pid_controller import PIDController

    # Make available for helper functions
    global AdaptivePhysicsTrainer
    global AdaptivePhysicsConfig
```

Additionally, type hints using these classes were removed from function signatures:
```python
# Before:
def run_experiment(config: AdaptivePhysicsConfig, ...)

# After:
def run_experiment(config, ...)  # AdaptivePhysicsConfig - type hint removed
```

---

## Validation Results

After fixing both issues, the experiment ran successfully:

### Performance Summary

| Strategy | Settling Time (epochs) | Final q | Overshoot | Oscillations |
|----------|------------------------|---------|-----------|--------------|
| **Fixed Increment (Baseline)** | 10.6 ± 1.5 | 1.216 ± 0.097 | 0.318 | 13.0 |
| **PID Default** | 12.8 ± 2.3 | 1.259 ± 0.087 | 0.351 | 13.4 |
| **PID Aggressive** | **6.6 ± 0.5** | 1.151 ± 0.104 | 0.323 | **11.8** |
| **PID Smooth** | 19.8 ± 2.1 | **1.118 ± 0.064** | **0.178** | 14.6 |

### Key Findings

1. **PID Aggressive (Kp=0.2) wins on speed**: 6.6 epochs settling time (38% faster than baseline)
2. **PID Default (Kp=0.1) is SLOWER than baseline**: -20.8% (worse, not better)
3. **PID Smooth (Kp=0.05) minimizes overshoot**: Only 0.178 overshoot vs 0.318 baseline
4. **Trade-offs are clear**:
   - Aggressive: Fast settling, but more variance and slight overshoot
   - Smooth: Minimal overshoot, but 87% slower settling
   - Default: Middle ground but unexpectedly slower than fixed increment

### Interpretation

The **hypothesis that PID control provides universally better performance is NOT supported**. Results show:

- PID tuning matters significantly (3× difference between aggressive and smooth)
- Fixed increment baseline is competitive for this particular dynamics model
- Aggressive PID gains provide meaningful speed improvement with acceptable stability
- The simplified dynamics model may not fully capture real training complexity

**Recommendation**: Use **PID Aggressive (Kp=0.2, Ki=0.02, Kd=0.05)** if settling speed is critical and some overshoot is acceptable. For production, validate with real training runs rather than simulated dynamics.

---

## Files Modified

**`/Users/preston/Projects/NSM/experiments/modal_pid_validation.py`**:

### Changes Made:

1. **Removed module-level NSM imports** (lines 33-34 → comment explaining why)
2. **Added imports inside `validate_pid_control()` function** (after `sys.path` setup)
3. **Removed type hints** from `simulate_physics_trajectory()` and `run_experiment()` signatures
4. **Added `@app.local_entrypoint()`** with results display (lines 682-708)
5. **Modified `validate_pid_control()` to return summary dict** for local display

### Diff Summary:
```diff
- from nsm.training.adaptive_physics_trainer import AdaptivePhysicsConfig, AdaptivePhysicsTrainer
- from nsm.training.pid_controller import PIDController
+ # NOTE: nsm imports moved inside Modal function

- def simulate_physics_trajectory(trainer: AdaptivePhysicsTrainer, ...)
+ def simulate_physics_trajectory(trainer, ...)  # Type hint removed

+ @app.local_entrypoint()
+ def main():
+     summary = validate_pid_control.remote()
+     # Display results...
```

---

## Lessons Learned

### Modal-Specific Patterns

1. **Always have a `@app.local_entrypoint()`**: Functions decorated with `@app.function()` won't run unless invoked by an entrypoint or explicitly called with `.remote()`

2. **Import after sys.path setup**: When using `.add_local_dir()` to a non-standard path:
   ```python
   # DON'T: Module-level imports fail
   from custom_package import MyClass

   # DO: Import inside function after sys.path setup
   @app.function(...)
   def my_function():
       import sys
       sys.path.insert(0, "/custom/path")
       from custom_package import MyClass
   ```

3. **Type hints with runtime imports**: If classes are imported at runtime, either:
   - Remove type hints (`def foo(arg)` instead of `def foo(arg: MyClass)`)
   - Use string annotations (`def foo(arg: 'MyClass')`)
   - Use `from __future__ import annotations`

4. **Debugging zero-log failures**: When Modal shows 0 tasks and no logs:
   - Check for `@app.local_entrypoint()` existence
   - Verify module-level code doesn't fail (imports, etc.)
   - Try adding `print()` at module level to test if script even loads

### Comparison with Working Scripts

Reference: `/Users/preston/Projects/NSM/experiments/modal_adaptive_validation.py`

**Key difference**:
- Working script: Imports **inside** the function (line 47+), **after** `sys.path.insert(0, "/root/NSM")` (line 53)
- Broken script: Imports **at module level** (line 33), **before** any function runs

---

## Next Steps

1. **Validate with real training**: Current results use simplified dynamics model. Run PID comparison on actual NSM training to confirm benefits.

2. **Investigate PID Default slowdown**: Why does Kp=0.1 perform worse than fixed increment? Possible hypotheses:
   - Integral term accumulation causing overcorrection
   - Derivative term dampening too aggressively
   - Simplified model doesn't match real training response

3. **Plot review**: Results generated plots in `/tmp/pid_validation/` (lost when container stopped). To preserve:
   - Use Modal Volume for persistent storage
   - Download plots via `.remote()` return value
   - Save to cloud storage (S3, etc.)

4. **Production integration**: If PID Aggressive proves effective in real runs, integrate into main adaptive training pipeline with:
   - Configurable gain scheduling
   - Runtime gain tuning based on observed dynamics
   - Safety limits on adjustment magnitude

---

## Appendix: How to Re-run

```bash
# Run the corrected experiment
modal run experiments/modal_pid_validation.py

# Results will be displayed in terminal
# Plots are generated but not returned (currently in /tmp)
```

To preserve plots, modify script to:
1. Return plot data as base64 strings
2. Save to Modal Volume
3. Upload to cloud storage

---

**Investigation Complete**
**Status**: ✅ Both bugs fixed, experiment completed successfully
**Runtime**: ~60 seconds total (simulation-based, no GPU training required)
