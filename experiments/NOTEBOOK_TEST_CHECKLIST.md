# NSM Training Notebook - Testing Checklist

Complete validation checklist before production use.

## Pre-Launch Checks

- [ ] Modal CLI installed: `modal --version`
- [ ] Modal authenticated: `modal token list`
- [ ] NSM codebase up to date: `git pull origin main`
- [ ] All required files present:
  - [ ] `experiments/nsm_training_notebook.py`
  - [ ] `experiments/NSM_Training_Dashboard.ipynb`
  - [ ] `nsm/data/planning_dataset.py`
  - [ ] `nsm/data/causal_dataset.py`
  - [ ] `nsm/data/kg_dataset.py`
  - [ ] `nsm/models/hierarchical_model.py`

## Launch Tests

### Test 1: Basic Launch
```bash
modal run experiments/nsm_training_notebook.py
```

**Expected output:**
- [ ] No syntax errors
- [ ] Container provisions successfully
- [ ] GPU allocated (A100-40GB)
- [ ] Jupyter URL displayed
- [ ] URL accessible in browser

**Checklist:**
- [ ] Launch completes in <2 minutes
- [ ] JupyterLab UI loads
- [ ] No error messages in terminal
- [ ] Green "Connected" indicator in JupyterLab

### Test 2: Environment Verification

Open `NSM_Training_Dashboard.ipynb` and run **Cell 1**.

**Expected output:**
- [ ] Python 3.10
- [ ] PyTorch 2.1.0
- [ ] CUDA Available: True
- [ ] GPU: NVIDIA A100-SXM4-40GB
- [ ] VRAM Total: 40.0GB
- [ ] CUDA Version: 11.8
- [ ] TF32 enabled
- [ ] Checkpoint dir exists

**Checklist:**
- [ ] Cell executes without errors
- [ ] GPU properly detected
- [ ] All imports successful
- [ ] Checkpoint volume mounted

## Functional Tests

### Test 3: Configuration (Cell 2)

**Expected output:**
- [ ] Configuration displays correctly
- [ ] All parameters set to defaults
- [ ] Domain config loaded

**Checklist:**
- [ ] No errors
- [ ] Values match expectations
- [ ] Easy to modify

### Test 4: Dataset Loading (Cell 3)

For each domain:
- [ ] Causal dataset loads
- [ ] Planning dataset loads
- [ ] KG dataset loads

**Expected output:**
- [ ] Dataset size: 2000 graphs
- [ ] Train/val/test split: 70/15/15%
- [ ] Sample graph has expected structure
- [ ] Node levels present: [0, 1, 2]
- [ ] DataLoaders created successfully

**Checklist:**
- [ ] No import errors
- [ ] Dataset generation completes
- [ ] Sample graph looks reasonable
- [ ] Splits add up correctly

### Test 5: Model Initialization (Cell 4)

**Expected output:**
- [ ] Model initializes on CUDA
- [ ] Total parameters: ~200K-500K
- [ ] Layer breakdown shows all components
- [ ] Optimizer initialized
- [ ] Scheduler initialized

**Checklist:**
- [ ] No CUDA errors
- [ ] Parameter count reasonable
- [ ] All layers present
- [ ] Model on correct device

### Test 6: Quick Training Test (Cell 5)

**Modify Cell 2 first:**
```python
EPOCHS = 5  # Quick test
EVAL_EVERY = 2
```

Then run Cell 5.

**Expected output:**
- [ ] Training starts immediately
- [ ] Progress bars display
- [ ] GPU memory usage stable
- [ ] Loss decreases
- [ ] Plots update at epoch 2, 4
- [ ] No OOM errors
- [ ] Checkpoints saved

**Checklist:**
- [ ] Training loop runs
- [ ] Metrics update
- [ ] Plots render correctly
- [ ] GPU memory <35GB
- [ ] No hanging or freezes
- [ ] Graceful completion

### Test 7: Checkpoint Management (Cell 6)

**Expected output:**
- [ ] Checkpoint list displays
- [ ] DataFrame shows metrics
- [ ] File sizes reasonable (5-20MB)

**Checklist:**
- [ ] Best model saved
- [ ] Periodic checkpoints saved
- [ ] All files accessible

### Test 8: Load Checkpoint (Cell 7)

**Expected output:**
- [ ] Checkpoint loads successfully
- [ ] Metrics display correctly
- [ ] Model weights loaded
- [ ] No errors

**Checklist:**
- [ ] Load completes quickly
- [ ] Metrics match training
- [ ] Model ready for inference

### Test 9: Test Evaluation (Cell 8)

**Expected output:**
- [ ] Validation runs successfully
- [ ] Test metrics displayed
- [ ] Confusion matrix shows
- [ ] Confidence distribution plotted
- [ ] Classification report printed

**Checklist:**
- [ ] No errors during evaluation
- [ ] Metrics reasonable (>50% acc)
- [ ] Plots render correctly
- [ ] No class collapse evident

### Test 10: Cross-Domain Comparison (Cell 9)

**Prerequisites:** Train all 3 domains (can skip for initial test)

**Expected output:**
- [ ] Results table displays
- [ ] Bar charts show
- [ ] Metrics comparable

**Checklist:**
- [ ] Handles missing domains gracefully
- [ ] Charts render when data available

### Test 11: Save & Export (Cell 10)

**Expected output:**
- [ ] History saved to JSON
- [ ] Results saved to JSON
- [ ] Volume commit successful
- [ ] Download command displayed

**Checklist:**
- [ ] Files written to /checkpoints
- [ ] JSON files valid
- [ ] Volume persists after commit

### Test 12: GPU Diagnostics (Cell 11)

**Expected output:**
- [ ] GPU info displays
- [ ] Memory stats accurate
- [ ] Optimizations enabled
- [ ] Cache clears successfully

**Checklist:**
- [ ] All metrics present
- [ ] Memory calculations correct
- [ ] Cache clear works

## Stress Tests

### Test 13: Full Training Run

**Modify Cell 2:**
```python
EPOCHS = 100
BATCH_SIZE = 64
```

**Expected:**
- [ ] Runs for ~30-60 minutes
- [ ] Completes without crashes
- [ ] Final accuracy >50%
- [ ] No memory leaks
- [ ] Checkpoints saved regularly

### Test 14: Interrupt & Resume

1. Start training (Cell 5)
2. After 10 epochs, interrupt (Kernel → Interrupt)
3. Load checkpoint (Cell 7)
4. Resume training

**Checklist:**
- [ ] Interrupt works gracefully
- [ ] Checkpoint preserved
- [ ] Resume possible
- [ ] No data loss

### Test 15: OOM Recovery

**Modify Cell 2:**
```python
BATCH_SIZE = 256  # Intentionally too large
```

**Expected:**
- [ ] OOM error caught
- [ ] Error message clear
- [ ] Kernel recoverable (restart)
- [ ] Can reduce batch size and retry

### Test 16: All Domains Sequential

1. Train causal (EPOCHS=10)
2. Change `DOMAIN="planning"` in Cell 2
3. Rerun Cells 2-5
4. Change `DOMAIN="kg"` in Cell 2
5. Rerun Cells 2-5
6. Run Cell 9 for comparison

**Checklist:**
- [ ] All domains train successfully
- [ ] Separate checkpoints saved
- [ ] No interference between domains
- [ ] Comparison chart shows all 3

## Performance Tests

### Test 17: Speed Benchmarks

**Expected times (A100-40GB, batch_size=64):**
- [ ] Dataset load: <30 seconds
- [ ] Model init: <5 seconds
- [ ] Single epoch: <60 seconds (for 2000 graphs)
- [ ] Validation: <30 seconds
- [ ] 10 epochs: <15 minutes
- [ ] 100 epochs: <90 minutes

### Test 18: Memory Efficiency

**Expected usage:**
- [ ] Peak GPU memory: <35GB (with batch_size=64)
- [ ] Peak GPU memory: <20GB (with batch_size=32)
- [ ] No memory leaks over 100 epochs
- [ ] Stable memory after initial ramp

### Test 19: Scaling Test

Test different batch sizes:
- [ ] batch_size=16: Works, ~10GB VRAM
- [ ] batch_size=32: Works, ~18GB VRAM
- [ ] batch_size=64: Works, ~32GB VRAM
- [ ] batch_size=128: May OOM or work, ~38GB VRAM

## Integration Tests

### Test 20: Volume Persistence

1. Train and save checkpoint
2. Shutdown notebook (Kernel → Shutdown)
3. Exit browser
4. Ctrl+C in terminal
5. Relaunch: `modal run experiments/nsm_training_notebook.py`
6. Check checkpoints in Cell 6

**Checklist:**
- [ ] Checkpoints still present
- [ ] Can load previous checkpoint
- [ ] History preserved
- [ ] Volume commit worked

### Test 21: Download Checkpoints

From local machine:
```bash
modal volume ls nsm-checkpoints
modal volume get nsm-checkpoints causal/best_model.pt ./test_download.pt
```

**Checklist:**
- [ ] Volume listing works
- [ ] File downloads successfully
- [ ] Downloaded file loads locally
- [ ] Metrics match

### Test 22: Concurrent Sessions

Open 2 browser tabs to the same notebook URL.

**Checklist:**
- [ ] Both tabs work
- [ ] Changes sync appropriately
- [ ] No corruption
- [ ] Kernel shared correctly

## Failure Recovery Tests

### Test 23: Network Interruption

1. Start training
2. Close browser tab mid-training
3. Reopen URL
4. Check training status

**Checklist:**
- [ ] Training continues in background
- [ ] Can reconnect to session
- [ ] Progress preserved

### Test 24: Manual Kill & Restart

```bash
# While training, press Ctrl+C in terminal
# Then relaunch
modal run experiments/nsm_training_notebook.py
```

**Checklist:**
- [ ] Graceful shutdown
- [ ] Checkpoints saved
- [ ] Can resume in new session

### Test 25: Volume Commit Failure

Simulate by not calling `volume.commit()`.

**Checklist:**
- [ ] Background commits still happen
- [ ] Major checkpoints persisted
- [ ] Warning/error if manual commit needed

## Documentation Tests

### Test 26: Guide Accuracy

Follow `MODAL_NOTEBOOK_GUIDE.md` step-by-step.

**Checklist:**
- [ ] All commands work as written
- [ ] No outdated information
- [ ] Examples execute correctly
- [ ] Troubleshooting tips accurate

### Test 27: Quick Reference

Try all operations in `NOTEBOOK_QUICK_REFERENCE.md`.

**Checklist:**
- [ ] All snippets valid
- [ ] Shortcuts work
- [ ] Paths correct
- [ ] Commands succeed

## Production Readiness

### Test 28: Error Handling

Introduce various errors:
- [ ] Missing dataset file (handled gracefully)
- [ ] Corrupt checkpoint (error message clear)
- [ ] Invalid configuration (validation catches)
- [ ] GPU OOM (recoverable)

### Test 29: Logging & Debugging

**Checklist:**
- [ ] Progress bars display correctly
- [ ] Print statements visible
- [ ] Plots update in real-time
- [ ] Error tracebacks useful

### Test 30: User Experience

**Checklist:**
- [ ] Intuitive cell order
- [ ] Clear documentation in markdown cells
- [ ] Helpful error messages
- [ ] Smooth workflow
- [ ] No unexpected behaviors

## Sign-Off

**Tested by:** _________________

**Date:** _________________

**Modal Version:** `modal --version` → _________________

**Results Summary:**

- [ ] All basic tests pass (1-12)
- [ ] All stress tests pass (13-16)
- [ ] All performance tests meet expectations (17-19)
- [ ] All integration tests pass (20-22)
- [ ] All failure recovery tests pass (23-25)
- [ ] Documentation accurate (26-27)
- [ ] Production ready (28-30)

**Issues Found:**

1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

**Recommendations:**

1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

**Approval for Production Use:**

- [ ] Approved
- [ ] Conditional (see issues)
- [ ] Not approved (major issues)

---

**Next Steps:**

1. Address any issues found
2. Retest failed cases
3. Document any workarounds
4. Update guides with findings
5. Deploy for user testing
