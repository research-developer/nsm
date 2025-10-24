# NSM Modal Training - Quick Start Guide

## Prerequisites

```bash
# Install Modal CLI
pip install modal

# Authenticate (one-time)
modal token set --token-id <id> --token-secret <secret>
```

---

## Quick Commands

### 1. Fast Validation (10 epochs, ~20 min)
```bash
modal run --detach experiments/modal_train.py::validate_3level
```

**What it does**:
- Runs all 3 domains in parallel
- Small datasets (500 samples)
- Batch size 32
- Saves to `/checkpoints/{domain}/`

**Use when**: Testing code changes, checking for class collapse

---

### 2. Full Production Training (100 epochs, ~2.5 hours)
```bash
modal run experiments/modal_train_production.py
```

**What it does**:
- Full datasets (2858 planning, 1000 causal, 2500 KG triples)
- Batch size 64 (optimized for A100-40GB)
- Early stopping (patience=20)
- Comprehensive results report

**Estimated cost**: ~$3.85 (A100-40GB @ $1.10/hr)

---

### 3. Monitor Running Jobs
```bash
# View live dashboard
open https://modal.com/apps/research-developer/main

# Or check via CLI
modal app list
modal app logs nsm-phase1.5
```

---

## Key Configurations

### Validation vs Production

| Parameter       | Validation | Production |
|-----------------|------------|------------|
| Epochs          | 10         | 100        |
| Batch size      | 32         | 64         |
| Dataset size    | 500        | Full       |
| Checkpoint freq | 5          | 10         |
| Timeout         | 1 hour     | 2 hours    |
| GPU             | A100-40GB  | A100-40GB  |

---

## Retrieve Results

### Option 1: From Modal Dashboard
1. Go to https://modal.com/apps/research-developer/main
2. Click on completed function
3. View logs and download results

### Option 2: Via Volume (programmatic)
```python
import modal

volume = modal.Volume.from_name("nsm-checkpoints")

# List checkpoints
with volume.batch_download(remote_path="/checkpoints") as entries:
    for path, f in entries:
        print(path)

# Download specific checkpoint
volume.download_file(
    remote_path="/checkpoints/planning/checkpoint_epoch_10.pt",
    local_path="./planning_checkpoint.pt"
)
```

---

## Optimization Features

### Enabled by Default
- ✅ TF32 acceleration (A100-specific, ~20% speedup)
- ✅ DataLoader prefetching (hide I/O latency)
- ✅ Pin memory (faster GPU transfers)
- ✅ Persistent workers (avoid restart overhead)
- ✅ Early stopping (patience=20)
- ✅ Checkpoint persistence (Modal Volume)

### Not Yet Implemented
- ❌ Mixed precision (AMP) - trainer needs update
- ❌ Learning rate scheduling - manual for now
- ❌ W&B logging - disabled for simplicity

---

## Troubleshooting

### Import Errors
**Fixed!** The `nsm.training.metrics` import error has been resolved.

If you see module errors:
```bash
# Rebuild image
modal deploy experiments/modal_train.py
```

### GPU Out of Memory
Reduce batch size in function call:
```python
train_planning.spawn(batch_size=32)  # Instead of 64
```

### Timeout Errors
Increase timeout (max 24 hours):
```python
@app.function(timeout=14400)  # 4 hours
```

### Checkpoint Not Found
Check volume contents:
```bash
modal volume ls nsm-checkpoints
```

---

## Cost Estimates

### A100-40GB Pricing
- **Modal**: ~$1.10/hour
- **Alternative (A100-80GB)**: ~$2.20/hour (avoid via strict `gpu="A100-40GB"`)

### Per-Domain Cost (100 epochs)
| Domain   | Time   | Cost  |
|----------|--------|-------|
| Planning | 1.3h   | $1.43 |
| Causal   | 1.0h   | $1.10 |
| KG       | 1.2h   | $1.32 |
| **Total**| **2.5h** | **$3.85** |

**Parallel execution**: Wall-clock time = max(1.3h) ≈ 1.5 hours

---

## Advanced Usage

### Custom Hyperparameters
```python
from modal_train import train_planning

train_planning.remote(
    epochs=50,
    batch_size=48,
    lr=5e-5,
    cycle_weight=0.02,
    num_problems=5000
)
```

### Single Domain Training
```bash
modal run experiments/modal_train.py::train_planning --epochs=100
modal run experiments/modal_train.py::train_causal --epochs=100
modal run experiments/modal_train.py::train_kg --epochs=100
```

### Detached Mode (Fire-and-Forget)
```bash
modal run --detach experiments/modal_train.py::train_all_domains
```

**Note**: Detached mode keeps functions alive after client disconnects. Check dashboard for completion.

---

## Expected Metrics

### Validation (10 epochs)
Based on recent runs:

| Domain   | Accuracy | Class Collapse? | Best Epoch |
|----------|----------|-----------------|------------|
| Planning | ~57%     | ✅ No           | 2          |
| Causal   | **59%**  | ✅ No           | 0          |
| KG       | 54%      | ✅ No           | 3          |

**Baseline (NSM-31)**: 43.5% accuracy

**Improvement**: +15.5% (causal domain)

### Production (100 epochs)
Expected with early stopping:

| Domain   | Target Accuracy | Confidence |
|----------|----------------|------------|
| Planning | 65-70%         | High       |
| Causal   | 70-75%         | High       |
| KG       | 60-65%         | Medium     |

**Note**: KG may need learning rate scheduling to break plateau.

---

## File Locations

### Code
- `experiments/modal_train.py` - Main training functions
- `experiments/modal_train_production.py` - Production entrypoint

### Results
- Modal Volume: `nsm-checkpoints`
- Path: `/checkpoints/{domain}/checkpoint_epoch_{N}.pt`
- Metadata: `/checkpoints/{domain}/modal_results.json`

### Logs
- Dashboard: https://modal.com/apps/research-developer/main
- CLI: `modal app logs nsm-phase1.5`

---

## Quick Debugging

### Interactive Mode
```bash
modal run -i experiments/modal_train.py::train_planning --epochs=1
```

Press Ctrl+C during training to drop into Python REPL.

### Check GPU Availability
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### Verify Imports
```python
# Inside Modal function
import sys
sys.path.insert(0, "/root/NSM")

from nsm.training import NSMTrainer, compute_classification_metrics  # ✅
```

---

## Next Steps After Training

1. **Download checkpoints**:
   ```bash
   modal volume ls nsm-checkpoints
   ```

2. **Compare to baseline** (NSM-31):
   - Causal: 43.5% → 59% (+15.5%)
   - Planning: TBD
   - KG: TBD

3. **Iterate if needed**:
   - Adjust hyperparameters
   - Try learning rate scheduling (KG)
   - Scale up dataset size

4. **Document results** in NSM-33 Linear issue

---

## Support

- **Modal Docs**: https://modal.com/docs
- **NSM CLAUDE.md**: `/Users/preston/Projects/NSM/CLAUDE.md`
- **Optimization Report**: `experiments/MODAL_OPTIMIZATION_REPORT.md`
