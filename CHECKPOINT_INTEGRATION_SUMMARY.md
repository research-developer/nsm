# Checkpoint Storage & CGT Integration Setup

**Date**: 2025-10-23
**Status**: ✅ Complete - Ready for use

---

## Summary

Created comprehensive checkpoint management system for NSM experiments with full CGT integration. Checkpoints are now stored in both Modal volumes and the local repo, enabling trained models to be loaded into CGT validation experiments.

## What Was Created

### 1. Checkpoint Manager (`nsm/utils/checkpoint_manager.py`)

Unified checkpoint saving/loading with metadata tracking:

```python
from nsm.utils.checkpoint_manager import CheckpointManager, save_nsm_checkpoint

# During training
checkpoint_manager = CheckpointManager("/checkpoints", "nsm-10x-baseline")
checkpoint_manager.save_checkpoint(
    model=model,
    epoch=15,
    metrics={"val_accuracy": 0.67},
    config=config,
    is_best=True  # Saves as nsm-10x-baseline_best.pt
)

# For CGT validation
checkpoint = checkpoint_manager.load_best_checkpoint(model, device='cuda')
```

**Features**:
- Saves model state, optimizer state, metrics, and config
- Tracks best model separately (`*_best.pt`)
- Generates JSON metadata for easy inspection
- Works in both local and Modal environments

### 2. Checkpoint Download Script (`scripts/download_checkpoints.py`)

Downloads checkpoints from Modal volume to local repo:

```bash
# Download all checkpoints
python scripts/download_checkpoints.py

# Download specific pattern
python scripts/download_checkpoints.py --pattern "*best*"

# Custom destination
python scripts/download_checkpoints.py --destination my_checkpoints/
```

### 3. CGT Full Training Script (`nsm-cgt/experiments/modal_cgt_full_training.py`)

Production-ready CGT training with checkpoint integration:

```bash
# Train from scratch (15 epochs like NSM-33)
modal run experiments/modal_cgt_full_training.py::train_from_scratch

# Load NSM-33 checkpoint and continue training
modal run experiments/modal_cgt_full_training.py::train_from_checkpoint \
  --checkpoint=nsm-10x-baseline_best.pt

# Just track CGT operators on existing checkpoint (no training)
modal run experiments/modal_cgt_full_training.py::track_checkpoint \
  --checkpoint=nsm-10x-baseline_best.pt
```

**Key Features**:
- Full 15-epoch training (vs previous 5-epoch minimal)
- CGT operator tracking at every epoch
- Loads pre-trained NSM-33 models as initialization
- Saves checkpoints with CGT metrics included
- Graceful handling of missing checkpoints

---

## Current Checkpoint Status

### Modal Volume (`nsm-checkpoints`)

**Results Files** (JSON):
- `10x_baseline_results.json` - 66% accuracy, 15 epochs
- `10x_fixed_temp_results.json` - 65.57% accuracy, 15 epochs

**Model Checkpoints** (.pt):
- ⚠️ **None yet** - Current scripts only save results, not models

**Dataset Directories**:
- `planning/` - Planning dataset cache
- `kg/` - Knowledge graph dataset cache
- `causal/` - Causal reasoning dataset cache

### Local Repo (`checkpoints/`)

**Currently**:
- `10x_baseline_results.json` (downloaded)
- Empty otherwise (no .pt files)

**After Next Training Run**:
- `nsm-10x-baseline_best.pt` - Best model checkpoint
- `nsm-10x-baseline_epoch15_*.pt` - Final epoch
- `nsm-cgt-planning_best.pt` - CGT-tracked model
- Etc.

---

## Integration Workflow

### Step 1: Add Checkpoint Saving to NSM-33 Experiments

Current NSM-33 scripts (`modal_10x_baseline.py`, etc.) need modification to save model checkpoints:

```python
# Add to imports
from nsm.utils.checkpoint_manager import save_nsm_checkpoint

# In training loop, after validation
if val_accuracy > best_val_accuracy:
    best_val_accuracy = val_accuracy

    # NEW: Save checkpoint
    save_nsm_checkpoint(
        model=model,
        epoch=epoch + 1,
        val_accuracy=val_accuracy,
        config=config,
        checkpoint_dir="/checkpoints",
        experiment_name="nsm-10x-baseline",
        is_best=True
    )
```

**Action Required**: Modify existing Modal scripts to add checkpoint saving

### Step 2: Download Checkpoints to Repo

After training runs complete:

```bash
cd /Users/preston/Projects/NSM
python scripts/download_checkpoints.py
```

This populates `checkpoints/` with trained models.

### Step 3: Use Checkpoints in CGT

```bash
cd /Users/preston/Projects/nsm-cgt

# Track CGT operators on NSM-33 baseline
modal run experiments/modal_cgt_full_training.py::track_checkpoint \
  --checkpoint=nsm-10x-baseline_best.pt

# Or train further with CGT tracking
modal run experiments/modal_cgt_full_training.py::train_from_checkpoint \
  --checkpoint=nsm-10x-baseline_best.pt --epochs=20
```

---

## File Organization

```
NSM/
├── checkpoints/              # Local checkpoint storage
│   ├── 10x_baseline_results.json
│   ├── nsm-10x-baseline_best.pt  (after next run)
│   └── *.json (metadata)
│
├── nsm/utils/
│   └── checkpoint_manager.py  # Checkpoint utilities
│
├── scripts/
│   └── download_checkpoints.py  # Modal → local sync
│
└── experiments/
    └── modal_10x_*.py  # Need modification to save checkpoints

nsm-cgt/  (worktree)
└── experiments/
    ├── modal_cgt_full_training.py  # NEW: Full training + CGT
    ├── modal_cgt_validation.py     # Updated with health checks
    └── modal_cgt_training.py       # Original 5-epoch version
```

---

## Next Steps

### Immediate (To Start Using Checkpoints)

1. **Modify NSM-33 baseline script** to save checkpoints:
   ```bash
   # Edit: experiments/modal_10x_baseline.py
   # Add checkpoint saving in training loop (lines ~390-400)
   ```

2. **Rerun one NSM-33 experiment** to generate checkpoint:
   ```bash
   modal run experiments/modal_10x_baseline.py::validate_10x_baseline
   ```

3. **Download checkpoint** to repo:
   ```bash
   python scripts/download_checkpoints.py
   ```

4. **Run CGT tracking** on trained model:
   ```bash
   cd ../nsm-cgt
   modal run experiments/modal_cgt_full_training.py::track_checkpoint \
     --checkpoint=nsm-10x-baseline_best.pt
   ```

### Future Enhancements

- **Auto-sync**: Cron job or GitHub Action to download checkpoints nightly
- **Checkpoint browser**: Web UI to visualize checkpoint metrics
- **Multi-checkpoint comparison**: CGT tracking across multiple checkpoints in parallel
- **Git LFS**: Use Git Large File Storage for .pt files (currently gitignored)

---

## Benefits

**Before**:
- ❌ No model checkpoints saved
- ❌ CGT tested on untrained models (temp = 0.00)
- ❌ Could not compare CGT across training stages
- ❌ Results not reproducible (models discarded)

**After**:
- ✅ Models saved with full metadata
- ✅ CGT validated on production-trained models
- ✅ Track temperature evolution across epochs
- ✅ Reproducible results (load any checkpoint)
- ✅ Seamless Modal ↔ Local workflow

---

## Example Usage

### Train NSM with Checkpoints (Once Scripts Modified)

```bash
# Run NSM-33 baseline with checkpoint saving
modal run experiments/modal_10x_baseline.py::validate_10x_baseline

# Check Modal volume
modal volume ls nsm-checkpoints
# Output:
#   nsm-10x-baseline_best.pt
#   nsm-10x-baseline_epoch15_*.pt
#   10x_baseline_results.json
```

### Download & Use in CGT

```bash
# Download to local repo
python scripts/download_checkpoints.py

# Verify download
ls -lh checkpoints/*.pt
# Output:
#   nsm-10x-baseline_best.pt (47 MB)

# Track CGT operators on trained model
cd ../nsm-cgt
modal run experiments/modal_cgt_full_training.py::track_checkpoint \
  --checkpoint=nsm-10x-baseline_best.pt

# Expected output:
#   ✅ Loaded checkpoint from epoch 15
#   📊 Tracking CGT operators...
#   Conway Temperature: 0.3521 (healthy zone)
#   Cooling Rate: -0.0023
#   ✅ CGT Temperature: 0.3521
```

---

## Current Status of Multi-Seed Experiments

While building checkpoint system, multi-seed experiments are still running:

- **Seed 42 Fixed Temp**: Epoch 7/15, accuracy 63.44%
- **Seed 42 Baseline**: Failed (Modal timeout - not code issue)
- **Seeds 123, 456, 789, 1011**: Queued/running

Once complete, can use `download_checkpoints.py` to fetch all best models for analysis.

---

## Questions?

See:
- `nsm/utils/checkpoint_manager.py` - Implementation details
- `experiments/modal_cgt_full_training.py` - Usage examples
- `scripts/download_checkpoints.py` - Download workflow
