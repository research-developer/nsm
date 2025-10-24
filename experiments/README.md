# NSM Experiments - Interactive Training Notebooks

Interactive Jupyter notebook environment for training NSM models on Modal's A100-40GB GPUs.

## Overview

This directory contains the interactive training infrastructure for the Neural Symbolic Model (NSM) project. Unlike the production batch training script (`modal_train_production.py`), these notebooks provide real-time visualization, interactive debugging, and exploratory analysis.

**Use Cases:**
- Interactive model development and debugging
- Real-time training visualization
- Hyperparameter exploration
- Quick validation runs
- Cross-domain comparison analysis
- Educational demonstrations

## Quick Start

```bash
# From NSM project root
cd /Users/preston/Projects/NSM

# Launch interactive notebook
modal run experiments/nsm_training_notebook.py

# Access the provided URL in your browser
# Open: NSM_Training_Dashboard.ipynb
# Run cells sequentially
```

## Files

### Core Files

- **`nsm_training_notebook.py`** - Modal app that provisions GPU and launches JupyterLab
- **`NSM_Training_Dashboard.ipynb`** - Main interactive notebook with training pipeline
- **`MODAL_NOTEBOOK_GUIDE.md`** - Comprehensive user guide (read this first!)
- **`NOTEBOOK_QUICK_REFERENCE.md`** - One-page cheat sheet for common operations
- **`NOTEBOOK_TEST_CHECKLIST.md`** - Testing checklist for validation
- **`README.md`** - This file

### Related Files

- **`../modal_train_production.py`** - Production batch training (use for overnight runs)
- **`../nsm/data/`** - Dataset implementations (planning, causal, kg)
- **`../nsm/models/`** - Model architectures (hierarchical, R-GCN, etc.)

## Architecture

### Modal Infrastructure

```
Local Machine                    Modal Cloud
─────────────────                ────────────────────────
│ experiments/                   │
│ ├── nsm_training_notebook.py ────→ Container Provision
│ └── NSM_Training_Dashboard.ipynb    │
│                                     ├─ A100-40GB GPU
│                                     ├─ PyTorch 2.1.0
│                                     ├─ PyG + CUDA 11.8
│                                     ├─ JupyterLab
│                                     └─ /checkpoints volume
│
│ Browser ←────────────────────────── Jupyter URL
│   ↓
│ Live Training Visualization
│ Real-time plots & metrics
```

### Notebook Structure

The dashboard notebook is organized into 11 cells:

1. **Setup & Environment** - Verify GPU, imports, paths
2. **Training Configuration** - Set hyperparameters (DOMAIN, EPOCHS, etc.)
3. **Load Dataset** - Load and split data for chosen domain
4. **Initialize Model** - Create NSM model, optimizer, scheduler
5. **Training Loop** - Main training with live plots (long-running)
6. **Checkpoint Browser** - List and inspect saved checkpoints
7. **Load Checkpoint** - Load specific checkpoint for analysis
8. **Test Evaluation** - Full test set evaluation with metrics
9. **Cross-Domain Comparison** - Compare results across domains
10. **Save & Export** - Persist results, commit volume
11. **GPU Diagnostics** - Monitor GPU status and memory

## Features

### Interactive Training
- Live loss/accuracy curves updated every validation epoch
- Real-time GPU memory monitoring
- Progress bars for batches and epochs
- Interrupt/resume capability (Kernel → Interrupt)

### Model Analysis
- Confusion matrices
- Confidence distribution histograms
- Per-class precision/recall/F1
- Layer-wise parameter counts
- Gradient flow inspection

### Checkpoint Management
- Automatic checkpoint saving (best + periodic)
- Browse all checkpoints with metrics
- Load any checkpoint for analysis
- Download to local machine via Modal CLI

### Cross-Domain Support
- Switch domains mid-session (causal, planning, kg)
- Compare performance across domains
- Visualize domain-specific characteristics

## Usage Patterns

### Pattern 1: Full Training Run

```python
# Cell 2: Configure
DOMAIN = "causal"
EPOCHS = 100
BATCH_SIZE = 64

# Run Cells 1-5 sequentially
# Wait for training to complete (~90 minutes)
# Run Cells 6-10 for analysis
```

### Pattern 2: Quick Validation

```python
# Cell 2: Short run
DOMAIN = "planning"
EPOCHS = 10
EVAL_EVERY = 2

# Run Cells 1-5 (completes in ~15 minutes)
```

### Pattern 3: Checkpoint Analysis Only

```python
# Run Cell 1 (setup)
# Run Cell 7 (load checkpoint)
# Run Cell 8 (test evaluation)
# Skip training entirely
```

### Pattern 4: Hyperparameter Search

```python
# Create new cell after Cell 4
configs = [
    {'lr': 1e-3, 'cycle_weight': 0.01},
    {'lr': 1e-4, 'cycle_weight': 0.05},
    {'lr': 5e-5, 'cycle_weight': 0.1},
]

results = []
for config in configs:
    # Modify optimizer
    # Train for 10 epochs
    # Store results
```

## Performance

### Expected Times (A100-40GB)

| Operation | Time |
|-----------|------|
| Container startup | 1-2 minutes |
| Dataset load (2000 graphs) | <30 seconds |
| Model initialization | <5 seconds |
| Single epoch (batch_size=64) | ~45-60 seconds |
| 10 epochs + validation | ~15 minutes |
| 100 epochs + validation | ~90 minutes |

### Expected Metrics (Phase 1.5)

| Domain | Val Accuracy | Val Loss | Notes |
|--------|--------------|----------|-------|
| Causal | ~59% | ~0.68 | Interventions & counterfactuals |
| Planning | ~57% | ~0.70 | PDDL-style preconditions |
| KG | ~54% | ~0.75 | Knowledge graph reasoning |

**No class collapse** - All domains show healthy 3-level hierarchy.

### Resource Usage

| Batch Size | GPU Memory | Training Speed |
|------------|------------|----------------|
| 16 | ~10GB | Slower |
| 32 | ~18GB | Balanced |
| 64 | ~32GB | **Recommended** |
| 128 | ~38GB (may OOM) | Fastest if fits |

## Troubleshooting

### GPU Not Available

```python
# Cell 1 shows: CUDA Available: False
# Solution: Restart kernel (Kernel → Restart)
```

### Out of Memory

```python
# Error: CUDA out of memory
# Solution 1: Reduce batch size in Cell 2
BATCH_SIZE = 32  # or 16

# Solution 2: Clear cache
torch.cuda.empty_cache()

# Solution 3: Restart kernel
```

### Import Errors

```python
# Error: No module named 'nsm'
# Solution: Ensure path is set (Cell 1)
import sys
sys.path.insert(0, '/root')
```

### Training Hangs

```python
# DataLoader hanging
# Solution: Reduce workers
DataLoader(..., num_workers=0)
```

### Volume Issues

```bash
# Checkpoints not persisting
# Solution: Manual commit
import modal
volume = modal.Volume.from_name("nsm-checkpoints")
volume.commit()
```

## Advanced Usage

### TensorBoard Integration

Add cell after Cell 4:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f'/checkpoints/{DOMAIN}/runs')

# In training loop (Cell 5), add:
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)

# View in notebook:
%load_ext tensorboard
%tensorboard --logdir /checkpoints/{DOMAIN}/runs
```

### Mixed Precision Training

Add to Cell 4:

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

Modify training loop in Cell 5:

```python
with autocast():
    out, reconstructed = model(batch)
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Checkpointing

For very large models:

```python
from torch.utils.checkpoint import checkpoint

# In model definition
def forward_chunk(x):
    return self.layer(x)

x = checkpoint(forward_chunk, x)
```

## Downloading Results

### List Volume Contents

```bash
modal volume ls nsm-checkpoints
modal volume ls nsm-checkpoints/causal
```

### Download Checkpoints

```bash
# Download entire domain
modal volume get nsm-checkpoints causal ./local_results/causal

# Download specific checkpoint
modal volume get nsm-checkpoints causal/best_model.pt ./best_causal.pt

# Download all domains
for domain in causal planning kg; do
    modal volume get nsm-checkpoints $domain ./local_results/$domain
done
```

### Load Locally

```python
import torch

checkpoint = torch.load('local_results/causal/best_model.pt')
print(f"Epoch: {checkpoint['epoch']}")
print(f"Val Acc: {checkpoint['val_acc']*100:.2f}%")
```

## Comparison: Notebook vs Production Script

| Feature | Notebook | Production Script |
|---------|----------|-------------------|
| Use case | Interactive, development | Batch training, overnight |
| Visualization | Real-time plots | Logs only |
| Debugging | Interactive (breakpoints) | Limited |
| Speed | Slower (plotting overhead) | Faster |
| Resource usage | Higher (Jupyter) | Lower |
| Session length | 4 hours max | Unlimited |
| Best for | Exploration, tuning | Final training, CI/CD |

**Recommendation:**
- Use **notebook** for development, debugging, and analysis
- Use **production script** for final full-scale training runs

## Tips & Best Practices

### 1. Start Small

Always test with small configurations first:
```python
EPOCHS = 5
BATCH_SIZE = 32
```

### 2. Monitor GPU Memory

Check Cell 11 frequently to ensure no memory leaks.

### 3. Save Often

Checkpoints auto-save, but commit volume manually if doing risky operations:
```python
volume.commit()
```

### 4. Use Command Mode

Learn keyboard shortcuts (see `NOTEBOOK_QUICK_REFERENCE.md`):
- `Shift+Enter` - Run cell
- `A` - Insert cell above
- `B` - Insert cell below
- `D,D` - Delete cell
- `I,I` - Interrupt kernel

### 5. Interrupt Gracefully

Use Kernel → Interrupt instead of kernel restart to preserve state.

### 6. Comment Your Experiments

Add markdown cells to document what you're testing:
```markdown
## Experiment: Higher Cycle Weight
Testing cycle_weight=0.1 vs 0.01 to improve reconstruction.
Expected: Better WHY↔WHAT symmetry, possibly lower task accuracy.
```

### 7. Version Your Checkpoints

Use descriptive names:
```python
torch.save(model.state_dict(),
    f'/checkpoints/{DOMAIN}/experiment_high_cycle_weight.pt')
```

## Resources

### Documentation

- **User Guide**: `MODAL_NOTEBOOK_GUIDE.md` - Read this first!
- **Quick Reference**: `NOTEBOOK_QUICK_REFERENCE.md` - One-page cheat sheet
- **Test Checklist**: `NOTEBOOK_TEST_CHECKLIST.md` - Validation checklist

### External Links

- **Modal Docs**: https://modal.com/docs
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io
- **JupyterLab**: https://jupyterlab.readthedocs.io

### Project Documentation

- **Main README**: `/Users/preston/Projects/NSM/README.md`
- **Claude Guide**: `/Users/preston/Projects/NSM/CLAUDE.md`
- **Phase 1.5 Summary**: `/Users/preston/Projects/NSM/NSM-10-CROSS-DOMAIN-COMPARISON.md`

## Contributing

When adding new features to the notebook:

1. Test thoroughly (see `NOTEBOOK_TEST_CHECKLIST.md`)
2. Add documentation in markdown cells
3. Update this README
4. Update guide documents as needed
5. Ensure backward compatibility

## Support

Issues or questions?

1. Check `MODAL_NOTEBOOK_GUIDE.md` troubleshooting section
2. Review `NOTEBOOK_QUICK_REFERENCE.md` for common operations
3. Check Modal docs for infrastructure issues
4. See main `CLAUDE.md` for NSM architecture questions

## License

Part of the NSM project. See main project LICENSE.

---

**Ready to start?**

```bash
modal run experiments/nsm_training_notebook.py
```

Then open the URL, load `NSM_Training_Dashboard.ipynb`, and run cells 1-5!
