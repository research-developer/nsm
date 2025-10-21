# NSM Notebook Quick Reference

One-page cheat sheet for common operations.

## Launch & Access

```bash
# Start notebook
modal run experiments/nsm_training_notebook.py

# Access URL (printed in terminal)
https://your-username--nsm-notebook-notebook.modal.run
```

## Training Configuration (Cell 2)

```python
DOMAIN = "causal"          # or "planning", "kg"
EPOCHS = 100              # Training epochs
BATCH_SIZE = 64           # Reduce if OOM: 32, 16
LEARNING_RATE = 1e-4      # Learning rate
CYCLE_WEIGHT = 0.01       # Cycle consistency weight
```

## Common Operations

### Start Training
```python
# Run Cell 5 - training loop with live plots
# Press Shift+Enter or click Run button
```

### Interrupt Training
```python
# Kernel → Interrupt (or press 'i' twice in command mode)
# Checkpoints preserved up to last save point
```

### Load Checkpoint
```python
checkpoint = torch.load("/checkpoints/causal/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

### Clear GPU Memory
```python
torch.cuda.empty_cache()
print(f"Free: {(40 - torch.cuda.memory_allocated(0)/1e9):.1f}GB")
```

### Save Plot
```python
plt.savefig('/checkpoints/causal/my_plot.png', dpi=300)
```

## Checkpoint Management

### List Checkpoints
```python
# Cell 6 - shows table of all checkpoints
list(Path("/checkpoints/causal").glob("*.pt"))
```

### Download Locally
```bash
# From your local machine
modal volume get nsm-checkpoints causal ./local_checkpoints/causal
modal volume get nsm-checkpoints causal/best_model.pt ./best.pt
```

### Commit Volume
```python
import modal
volume = modal.Volume.from_name("nsm-checkpoints")
volume.commit()
```

## Troubleshooting

### GPU Not Available
```python
# Check CUDA
import torch
torch.cuda.is_available()  # Should be True
torch.cuda.get_device_name(0)  # Should show A100

# If False, restart kernel: Kernel → Restart
```

### Out of Memory
```python
# 1. Reduce batch size in Cell 2
BATCH_SIZE = 32  # or 16

# 2. Clear cache
torch.cuda.empty_cache()

# 3. Restart kernel
# Kernel → Restart & Clear Output
```

### Import Errors
```python
# Add NSM to path
import sys
sys.path.insert(0, '/root')

# Verify
!ls /root/nsm
```

### Training Hangs
```python
# Reduce DataLoader workers
DataLoader(..., num_workers=0)  # or 2
```

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Run cell | Shift+Enter |
| Run cell (stay) | Ctrl+Enter |
| Insert cell above | A (command mode) |
| Insert cell below | B (command mode) |
| Delete cell | D,D (command mode) |
| Undo delete | Z (command mode) |
| Change to code | Y (command mode) |
| Change to markdown | M (command mode) |
| Command mode | Esc |
| Edit mode | Enter |
| Interrupt kernel | I,I (command mode) |
| Restart kernel | 0,0 (command mode) |

## Quick Validation

```python
# 1. Load best model (Cell 7)
checkpoint = torch.load("/checkpoints/causal/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# 2. Evaluate on test set (Cell 8)
test_metrics = validate(model, test_loader)
print(f"Acc: {test_metrics['acc']*100:.2f}%")
```

## Performance Tips

### Optimal DataLoader
```python
DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

### Mixed Precision (if needed)
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    out, reconstructed = model(batch)
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Monitor GPU
```python
# Cell 11 - full diagnostics
# Or quick check:
print(f"Allocated: {torch.cuda.memory_allocated(0)/1e9:.1f}GB")
```

## Cell Execution Order

Recommended sequence for first run:

1. **Cell 1**: Setup & Environment Check ✓
2. **Cell 2**: Training Configuration ✓
3. **Cell 3**: Load Dataset ✓
4. **Cell 4**: Initialize Model ✓
5. **Cell 5**: Training Loop (long-running)
6. **Cell 6**: Checkpoint Browser
7. **Cell 7**: Load Checkpoint
8. **Cell 8**: Test Evaluation
9. **Cell 9**: Cross-Domain Comparison
10. **Cell 10**: Save & Export
11. **Cell 11**: GPU Diagnostics (anytime)

## Useful Commands

```python
# Check environment
!nvidia-smi
!df -h /checkpoints

# List files
!ls -lh /checkpoints/causal

# Python info
!python --version
!pip list | grep torch

# Model summary
print(model)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Plot current figure
plt.gcf()  # Get current figure
plt.gca()  # Get current axes
```

## Magic Commands

```python
%time code()              # Time single statement
%%time                    # Time entire cell
%timeit code()           # Benchmark statement
%debug                   # Drop into debugger
%who                     # List variables
%whos                    # Detailed variable list
%load_ext tensorboard    # Load TensorBoard
%matplotlib inline       # Inline plots (default)
```

## File Paths

```
/root/                          # Project root
/root/nsm/                      # NSM codebase
/checkpoints/                   # Persistent volume
/checkpoints/{domain}/          # Per-domain checkpoints
/checkpoints/{domain}/best_model.pt
/checkpoints/{domain}/checkpoint_epoch_X.pt
/checkpoints/{domain}/training_history.json
/checkpoints/{domain}/final_results.json
```

## Result Files

After training, find:
- `best_model.pt` - Best validation checkpoint
- `checkpoint_epoch_X.pt` - Periodic checkpoints
- `training_history.json` - Full training curves
- `final_results.json` - Summary metrics

## Switch Domain Mid-Session

```python
# 1. Change in Cell 2
DOMAIN = "planning"  # was "causal"

# 2. Rerun from Cell 2
#    (Cells 2, 3, 4, 5 in sequence)

# 3. New checkpoints saved to /checkpoints/planning/
```

## Exit Session

```python
# 1. Save checkpoints (automatic during training)
# 2. Commit volume (Cell 10)
volume.commit()

# 3. Shutdown
# Kernel → Shutdown All Kernels

# 4. Ctrl+C in launch terminal
```

## Resume Training

```python
# Load checkpoint with history
checkpoint = torch.load("/checkpoints/causal/checkpoint_epoch_50.pt")

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
history = checkpoint['history']

# Modify Cell 5 to start from epoch 51
for epoch in range(51, EPOCHS + 1):
    # ... training loop
```

## Expected Metrics (Phase 1.5)

| Domain | Val Acc | Val Loss | Status |
|--------|---------|----------|--------|
| Causal | ~59% | ~0.68 | ✓ Validated |
| Planning | ~57% | ~0.70 | ✓ Validated |
| KG | ~54% | ~0.75 | ✓ Validated |

No class collapse - 3-level hierarchy working!

---

**Quick Start**: Run cells 1-5 sequentially → watch training → run cells 6-10 for analysis
