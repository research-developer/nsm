# NSM Training Notebook Guide

Interactive Jupyter notebook environment for NSM training on Modal A100-40GB GPUs.

## Quick Start

### 1. Launch Notebook

```bash
cd /Users/preston/Projects/NSM
modal run experiments/nsm_training_notebook.py
```

**Expected output:**
```
🎯 Launching NSM Training Notebook...
⏳ This may take 1-2 minutes to provision GPU and load environment

🚀 NSM Training Notebook Starting
============================================================

📊 Environment Info:
  ✓ GPU: NVIDIA A100-SXM4-40GB
  ✓ VRAM: 40.0GB
  ✓ CUDA: 11.8

📁 Volumes:
  ✓ Checkpoints: /checkpoints
  ✓ Found X existing checkpoints

============================================================
🔗 Access your notebook via the URL below
============================================================

View Jupyter Lab at https://your-username--nsm-notebook-notebook.modal.run
```

### 2. Access Notebook

Modal will provide a URL like:
```
https://your-username--nsm-notebook-notebook.modal.run
```

Click the link to open JupyterLab in your browser (no password required).

### 3. Open Dashboard

In JupyterLab, navigate to:
```
NSM_Training_Dashboard.ipynb
```

### 4. Run Training

Execute cells sequentially (Shift+Enter):

1. **Cell 1**: Verify GPU access and environment
2. **Cell 2**: Configure training parameters (modify as needed)
3. **Cell 3**: Load dataset
4. **Cell 4**: Initialize model
5. **Cell 5**: Start training with live monitoring

## Features

### Live Training Visualization

The training loop (Cell 5) displays:
- **Real-time plots** updated every validation epoch
- **Loss curves**: Train/val total loss, cycle loss, task loss
- **Accuracy tracking**: Train/val accuracy over time
- **Learning rate schedule**: Visualize scheduler adjustments
- **GPU memory usage**: Monitor VRAM consumption

### Interactive Controls

**Training Configuration (Cell 2):**
```python
DOMAIN = "causal"  # Change to: "causal", "planning", "kg"
EPOCHS = 100       # Number of training epochs
BATCH_SIZE = 64    # Batch size (reduce if OOM)
LEARNING_RATE = 1e-4
CYCLE_WEIGHT = 0.01
```

**Interrupt Training:**
- Use Kernel → Interrupt to stop training gracefully
- Model state and history are preserved
- Checkpoints saved up to interruption point

### Checkpoint Management (Cell 6-7)

**Browse checkpoints:**
```python
# Lists all checkpoints with metrics
checkpoints = sorted(checkpoint_dir.glob("*.pt"))
```

**Load specific checkpoint:**
```python
checkpoint = torch.load("/checkpoints/causal/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

### Testing & Analysis (Cell 8)

Run full test set evaluation with:
- Accuracy and loss metrics
- Confusion matrix visualization
- Confidence distribution analysis
- Per-class classification report

### Cross-Domain Comparison (Cell 9)

Compare results across all three domains:
- Side-by-side accuracy/loss comparison
- Bar charts for visual comparison
- Automatically loads results from persistent volume

## Advanced Usage

### Change Domain Mid-Session

1. Modify `DOMAIN` in Cell 2
2. Rerun from Cell 2 onwards
3. New checkpoints saved to `/checkpoints/{new_domain}/`

### Resume Training

If training was interrupted:

```python
# Load checkpoint
checkpoint = torch.load(checkpoint_dir / 'checkpoint_epoch_50.pt')

# Restore state
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
history = checkpoint['history']

# Continue training from epoch 51
```

### Experiment with Hyperparameters

Create a new cell for hyperparameter sweeps:

```python
# Hyperparameter sweep
configs = [
    {'lr': 1e-3, 'cycle_weight': 0.01},
    {'lr': 1e-4, 'cycle_weight': 0.05},
    {'lr': 5e-5, 'cycle_weight': 0.1},
]

results = []
for config in configs:
    # Train with config
    # Store results
    results.append(...)
```

### GPU Memory Management

If you encounter OOM errors:

```python
# Reduce batch size
BATCH_SIZE = 32  # or 16

# Clear GPU cache
torch.cuda.empty_cache()

# Check memory usage (Cell 11)
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")
```

## Checkpoint Download

### Download all checkpoints for a domain

```bash
# From your local machine
modal volume get nsm-checkpoints causal ./local_checkpoints/causal
modal volume get nsm-checkpoints planning ./local_checkpoints/planning
modal volume get nsm-checkpoints kg ./local_checkpoints/kg
```

### Download specific checkpoint

```bash
modal volume get nsm-checkpoints causal/best_model.pt ./best_model.pt
```

### List all files in volume

```bash
modal volume ls nsm-checkpoints
modal volume ls nsm-checkpoints/causal
```

## Tips & Tricks

### 1. Notebook Magic Commands

```python
# Time cell execution
%time train_epoch(model, train_loader, optimizer)

# Time entire cell
%%time
# ... cell code ...

# Interactive debugging
%debug

# Show all variables
%whos
```

### 2. Save Plots

```python
# Save current figure
plt.savefig('/checkpoints/causal/training_curves.png', dpi=300, bbox_inches='tight')
```

### 3. TensorBoard Integration

Add a cell for TensorBoard logging:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f'/checkpoints/{DOMAIN}/runs')

# In training loop
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)

# View in notebook
%load_ext tensorboard
%tensorboard --logdir /checkpoints/{DOMAIN}/runs
```

### 4. Model Inspection

```python
# Layer-wise parameter counts
for name, param in model.named_parameters():
    print(f"{name}: {param.shape} ({param.numel():,} params)")

# Gradient inspection
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.6f}")
```

### 5. Quick Validation Run

Test a checkpoint without full training:

```python
# Load checkpoint
checkpoint = torch.load('/checkpoints/causal/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Quick validation
val_metrics = validate(model, val_loader)
print(f"Val Loss: {val_metrics['loss']:.4f}")
print(f"Val Acc: {val_metrics['acc']*100:.2f}%")
```

## Troubleshooting

### GPU Not Detected

Check environment info in Cell 1:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

If GPU is missing, restart the notebook kernel.

### Out of Memory Errors

1. Reduce batch size in Cell 2
2. Clear GPU cache: `torch.cuda.empty_cache()`
3. Restart kernel to free all memory
4. Reduce model size (decrease `hidden_dim`)

### Import Errors

If `nsm` modules aren't found:
```python
import sys
sys.path.insert(0, '/root')
```

Check that NSM code is available:
```bash
# In a terminal cell
!ls -la /root/nsm
```

### Checkpoint Save Failures

Ensure volume is writable:
```python
from pathlib import Path
checkpoint_dir = Path(f"/checkpoints/{DOMAIN}")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
```

Commit volume manually:
```python
import modal
volume = modal.Volume.from_name("nsm-checkpoints")
volume.commit()
```

### Training Hangs

Check DataLoader workers:
```python
# Reduce num_workers if hanging
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0  # Try 0 if hanging
)
```

## Session Management

### Keep Session Alive

Modal notebooks have a 4-hour timeout. To extend:

1. Run a long training job (auto-extends session)
2. Interact with the notebook periodically
3. Use `--detach` for very long runs (run training outside notebook)

### End Session

1. Save all work (checkpoints auto-saved)
2. Commit volume (Cell 10)
3. Stop the notebook: Kernel → Shutdown All Kernels
4. Exit browser or Ctrl+C in terminal

### Resume Later

Checkpoints are persisted in the volume. Simply:
1. Relaunch notebook: `modal run experiments/nsm_training_notebook.py`
2. Load checkpoint in Cell 7
3. Continue from where you left off

## Performance Optimization

### DataLoader Tuning

```python
# Optimal settings for A100-40GB
DataLoader(
    dataset,
    batch_size=64,          # Max that fits in VRAM
    num_workers=4,          # 4 CPU cores
    pin_memory=True,        # Faster CPU→GPU transfer
    persistent_workers=True # Keep workers alive
)
```

### Mixed Precision Training

Add to training loop:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
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

# In model forward
def forward_chunk(x):
    return self.layer(x)

x = checkpoint(forward_chunk, x)
```

## Getting Help

### Check Logs

In the terminal where you ran `modal run`:
- Live logs show container startup and errors
- GPU allocation status
- Volume mount confirmation

### Debug Mode

Enable verbose logging:
```bash
MODAL_LOGLEVEL=DEBUG modal run experiments/nsm_training_notebook.py
```

### Modal Shell

Access running container:
```bash
# List containers
modal container list

# Exec into container
modal container exec <container-id> bash

# Check GPU
nvidia-smi
```

## Resources

- **Modal Docs**: https://modal.com/docs
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io
- **NSM Project**: See `/Users/preston/Projects/NSM/CLAUDE.md`

## Example Workflow

### Full Training Run

```bash
# 1. Launch notebook
modal run experiments/nsm_training_notebook.py

# 2. In browser: Open NSM_Training_Dashboard.ipynb

# 3. Configure (Cell 2)
DOMAIN = "causal"
EPOCHS = 100

# 4. Run cells 1-5 sequentially

# 5. Monitor training in real-time

# 6. After training completes, run cells 6-10 for analysis

# 7. Download checkpoints
modal volume get nsm-checkpoints causal ./results/causal
```

### Quick Validation

```bash
# 1. Launch notebook
modal run experiments/nsm_training_notebook.py

# 2. Run Cell 1 (setup)
# 3. Run Cell 7 (load checkpoint)
# 4. Run Cell 8 (test evaluation)
# 5. Review results
```

### Cross-Domain Comparison

```bash
# Train all domains (can run in parallel)
# 1. Launch 3 notebook instances (different terminals)
# 2. Set DOMAIN in each: "causal", "planning", "kg"
# 3. Run training
# 4. In any notebook, run Cell 9 for comparison
```

## Advanced: Detached Training

For very long runs, consider using the production script instead:

```bash
# Use this for overnight/multi-day runs
modal run --detach modal_train_production.py::train_all
```

Then use the notebook for interactive analysis of results.

---

**Enjoy your interactive NSM training environment!**

Questions? Check the NSM project documentation or Modal support.
