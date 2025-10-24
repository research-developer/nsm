# NSM Interactive Notebook Workflow

Visual guide to the complete notebook workflow.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LOCAL MACHINE                               │
│                                                                     │
│  Terminal                                                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ $ modal run experiments/nsm_training_notebook.py         │      │
│  │                                                          │      │
│  │ 🚀 NSM Training Notebook Starting                        │      │
│  │ ✓ GPU: NVIDIA A100-SXM4-40GB                            │      │
│  │ ✓ VRAM: 40.0GB                                          │      │
│  │                                                          │      │
│  │ View at: https://username--nsm-notebook.modal.run       │      │
│  └──────────────────────────────────────────────────────────┘      │
│                          ↓                                          │
│  Browser                                                            │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  JupyterLab Interface                                    │      │
│  │  ┌────────────────────────────────────────────────┐      │      │
│  │  │ NSM_Training_Dashboard.ipynb                   │      │      │
│  │  │                                                │      │      │
│  │  │ Cell 1: Setup ✓                               │      │      │
│  │  │ Cell 2: Config [EPOCHS=100, DOMAIN="causal"]   │      │      │
│  │  │ Cell 3: Load Data ✓                           │      │      │
│  │  │ Cell 4: Init Model ✓                          │      │      │
│  │  │ Cell 5: Training [▓▓▓▓▓▓▓░░░] 75%             │      │      │
│  │  │                                                │      │      │
│  │  │ 📊 Live Plots:                                 │      │      │
│  │  │ [Loss curves, accuracy, cycle consistency]     │      │      │
│  │  └────────────────────────────────────────────────┘      │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
                                 ↕
┌─────────────────────────────────────────────────────────────────────┐
│                         MODAL CLOUD                                 │
│                                                                     │
│  Container (A100-40GB GPU)                                          │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ /root/                                                   │      │
│  │ ├── nsm/                    ← Your codebase              │      │
│  │ │   ├── data/               ← Datasets                  │      │
│  │ │   ├── models/             ← Model architectures       │      │
│  │ │   └── training/           ← Training utilities        │      │
│  │ └── NSM_Training_Dashboard.ipynb                        │      │
│  │                                                          │      │
│  │ /checkpoints/               ← Persistent Volume          │      │
│  │ ├── causal/                                             │      │
│  │ │   ├── best_model.pt                                   │      │
│  │ │   ├── checkpoint_epoch_50.pt                          │      │
│  │ │   └── training_history.json                           │      │
│  │ ├── planning/                                           │      │
│  │ └── kg/                                                 │      │
│  │                                                          │      │
│  │ PyTorch 2.1.0 + CUDA 11.8 + PyG 2.4.0                   │      │
│  │ JupyterLab + Widgets + Matplotlib                       │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Cell Execution Flow

```
┌─────────────────┐
│ Cell 1: Setup   │  5 seconds
│                 │  • Import libraries
│ Environment     │  • Check GPU
│ GPU Check       │  • Verify paths
│ Imports         │  • Set style
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Cell 2: Config  │  Instant
│                 │  • Set DOMAIN
│ DOMAIN="causal" │  • Set EPOCHS
│ EPOCHS=100      │  • Set BATCH_SIZE
│ BATCH_SIZE=64   │  • Set LR, weights
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Cell 3: Dataset │  20-30 seconds
│                 │  • Generate/load data
│ Load Data       │  • 70/15/15 split
│ Create Loaders  │  • Build DataLoaders
│ Sample Graph    │  • Show sample
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Cell 4: Model   │  5 seconds
│                 │  • Initialize NSM
│ Init NSM Model  │  • Create optimizer
│ Optimizer       │  • Setup scheduler
│ Scheduler       │  • Count params
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Cell 5: Train   │  15 min (10 ep)
│                 │  90 min (100 ep)
│ Training Loop   │  • Train epoch
│ Live Plots      │  • Validate
│ Checkpointing   │  • Update plots
│ GPU Monitoring  │  • Save checkpoints
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Cell 6: Browse  │  Instant
│                 │  • List checkpoints
│ List Checkpoints│  • Show metrics
│ Metrics Table   │  • Display sizes
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Cell 7: Load    │  5 seconds
│                 │  • Load checkpoint
│ Load Best Model │  • Restore weights
│ Show Metrics    │  • Display info
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Cell 8: Test    │  30 seconds
│                 │  • Run test set
│ Test Evaluation │  • Confusion matrix
│ Confusion Matrix│  • Confidence dist
│ Classification  │  • Classification rpt
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Cell 9: Compare │  Instant
│                 │  • Load all domains
│ Cross-Domain    │  • Compare metrics
│ Bar Charts      │  • Plot charts
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Cell 10: Export │  5 seconds
│                 │  • Save history JSON
│ Save Results    │  • Save results JSON
│ Commit Volume   │  • Commit volume
│ Download Info   │  • Show commands
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Cell 11: GPU    │  Instant
│                 │  • GPU info
│ Diagnostics     │  • Memory stats
│ Memory Stats    │  • Clear cache
│ Cache Clear     │
└─────────────────┘
```

---

## Typical User Workflows

### Workflow 1: Quick Validation (15 minutes)

```
1. Launch notebook
   ↓
2. Cell 1 - Verify environment ✓
   ↓
3. Cell 2 - Set EPOCHS=10, EVAL_EVERY=2
   ↓
4. Cell 3 - Load dataset
   ↓
5. Cell 4 - Init model
   ↓
6. Cell 5 - Train (watch live plots)
   ↓
7. Cell 8 - Test evaluation
   ↓
✅ Done - Results validated
```

### Workflow 2: Full Training Run (90 minutes)

```
1. Launch notebook
   ↓
2. Cell 1 - Verify environment ✓
   ↓
3. Cell 2 - Set EPOCHS=100, BATCH_SIZE=64
   ↓
4. Cell 3 - Load dataset
   ↓
5. Cell 4 - Init model
   ↓
6. Cell 5 - Train (monitor progress)
   ↓
7. Cell 8 - Test evaluation
   ↓
8. Cell 10 - Save & export
   ↓
9. Download checkpoints:
   $ modal volume get nsm-checkpoints causal ./results/
   ↓
✅ Done - Full training complete
```

### Workflow 3: All Domains Comparison (5 hours)

```
1. Launch notebook
   ↓
2. Train Causal:
   Cell 2: DOMAIN="causal", EPOCHS=100
   Cells 3-5: Train
   ↓
3. Train Planning:
   Cell 2: DOMAIN="planning", EPOCHS=100
   Cells 3-5: Train (rerun)
   ↓
4. Train KG:
   Cell 2: DOMAIN="kg", EPOCHS=100
   Cells 3-5: Train (rerun)
   ↓
5. Cell 9 - Cross-domain comparison
   ↓
6. Cell 10 - Export all results
   ↓
✅ Done - Full cross-domain analysis
```

### Workflow 4: Checkpoint Analysis Only (5 minutes)

```
1. Launch notebook
   ↓
2. Cell 1 - Setup
   ↓
3. Cell 6 - Browse checkpoints
   ↓
4. Cell 7 - Load best checkpoint
   ↓
5. Cell 8 - Test evaluation
   ↓
6. Cell 9 - Cross-domain comparison
   ↓
✅ Done - Analysis without training
```

### Workflow 5: Hyperparameter Exploration (variable)

```
1. Launch notebook
   ↓
2. Cell 1 - Setup
   ↓
3. Cell 2 - Config 1: LR=1e-3, CYCLE_WEIGHT=0.01
   ↓
4. Cells 3-5 - Train (10 epochs)
   ↓
5. Note results
   ↓
6. Cell 2 - Config 2: LR=1e-4, CYCLE_WEIGHT=0.05
   ↓
7. Cells 3-5 - Train (10 epochs)
   ↓
8. Note results
   ↓
9. Cell 2 - Config 3: LR=5e-5, CYCLE_WEIGHT=0.1
   ↓
10. Cells 3-5 - Train (10 epochs)
   ↓
11. Compare results
   ↓
✅ Done - Optimal config identified
```

---

## Data Flow

```
┌──────────────┐
│ User Input   │
│ (Cell 2)     │
│ DOMAIN       │
│ EPOCHS       │
│ BATCH_SIZE   │
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Dataset Gen  │  PlanningTripleDataset
│ (Cell 3)     │  CausalDataset
│              │  KGDataset
│ 2000 graphs  │
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ DataLoader   │  batch_size=64
│              │  num_workers=4
│              │  pin_memory=True
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ NSM Model    │  3-level hierarchy
│ (Cell 4)     │  ~200K-500K params
│              │  R-GCN + Pooling
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Training     │  Forward pass
│ (Cell 5)     │  Loss computation
│              │  Backward pass
│              │  Optimizer step
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Checkpoints  │  best_model.pt
│              │  checkpoint_epoch_*.pt
│              │  training_history.json
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Results      │  Test accuracy
│ (Cell 8-10)  │  Confusion matrix
│              │  Cross-domain comp
└──────────────┘
```

---

## Checkpoint Lifecycle

```
Training Epoch
     │
     ├─ Every epoch: Update history
     │
     ├─ Every EVAL_EVERY epochs:
     │  ├─ Run validation
     │  ├─ Update plots
     │  └─ If best: Save best_model.pt
     │
     └─ Every SAVE_EVERY epochs:
        └─ Save checkpoint_epoch_N.pt

Saved Files:
  /checkpoints/{domain}/
  ├── best_model.pt                  ← Best validation loss
  ├── checkpoint_epoch_10.pt         ← Every 10 epochs
  ├── checkpoint_epoch_20.pt
  ├── ...
  ├── training_history.json          ← Full training curves
  └── final_results.json             ← Summary metrics

Background:
  Volume auto-commits in background
  Manual commit: volume.commit()

Retrieval:
  In notebook: Cell 6 (browse), Cell 7 (load)
  Local download: modal volume get ...
```

---

## Interactive Features

### Live Plotting

```
Every EVAL_EVERY epochs (default: 5):

┌─────────────────────────────────────────────┐
│ Training & Validation Plots (6 subplots)    │
│                                             │
│ ┌──────┐ ┌──────┐ ┌──────┐                │
│ │ Loss │ │ Acc  │ │Cycle │                │
│ │      │ │      │ │Loss  │                │
│ └──────┘ └──────┘ └──────┘                │
│                                             │
│ ┌──────┐ ┌──────┐ ┌──────┐                │
│ │ Task │ │  LR  │ │ GPU  │                │
│ │ Loss │ │      │ │ Mem  │                │
│ └──────┘ └──────┘ └──────┘                │
│                                             │
│ Latest Metrics:                             │
│ Train Loss: 0.6234 | Acc: 62.45%           │
│ Val Loss: 0.6812 | Acc: 59.23%             │
│ Cycle Loss: 0.1234                          │
└─────────────────────────────────────────────┘

Updates automatically every validation cycle!
```

### Progress Bars

```
Training:   █████████████████████░░░░░  75% [75/100 epochs]
Batch:      ████████████████████████░  96% [27/28 batches]
GPU Memory: ████████████░░░░░░░░░░░░░  32.4GB / 40.0GB
```

### Interrupt & Resume

```
User presses: Kernel → Interrupt (or I, I)
              ↓
Training stops gracefully
              ↓
Checkpoints preserved
              ↓
Load checkpoint in Cell 7
              ↓
Modify Cell 5 to resume from epoch N
              ↓
Continue training
```

---

## Multi-Domain Training

```
Session 1: Causal
┌────────────────────────┐
│ Cell 2: DOMAIN="causal"│
│ Cells 3-5: Train       │
│ → /checkpoints/causal/ │
└────────────────────────┘

Session 2: Planning (same notebook)
┌────────────────────────┐
│ Cell 2: DOMAIN="planning"│
│ Cells 3-5: Train       │
│ → /checkpoints/planning/│
└────────────────────────┘

Session 3: KG (same notebook)
┌────────────────────────┐
│ Cell 2: DOMAIN="kg"    │
│ Cells 3-5: Train       │
│ → /checkpoints/kg/     │
└────────────────────────┘

Then:
┌────────────────────────┐
│ Cell 9: Compare all 3  │
│ Bar charts, tables     │
└────────────────────────┘
```

---

## Error Handling Flow

```
Error Occurs
    │
    ├─ GPU OOM?
    │  → Reduce BATCH_SIZE in Cell 2
    │  → Rerun Cells 3-5
    │
    ├─ Import Error?
    │  → Check sys.path in Cell 1
    │  → Verify /root/nsm exists
    │
    ├─ Training Hangs?
    │  → Set num_workers=0 in Cell 3
    │  → Restart kernel
    │
    ├─ Checkpoint Load Fail?
    │  → Check file path in Cell 6
    │  → Verify volume commit in Cell 10
    │
    └─ Unexpected Error?
       → Check error traceback
       → Review MODAL_NOTEBOOK_GUIDE.md troubleshooting
       → Modal container exec for debugging
```

---

## Resource Monitoring

```
Cell 11: GPU Diagnostics
┌──────────────────────────────────────┐
│ 🎮 GPU Information                   │
│   Device: NVIDIA A100-SXM4-40GB      │
│   CUDA: 11.8                         │
│                                      │
│ Memory:                              │
│   Total:     40.00GB                 │
│   Allocated: 32.45GB (81.1%)         │
│   Reserved:  34.20GB (85.5%)         │
│   Free:       7.55GB (18.9%)         │
│                                      │
│ Optimizations:                       │
│   ✓ TF32 (matmul): Enabled          │
│   ✓ TF32 (cudnn): Enabled           │
│   ✓ cuDNN Benchmark: Auto           │
│                                      │
│ 🧹 Clear cache: torch.cuda.empty_cache()│
└──────────────────────────────────────┘

Run anytime to check GPU status!
```

---

## Summary

**11 Cells** → **5 Workflows** → **3 Domains** → **Complete Training Pipeline**

**Key Features:**
- Live visualization
- Interactive debugging
- Checkpoint management
- Cross-domain comparison
- Full reproducibility

**Next Step:**
```bash
modal run experiments/nsm_training_notebook.py
```

Open the URL, load the notebook, and start training! 🚀
