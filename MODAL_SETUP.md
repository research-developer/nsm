# Modal.com GPU Training Setup

## Authentication

Modal uses token-based authentication. You need both a `token-id` and `token-secret`.

### Option 1: Use Existing Token (if you have the secret)

```bash
modal token set --token-id ak-jPk4EFbHZ9EqvnNj1J7Gop --token-secret <your-secret>
```

### Option 2: Create New Token

```bash
# This will open a browser for authentication
modal token new
```

### Verify Authentication

```bash
modal profile current
```

## Running NSM Training on Modal

### Quick Validation (10 epochs, ~5-10 minutes on A100)

```bash
cd /Users/preston/Projects/NSM
modal run experiments/modal_train.py::validate_3level
```

This will:
- Train all 3 domains in parallel on A100 GPUs
- Run for 10 epochs each
- Check for class collapse
- Save checkpoints to Modal volume `/checkpoints`

### Full Training (100 epochs)

```bash
# Train all domains in parallel
modal run experiments/modal_train.py::train_all_domains

# Or train individual domains
modal run experiments/modal_train.py::train_planning
modal run experiments/modal_train.py::train_causal
modal run experiments/modal_train.py::train_kg
```

## Monitoring

Modal provides a web dashboard at https://modal.com/apps

You can also stream logs in real-time:

```bash
# While job is running, logs will stream to console
# Use Ctrl+C to detach (job continues running)
```

## Checkpoints

All checkpoints are saved to the persistent Modal volume `nsm-checkpoints`:

```
/checkpoints/
  ├── planning/
  │   ├── best_model.pt
  │   └── modal_results.json
  ├── causal/
  │   ├── best_model.pt
  │   └── modal_results.json
  └── kg/
      ├── best_model.pt
      └── modal_results.json
```

## Cost Estimation

- **A100 GPU**: ~$4/hour
- **10-epoch validation**: ~5-10 minutes = ~$0.33-0.67 per domain
- **100-epoch full training**: ~30-60 minutes = ~$2-4 per domain

Total for 3-level validation: **~$1-2**
Total for full Phase 1.5 training: **~$6-12**

## Troubleshooting

### Authentication Issues

If you get "Unauthorized" errors:

```bash
# Check current profile
modal profile current

# Re-authenticate
modal token new
```

### Volume Issues

If checkpoints aren't persisting:

```bash
# List volumes
modal volume list

# Create volume manually if needed
modal volume create nsm-checkpoints
```

### Image Build Issues

Modal automatically builds the image on first run. If there are dependency issues:

```bash
# Test image build without running training
modal run experiments/modal_train.py --help
```

## Advantages over Local CPU Training

- **50-100x faster**: A100 GPU vs local CPU
- **Parallel training**: All 3 domains simultaneously
- **No local resource usage**: Frees up local machine
- **Automatic checkpointing**: Persistent across runs
- **Auto-retry**: Handles preemption gracefully
- **Pay-per-use**: Only charged for GPU time used

## Next Steps

1. Authenticate with Modal (see above)
2. Run quick validation to verify setup
3. Launch full 100-epoch training
4. Monitor results via web dashboard
5. Download checkpoints when complete
