# NSM Training Validation Framework

**Type-safe validation for NSM training runs with automated bug detection.**

All lessons learned from NSM-10 dataset exploration (3 critical bugs) are encoded as Pydantic validators that catch issues **before** and **during** training.

---

## Features

### 🔍 Pre-Flight Checks (Before Training)
- **Dataset balance**: Catch degenerate label distributions (NSM-10 Bug #1)
- **Dataset size**: Enforce standardized train sizes (~2,000 examples)
- **Hyperparameter bounds**: Validate learned optimal ranges
- **Domain-specific config**: Check pool ratios, task parameters

### 🚨 Checkpoint Validation (During/After Training)
- **Zero task loss**: Detect trivial tasks (NSM-10 Bug #3)
- **Class imbalance collapse**: Catch always-predict-one-class failures
- **Vanishing gradients**: Monitor gradient flow
- **Stuck training**: Detect plateaus
- **Early stopping**: Automatic patience-based stopping
- **Task mismatch**: Detect loss/metric misalignment (NSM-10 KG issue)

### 📊 Experiment Tracking
- **JSONL logging**: Append-only experiment log
- **Domain filtering**: Query by domain type
- **Summary statistics**: Best accuracy per domain
- **Full reproducibility**: All config + metrics saved

---

## Quick Start

```python
from pathlib import Path
from nsm.validation import (
    TestViewController,
    DatasetConfig,
    HyperparametersConfig,
    DomainType,
    EpochMetrics,
)

# Initialize controller
controller = TestViewController(
    experiment_log_path=Path("experiments/training_log.jsonl")
)

# Create validated training run
dataset_config = DatasetConfig(
    domain=DomainType.CAUSAL,
    split="train",
    total_size=2500,
    train_size=2000,
    val_size=500,
    label_balance_class_0=0.50,  # Balanced (40-60%)
    label_balance_class_1=0.50,
    domain_params={
        "num_scenarios": 2500,
        "effectiveness_range": [0.2, 0.9],  # NSM-10 Bug #1 fix
    }
)

hyperparameters = HyperparametersConfig(
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    cycle_loss_weight=0.1,  # <0.2 to avoid class collapse
    pool_ratio=0.25,  # Learned optimal for Causal
)

# Create with automatic validation
training_run = controller.create_training_run(
    run_id="causal_20251020_120000",
    domain=DomainType.CAUSAL,
    dataset_config=dataset_config,
    hyperparameters=hyperparameters,
)

# Run pre-flight checks
preflight_results = controller.run_preflight_checks(training_run)
print(controller.format_test_results(preflight_results))

if not preflight_results["all_passed"]:
    raise ValueError("Pre-flight checks failed!")

# During training: add metrics and validate
for epoch in range(100):
    # ... train ...

    metrics = EpochMetrics(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        task_loss=task_loss,  # MUST be non-zero
        cycle_loss=cycle_loss,
        val_accuracy=val_accuracy,
        accuracy_class_0=acc_class_0,  # Monitor per-class
        accuracy_class_1=acc_class_1,
        grad_norm_mean=grad_norm,
        learning_rate=lr,
    )

    training_run.add_epoch_metrics(metrics)

    # Run checkpoint validation
    checkpoint_results = controller.run_checkpoint_checks(training_run)

    if checkpoint_results["critical_failures"]:
        print("CRITICAL FAILURES DETECTED:")
        for failure in checkpoint_results["critical_failures"]:
            print(f"  {failure['test']}: {failure['message']}")
        break  # Stop training

    if training_run.should_early_stop:
        print(f"Early stopping at epoch {epoch}")
        break

# Log completed run
training_run.status = TrainingStatus.COMPLETED
controller.log_experiment(training_run)
```

---

## NSM-10 Bugs Automatically Caught

### Bug #1: Degenerate Label Distributions
**What happened**: All three datasets only generated positive examples (label=1).

**How caught**:
```python
dataset_config = DatasetConfig(
    label_balance_class_0=0.0,  # ❌ CAUGHT: Must be 0.4-0.6
    label_balance_class_1=1.0,  # ❌ CAUGHT: Must be 0.4-0.6
)
# ValidationError: Input should be greater than or equal to 0.4
```

**Validators**:
- `label_balance_class_0`: `Field(ge=0.4, le=0.6)`
- `label_balance_class_1`: `Field(ge=0.4, le=0.6)`
- `DatasetBalanceTest`: Pre-flight check

### Bug #2: Inconsistent Dataset Sizes
**What happened**: KG had 400 train examples, Causal had 800, Planning had 700.

**How caught**:
```python
dataset_config = DatasetConfig(
    train_size=400,  # ❌ CAUGHT: Must be 1800-2200
)
# ValidationError: Training size 400 outside standardized range [1800, 2200]
```

**Validators**:
- `train_size` validator: Enforces [1800, 2200] range
- `DatasetSizeTest`: Pre-flight check

### Bug #3: Planning Architecture Mismatch
**What happened**: Planning returned individual triples (80,194) instead of complete problems (2,000), causing zero task loss.

**How caught**:
```python
metrics = EpochMetrics(
    epoch=1,
    task_loss=0.0,  # ❌ CAUGHT: Zero task loss
    ...
)
# ValidationError: task_loss=0.00000000 is essentially zero
```

**Validators**:
- `task_loss` validator: `if v < 1e-8: raise ValueError`
- `ZeroTaskLossTest`: Checkpoint validation
- `PlanningTrainingRun.validate_no_zero_task_loss()`: Domain-specific

### Causal Class Imbalance Collapse
**What happened**: Model collapsed to always-predict-class-1 despite balanced data.

**How caught**:
```python
metrics = EpochMetrics(
    accuracy_class_0=0.0,  # Never predicts class 0
    accuracy_class_1=1.0,  # Always predicts class 1
    ...
)
training_run.add_epoch_metrics(metrics)
checkpoint_results = controller.run_checkpoint_checks(training_run)
# CRITICAL FAILURE: class_imbalance_collapse
```

**Validators**:
- `EpochMetrics.has_class_imbalance` computed field
- `ClassImbalanceCollapseTest`: Checkpoint validation

---

## Domain-Specific Validation

### Causal Domain
```python
CausalTrainingRun(
    domain=DomainType.CAUSAL,
    hyperparameters=HyperparametersConfig(
        pool_ratio=0.25,  # ❌ Other values rejected
    ),
    dataset_config=DatasetConfig(
        domain_params={
            "num_scenarios": 2500,  # ±100 tolerance
            "effectiveness_range": [0.2, 0.9],  # Required
        }
    )
)
```

### Knowledge Graph Domain
```python
KnowledgeGraphTrainingRun(
    domain=DomainType.KNOWLEDGE_GRAPH,
    hyperparameters=HyperparametersConfig(
        pool_ratio=0.13,  # ❌ Other values rejected
    ),
    dataset_config=DatasetConfig(
        domain_params={
            "num_entities": 200,  # 180-220 range
            "num_triples": 2500,  # 2400-2600 range
            "use_negative_sampling": True,  # ❌ False rejected
        }
    )
)
```

### Planning Domain
```python
PlanningTrainingRun(
    domain=DomainType.PLANNING,
    hyperparameters=HyperparametersConfig(
        pool_ratio=0.5,  # ❌ Other values rejected
    ),
    dataset_config=DatasetConfig(
        domain_params={
            "num_problems": 2858,  # 2700-2900 range
            "returns_complete_problems": True,  # ❌ False rejected
            "validity_threshold": 0.5,  # ❌ Other values rejected
        }
    )
)
```

---

## Experiment Tracking

### Logging Runs
```python
# Automatically logs to JSONL
controller.log_experiment(training_run)
```

### Querying Experiments
```python
# Load all experiments
all_experiments = controller.load_experiments()

# Filter by domain
causal_experiments = controller.load_experiments(domain=DomainType.CAUSAL)

# Get summary statistics
summary = controller.get_experiment_summary()
print(f"Total experiments: {summary['total_experiments']}")
print(f"Best accuracy by domain: {summary['best_accuracy_by_domain']}")
```

### JSONL Format
Each line is a complete experiment record:
```json
{
  "timestamp": "2025-10-20T12:00:00",
  "run_data": {
    "run_id": "causal_20251020_120000",
    "domain": "causal",
    "status": "completed",
    "dataset_config": {...},
    "hyperparameters": {...},
    "metrics_history": [...],
    "best_val_accuracy": 0.614,
    "best_val_loss": 0.7269,
    ...
  }
}
```

---

## Testing

Run examples to verify system:
```bash
cd /Users/preston/Projects/NSM
PYTHONPATH=. python examples/validation_example.py
```

Examples demonstrate:
1. ✅ Valid Causal run creation
2. ✅ Detecting Bug #1 (degenerate labels)
3. ✅ Detecting Bug #2 (inconsistent size)
4. ✅ Detecting Bug #3 (zero task loss)
5. ✅ Detecting class imbalance collapse
6. ✅ Experiment tracking and querying

---

## Integration with Training Scripts

### Before Training (Pre-Flight)
```python
# In train_causal.py
from nsm.validation import TestViewController, DatasetConfig, ...

controller = TestViewController(Path("experiments/training_log.jsonl"))

# Validate configuration
training_run = controller.create_training_run(...)
preflight_results = controller.run_preflight_checks(training_run)

if not preflight_results["all_passed"]:
    print(controller.format_test_results(preflight_results))
    sys.exit(1)
```

### During Training (Checkpoints)
```python
# After each epoch
metrics = EpochMetrics(...)
training_run.add_epoch_metrics(metrics)

checkpoint_results = controller.run_checkpoint_checks(training_run)

# Check for critical failures
if checkpoint_results["critical_failures"]:
    for failure in checkpoint_results["critical_failures"]:
        logger.error(f"{failure['test']}: {failure['message']}")
    break

# Check early stopping
if training_run.should_early_stop:
    logger.info(f"Early stopping: best epoch {training_run.best_epoch}")
    break
```

### After Training (Logging)
```python
training_run.status = TrainingStatus.COMPLETED
training_run.end_time = datetime.now()
controller.log_experiment(training_run)

print(f"Logged to: {controller.experiment_log_path}")
print(f"Best val accuracy: {training_run.best_val_accuracy:.2%}")
```

---

## Architecture

```
nsm/validation/
├── training_models.py       # Pydantic models with validators
│   ├── TrainingRunBase      # Base class with computed fields
│   ├── CausalTrainingRun    # Domain-specific subclass
│   ├── KnowledgeGraphTrainingRun
│   ├── PlanningTrainingRun
│   ├── DatasetConfig        # Dataset validation
│   ├── HyperparametersConfig  # Hyperparameter bounds
│   └── EpochMetrics         # Per-epoch metrics
│
├── test_controller.py       # Validation controller
│   ├── TestViewController   # Central controller
│   ├── ValidationTest       # Base test class
│   ├── PreFlightTest        # Pre-training tests
│   └── CheckpointTest       # During-training tests
│
└── __init__.py              # Public API
```

---

## Lessons Encoded

From NSM-10 debugging, these constraints are now **automatic**:

1. ✅ Label distributions must be balanced (40-60% per class)
2. ✅ Training sizes must be standardized (~2,000 ±10%)
3. ✅ Task loss must be non-zero (>1e-8)
4. ✅ Per-class accuracy must be monitored
5. ✅ Cycle loss weight must be reasonable (<0.2)
6. ✅ Domain-specific pool ratios are enforced
7. ✅ Domain-specific parameters are validated
8. ✅ Early stopping is automatic
9. ✅ Gradient flow is monitored
10. ✅ All experiments are logged for reproducibility

---

## Future Extensions

Potential additions:
- **Baseline comparisons**: Automatic random/GCN baselines
- **Hyperparameter search**: Integrate with Optuna/Ray Tune
- **Distributed training**: Multi-GPU validation
- **Visualization**: Plotly/Tensorboard integration
- **Alerts**: Slack/email notifications on failures
- **Auto-recovery**: Checkpoint restoration on crashes

---

## References

- **NSM-10**: Dataset Exploration issue
- **NSM-10-FINAL-STATUS.md**: Complete bug analysis
- **NSM-10-CRITICAL-FINDINGS.md**: Detailed findings
- **NSM-10-PLANNING-ZERO-LOSS-ANALYSIS.md**: Bug #3 deep dive
