# NSM Experiments - Agent & Experiment Tracking Guide

Complete guide for understanding and working with experiment logs in the NSM project.

## Overview

The NSM project uses **JSON Lines (.jsonl)** format for experiment tracking. Each line is a self-contained JSON object representing a single experiment run, enabling both human readability and programmatic analysis.

**Two primary log files:**
- **`baselines.jsonl`** - Historical baseline results (root directory)
- **`training_log.jsonl`** - Detailed training runs (experiments directory)

## Quick Start

### Reading Experiment Logs

```python
import json

# Read all experiments
experiments = []
with open('experiments/training_log.jsonl', 'r') as f:
    for line in f:
        experiments.append(json.loads(line))

# Get latest experiment
latest = experiments[-1]
print(f"Run: {latest['run_data']['run_id']}")
print(f"Accuracy: {latest['run_data']['best_val_accuracy']}")
```

### Adding a New Experiment

```python
import json
from datetime import datetime

experiment_entry = {
    "timestamp": datetime.utcnow().isoformat(),
    "run_data": {
        "run_id": "my_experiment_20251023",
        "domain": "planning",
        "status": "completed",
        # ... (see schema below)
    }
}

with open('experiments/training_log.jsonl', 'a') as f:
    f.write(json.dumps(experiment_entry) + '\n')
```

## File Formats

### 1. baselines.jsonl (Baseline Results)

**Location**: `/home/user/nsm/baselines.jsonl`

**Purpose**: Track baseline experiments and architectural comparisons

**Schema**:
```json
{
  "branch": "main",                          // Git branch
  "commit": "b77f986",                       // Git commit hash (short)
  "timestamp": "2025-10-21T00:00:00Z",      // ISO 8601 format
  "experiment": "6level_initial",            // Experiment identifier
  "metrics": {
    "accuracy": 0.5322,                      // Primary metric
    "balance_delta": 0.3997,                 // Class balance (0=perfect, 1=total collapse)
    "cycle_loss": 1.53,                      // WHY↔WHAT reconstruction loss
    "cycle_loss_upper": null,                // Upper level cycle loss (if applicable)
    "cycle_loss_lower": null,                // Lower level cycle loss (if applicable)
    "cycle_loss_cross": null,                // Cross-level cycle loss (if applicable)
    "q_neural": null,                        // Fusion plasma Q (physics validation)
    "temperature_gradient": null,            // Temperature control metrics
    "lawson_criterion": null,                // Physics-based validation
    "beta_limit": null                       // Stability metric
  },
  "config": {
    "variant": "6level_full",                // Architecture variant
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "cycle_weight": 0.01,                    // Cycle loss weight (λ_cycle)
    "diversity_weight": 0.0,                 // Diversity regularization
    "pool_ratio": 0.5,                       // Pooling compression ratio
    "dropout": 0.1,
    "node_features": 64,                     // Feature dimensionality
    "num_relations": 16,                     // Number of edge types (R-GCN)
    "num_classes": 2                         // Classification classes
  },
  "notes": "Human-readable experiment description"
}
```

**Key Metrics Explained**:
- **accuracy**: Validation accuracy (target: >0.55 for Phase 1.5)
- **balance_delta**: `|acc_class_0 - acc_class_1|` (target: <0.40)
- **cycle_loss**: Reconstruction error for WHY(WHAT(x)) ≈ x (target: <0.20)
- **q_neural**: Neural fusion quality factor (physics experiments only)

### 2. training_log.jsonl (Detailed Training Runs)

**Location**: `/home/user/nsm/experiments/training_log.jsonl`

**Purpose**: Comprehensive training run logs with full provenance

**Schema**:
```json
{
  "timestamp": "2025-10-21T00:00:00.000000",
  "run_data": {
    // Identification
    "run_id": "baseline_single_pass_20251021",
    "domain": "planning",                    // Dataset: planning, causal, knowledge_graph
    "status": "completed",                   // Status: running, completed, failed

    // Dataset Configuration
    "dataset_config": {
      "domain": "planning",
      "split": "train",
      "total_size": 2858,
      "train_size": 2000,
      "val_size": 429,
      "label_balance_class_0": 0.5,
      "label_balance_class_1": 0.5,
      "domain_params": {},                   // Domain-specific parameters
      "is_balanced": true
    },

    // Hyperparameters
    "hyperparameters": {
      "epochs": 10,
      "batch_size": 64,
      "learning_rate": 0.0001,
      "seed": 42,
      "cycle_loss_weight": 0.01,
      "patience": 20,                        // Early stopping patience
      "min_delta": 0.001,                    // Early stopping threshold
      "grad_clip_norm": null,                // Gradient clipping (if used)
      "pool_ratio": 0.5,                     // Pooling compression
      "use_dual_pass": false,                // Dual-pass architecture flag
      "fusion_mode": null                    // Fusion strategy: equal, learned, null
    },

    // Architecture (Optional)
    "architecture": {
      "variant": "baseline_single_pass",
      "description": "3-level hierarchy with single bottom-up pass",
      "num_levels": 3,
      "passes": 1,                           // 1 or 2 (dual-pass)
      "fusion_weights": null                 // Fusion configuration
    },

    // Results
    "metrics_history": [],                   // Per-epoch metrics (optional)
    "best_val_loss": 0.793800413608551,
    "best_val_accuracy": 0.435,
    "best_epoch": null,                      // Epoch of best validation

    // Final Metrics (Detailed)
    "final_metrics": {
      "accuracy": 0.435,
      "accuracy_class_0": 0.004424778761061947,
      "accuracy_class_1": 0.9942528735632183,
      "class_balance_delta": 0.9898280948021564,
      "task_loss": 0.6968503168651036,
      "cycle_loss": 0.793800413608551
    },

    // Timing
    "training_time_seconds": 33.966574,
    "start_time": "2025-10-21T00:00:00Z",
    "end_time": "2025-10-21T00:00:34Z",

    // Execution Context
    "pid": null,                             // Process ID (if tracked)
    "log_path": null,                        // Path to detailed logs
    "checkpoint_dir": null,                  // Checkpoint directory

    // Experiment Metadata
    "experiment_type": "dual_pass_validation",
    "error_message": null,                   // Error details if failed
    "findings": "Human-readable summary of results",

    // Domain-Specific Metrics (conditionally present)
    "counterfactual_accuracy": null,         // Causal domain
    "intervention_accuracy": null,           // Causal domain
    "hits_at_10": null,                      // Knowledge graph domain
    "mrr": null,                             // Knowledge graph: Mean Reciprocal Rank
    "analogical_reasoning_acc": null,        // Knowledge graph domain
    "goal_achievement_rate": null,           // Planning domain
    "temporal_ordering_acc": null,           // Planning domain

    // Training State (for resumable runs)
    "current_epoch": 0,
    "is_stuck": false,                       // Training stuck detection
    "should_early_stop": false,
    "has_converged": false,
    "has_task_mismatch": false               // Architecture mismatch flag
  }
}
```

## Experiment Types

### Baseline Comparisons (baselines.jsonl)

**Variants**:
- `6level_full` - Full 6-level hierarchy (NSM-33 pilot)
- `3level_fusion` - 3-level with fusion layer
- `3level_attention` - 3-level with multi-head attention
- `baseline_single_pass` - Standard bottom-up only

**Key Comparisons**:
```python
# Load baselines
import json
baselines = []
with open('baselines.jsonl', 'r') as f:
    for line in f:
        baselines.append(json.loads(line))

# Compare variants
for exp in baselines:
    print(f"{exp['experiment']}: "
          f"acc={exp['metrics']['accuracy']:.3f}, "
          f"balance={exp['metrics']['balance_delta']:.3f}")
```

### Training Runs (training_log.jsonl)

**Experiment Types**:
1. **Domain Exploration** (`experiment_type: "domain_exploration"`)
   - Compare planning vs causal vs knowledge_graph
   - Domain-specific metrics populated

2. **Dual-Pass Validation** (`experiment_type: "dual_pass_validation"`)
   - Test dual-pass architectures
   - Fusion mode variations (equal, learned, attention)

3. **Hyperparameter Search** (`experiment_type: "hyperparam_search"`)
   - Sweep cycle_weight, pool_ratio, learning_rate
   - Automated grid/random search logs

4. **Physics Validation** (`experiment_type: "physics_validation"`)
   - Temperature control experiments
   - Lawson criterion tracking
   - Adaptive control validation

## Domain-Specific Metrics

### Causal Domain
```python
"counterfactual_accuracy": 0.72,      # Accuracy on counterfactual queries
"intervention_accuracy": 0.68         # Accuracy on intervention tasks
```

**Use Cases**:
- Counterfactual reasoning ("What if X had not happened?")
- Intervention prediction ("What happens if we change Y?")

### Knowledge Graph Domain
```python
"hits_at_10": 0.85,                   # Top-10 retrieval accuracy
"mrr": 0.62,                          # Mean Reciprocal Rank
"analogical_reasoning_acc": 0.58      # A:B::C:? analogy tasks
```

**Use Cases**:
- Link prediction
- Entity retrieval
- Analogical reasoning

### Planning Domain
```python
"goal_achievement_rate": 0.64,        # Fraction of valid plans reaching goal
"temporal_ordering_acc": 0.71         # Accuracy of action sequencing
```

**Use Cases**:
- PDDL-style planning
- Precondition validation
- Goal decomposition

## Analysis Recipes

### 1. Find Best Performing Experiment

```python
import json

def find_best_run(domain="planning", metric="best_val_accuracy"):
    """Find best run for a domain."""
    best_run = None
    best_score = -1

    with open('experiments/training_log.jsonl', 'r') as f:
        for line in f:
            exp = json.loads(line)
            if exp['run_data']['domain'] == domain:
                score = exp['run_data'].get(metric, -1)
                if score and score > best_score:
                    best_score = score
                    best_run = exp

    return best_run

best = find_best_run("planning")
print(f"Best planning run: {best['run_data']['run_id']}")
print(f"Accuracy: {best['run_data']['best_val_accuracy']}")
```

### 2. Compare Fusion Modes

```python
def compare_fusion_modes():
    """Compare dual-pass fusion strategies."""
    results = {}

    with open('experiments/training_log.jsonl', 'r') as f:
        for line in f:
            exp = json.loads(line)
            hp = exp['run_data']['hyperparameters']

            if hp.get('use_dual_pass'):
                mode = hp.get('fusion_mode', 'none')
                acc = exp['run_data']['best_val_accuracy']
                balance = exp['run_data']['final_metrics']['class_balance_delta']

                results[mode] = {
                    'accuracy': acc,
                    'balance_delta': balance
                }

    return results

fusion_comparison = compare_fusion_modes()
for mode, metrics in fusion_comparison.items():
    print(f"{mode}: acc={metrics['accuracy']:.3f}, "
          f"balance={metrics['balance_delta']:.3f}")
```

### 3. Track Experiment Over Time

```python
import matplotlib.pyplot as plt
from datetime import datetime

def plot_experiment_progress(experiment_type="dual_pass_validation"):
    """Plot accuracy over time for an experiment type."""
    timestamps = []
    accuracies = []

    with open('experiments/training_log.jsonl', 'r') as f:
        for line in f:
            exp = json.loads(line)
            if exp['run_data'].get('experiment_type') == experiment_type:
                ts = datetime.fromisoformat(exp['timestamp'])
                acc = exp['run_data']['best_val_accuracy']

                timestamps.append(ts)
                accuracies.append(acc)

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, accuracies, marker='o')
    plt.xlabel('Time')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Progress: {experiment_type}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{experiment_type}_progress.png')

plot_experiment_progress()
```

### 4. Generate Experiment Report

```python
def generate_report(output_file='experiment_report.md'):
    """Generate markdown report from training logs."""
    experiments = []

    with open('experiments/training_log.jsonl', 'r') as f:
        for line in f:
            experiments.append(json.loads(line))

    with open(output_file, 'w') as out:
        out.write('# NSM Experiment Report\n\n')
        out.write(f'Total Experiments: {len(experiments)}\n\n')

        # Group by domain
        domains = {}
        for exp in experiments:
            domain = exp['run_data']['domain']
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(exp)

        for domain, exps in domains.items():
            out.write(f'## {domain.title()} Domain\n\n')
            out.write('| Run ID | Accuracy | Balance | Cycle Loss | Notes |\n')
            out.write('|--------|----------|---------|------------|-------|\n')

            for exp in exps:
                run_id = exp['run_data']['run_id']
                acc = exp['run_data']['best_val_accuracy']
                final = exp['run_data'].get('final_metrics', {})
                balance = final.get('class_balance_delta', 'N/A')
                cycle = final.get('cycle_loss', 'N/A')
                findings = exp['run_data'].get('findings', '')[:50]

                out.write(f'| {run_id} | {acc:.3f} | {balance:.3f} | '
                         f'{cycle:.3f} | {findings}... |\n')

            out.write('\n')

generate_report()
```

## Best Practices

### 1. Experiment Naming Convention

Use descriptive, timestamped run IDs:
```
{experiment_type}_{variant}_{date}
```

**Examples**:
- `baseline_single_pass_20251021`
- `dual_pass_equal_fusion_20251021`
- `planning_high_cycle_weight_20251023`

### 2. Always Include Findings

Every experiment should have a `findings` field summarizing results:
```python
"findings": "Severe class collapse (99.4% predict class 1). Baseline for dual-pass comparison."
```

### 3. Track Hyperparameter Provenance

Always log complete hyperparameters, even defaults:
```python
"hyperparameters": {
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "seed": 42,                    # CRITICAL for reproducibility
    "cycle_loss_weight": 0.01,
    "patience": 20,
    "min_delta": 0.001,
    "pool_ratio": 0.5
}
```

### 4. Log Architecture Details

For architectural experiments, include full configuration:
```python
"architecture": {
    "variant": "dual_pass_learned_fusion",
    "description": "Dual-pass with learned attention fusion",
    "num_levels": 3,
    "passes": 2,
    "fusion_weights": "learned_via_attention",
    "attention_heads": 8              # Variant-specific params
}
```

### 5. Capture Error States

For failed experiments, log comprehensive error info:
```python
"status": "failed",
"error_message": "CUDA out of memory at epoch 7, batch 42",
"final_metrics": null,
"last_successful_epoch": 6
```

### 6. Use Consistent Timestamps

Always use ISO 8601 format with UTC timezone:
```python
from datetime import datetime

timestamp = datetime.utcnow().isoformat()  # "2025-10-21T00:00:00.000000"
```

### 7. Validate Before Appending

Ensure JSON is valid before writing:
```python
import json

entry = {...}

# Validate
try:
    json.dumps(entry)
except (TypeError, ValueError) as e:
    print(f"Invalid JSON: {e}")
    # Fix entry before writing

# Write
with open('training_log.jsonl', 'a') as f:
    f.write(json.dumps(entry) + '\n')
```

## Integration with Modal Scripts

### Logging from Modal Experiments

```python
import modal
import json
from datetime import datetime

app = modal.App("nsm-experiment")
volume = modal.Volume.from_name("nsm-checkpoints")

@app.function(volumes={"/checkpoints": volume})
def train_and_log(config):
    # ... training code ...

    # Log experiment
    experiment_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "run_data": {
            "run_id": f"{config['experiment_type']}_{datetime.now().strftime('%Y%m%d')}",
            "domain": config['domain'],
            "status": "completed",
            "dataset_config": {...},
            "hyperparameters": config,
            "final_metrics": results,
            "training_time_seconds": elapsed_time,
            "experiment_type": config['experiment_type'],
            "findings": generate_findings(results)
        }
    }

    # Append to log
    with open('/checkpoints/training_log.jsonl', 'a') as f:
        f.write(json.dumps(experiment_entry) + '\n')

    volume.commit()
```

### Reading Logs Locally

```python
import modal

# Download logs
volume = modal.Volume.lookup("nsm-checkpoints")
volume.get_file("training_log.jsonl", "./local_training_log.jsonl")

# Analyze locally
import json
with open('local_training_log.jsonl', 'r') as f:
    experiments = [json.loads(line) for line in f]

print(f"Total experiments: {len(experiments)}")
```

## Success Criteria by Experiment Type

### Domain Exploration
```python
{
    "accuracy": ">0.55",              # Above random baseline
    "balance_delta": "<0.40",         # Reasonable class balance
    "cycle_loss": "<0.80",            # Decent reconstruction
    "domain_metrics": "varies"        # Domain-specific targets
}
```

### Dual-Pass Validation
```python
{
    "accuracy": ">0.50",              # Competitive with baseline
    "balance_delta": "<0.30",         # IMPROVED balance vs baseline
    "cycle_loss": "<1.0",             # Acceptable reconstruction
    "fusion_effectiveness": "show improvement over single-pass"
}
```

### Hyperparameter Search
```python
{
    "accuracy": ">best_baseline",    # Beat previous best
    "balance_delta": "<0.35",         # Maintain balance
    "cycle_loss": "depends on cycle_weight",
    "convergence": "monotonic decrease"
}
```

### Physics Validation (NSM-33)
```python
{
    "q_neural": ">1.0",               # Fusion quality (plasma analogy)
    "lawson_criterion": "achieved",   # Confinement quality
    "temperature_gradient": "stable", # Controlled evolution
    "beta_limit": "<1.0"              # Stability maintained
}
```

## Common Queries

### Get all experiments for a domain
```bash
cat experiments/training_log.jsonl | jq 'select(.run_data.domain == "planning")'
```

### Find experiments with high accuracy
```bash
cat experiments/training_log.jsonl | jq 'select(.run_data.best_val_accuracy > 0.6)'
```

### Count experiments by status
```bash
cat experiments/training_log.jsonl | jq '.run_data.status' | sort | uniq -c
```

### Get latest experiment
```bash
tail -n 1 experiments/training_log.jsonl | jq .
```

### Find failed experiments
```bash
cat experiments/training_log.jsonl | jq 'select(.run_data.status == "failed")'
```

## Troubleshooting

### Malformed JSON Lines

```python
# Validate all lines
import json

with open('training_log.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Line {i}: {e}")
```

### Duplicate Entries

```python
# Check for duplicate run_ids
import json

run_ids = set()
duplicates = []

with open('training_log.jsonl', 'r') as f:
    for line in f:
        exp = json.loads(line)
        run_id = exp['run_data']['run_id']

        if run_id in run_ids:
            duplicates.append(run_id)
        run_ids.add(run_id)

if duplicates:
    print(f"Duplicate run_ids: {duplicates}")
```

### Missing Required Fields

```python
# Validate schema
REQUIRED_FIELDS = ['timestamp', 'run_data']
RUN_DATA_FIELDS = ['run_id', 'domain', 'status']

with open('training_log.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        exp = json.loads(line)

        # Check top-level
        for field in REQUIRED_FIELDS:
            if field not in exp:
                print(f"Line {i}: Missing {field}")

        # Check run_data
        for field in RUN_DATA_FIELDS:
            if field not in exp.get('run_data', {}):
                print(f"Line {i}: Missing run_data.{field}")
```

## Migration Guide

### Converting Old Format to New Format

If you have experiments in a different format:

```python
import json
from datetime import datetime

def migrate_old_to_new(old_log_path, new_log_path):
    """Migrate old experiment format to training_log.jsonl format."""
    with open(old_log_path, 'r') as old, open(new_log_path, 'w') as new:
        for line in old:
            old_exp = json.loads(line)

            # Convert to new format
            new_exp = {
                "timestamp": old_exp.get('timestamp', datetime.utcnow().isoformat()),
                "run_data": {
                    "run_id": old_exp['experiment_id'],
                    "domain": old_exp['dataset'],
                    "status": "completed",
                    "dataset_config": {...},  # Extract from old_exp
                    "hyperparameters": {...},  # Extract from old_exp
                    "best_val_accuracy": old_exp['accuracy'],
                    # ... map other fields ...
                }
            }

            new.write(json.dumps(new_exp) + '\n')
```

## Contributing

When adding new experiment types:

1. **Document the schema** - Add to this guide
2. **Define success criteria** - What metrics matter?
3. **Provide examples** - Show typical log entries
4. **Update analysis recipes** - How to query this experiment type?
5. **Add validation** - Schema validation functions

## Resources

### Related Files
- **Modal Scripts**: `modal_*.py` - Experiment execution
- **Baselines**: `../baselines.jsonl` - Baseline results
- **Dataset Docs**: `../nsm/data/README.md` - Dataset specifications

### External Tools
- **jq**: Command-line JSON processor (https://stedolan.github.io/jq/)
- **Pandas**: For complex analysis (`pd.read_json(..., lines=True)`)
- **Plotly/Matplotlib**: For visualization

### NSM Project
- **Architecture**: `../CLAUDE.md` - NSM architecture guide
- **Phase 1.5 Results**: `../NSM-10-CROSS-DOMAIN-COMPARISON.md`
- **Linear Issues**: NSM-33, NSM-20 - Pilot studies and implementation

---

**Last Updated**: 2025-10-23

**Maintained By**: NSM Development Team

**Questions?** See `INDEX.md` for navigation guide
