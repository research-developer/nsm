"""
Modal.com Quick Validation: Dual-Pass vs Single-Pass Architecture

Tests 4 variants side-by-side on Planning domain (10 epochs each):
1. Baseline: Single-pass (current architecture)
2. Dual-pass with equal fusion (α=β=0.5)
3. Dual-pass with learned fusion (attention)
4. Dual-pass without cycle loss (cycle_weight=0)

Cost: ~$2 total
Time: ~10 minutes
"""

import modal
from pathlib import Path

app = modal.App("nsm-dual-pass-validation")
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Use same image as main training
image = (
    modal.Image.from_registry("pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime", add_python="3.10")
    .run_commands(
        "pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html",
        "pip install torch-geometric==2.4.0"
    )
    .pip_install("numpy", "scipy", "networkx", "matplotlib", "tensorboard")
    .add_local_dir(PROJECT_ROOT, "/root/NSM", copy=True, ignore=["*.pyc", "__pycache__", ".git", "logs", "checkpoints", "data", ".pytest_cache"])
)

volume = modal.Volume.from_name("nsm-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = "/checkpoints"
DATA_DIR = "/data"


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,  # 30 minutes max per variant
    volumes={CHECKPOINT_DIR: volume},
    cpu=4.0
)
def train_variant(
    variant_name: str,
    use_dual_pass: bool,
    fusion_mode: str = 'equal',
    cycle_weight: float = 0.01,
    epochs: int = 10,
    batch_size: int = 64,
    num_problems: int = 2858,
    seed: int = 42
):
    """Train a single variant of NSM architecture.

    Args:
        variant_name: Human-readable name for logging
        use_dual_pass: Enable dual-pass mode
        fusion_mode: 'equal', 'learned', 'abstract_only', 'concrete_only'
        cycle_weight: Weight for cycle consistency loss
        epochs: Number of training epochs
        batch_size: Batch size
        num_problems: Dataset size
        seed: Random seed
    """
    import torch
    import json
    from datetime import datetime
    import sys
    sys.path.insert(0, "/root/NSM")

    from nsm.data.planning_dataset import PlanningTripleDataset
    from nsm.models import NSMModel
    from nsm.training import NSMTrainer, compute_classification_metrics
    from nsm.models.confidence.temperature import TemperatureScheduler
    from torch.utils.data import DataLoader, random_split
    from torch_geometric.data import Batch

    device = torch.device('cuda')
    print(f"\n{'='*80}")
    print(f"🧪 VARIANT: {variant_name}")
    print(f"{'='*80}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config:")
    print(f"  use_dual_pass: {use_dual_pass}")
    print(f"  fusion_mode: {fusion_mode}")
    print(f"  cycle_weight: {cycle_weight}")
    print(f"  batch_size: {batch_size}")
    print(f"\n")

    checkpoint_path = Path(CHECKPOINT_DIR) / f"dual_pass_validation/{variant_name}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Enable TF32 for better A100 performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Dataset
    dataset = PlanningTripleDataset(root=f"{DATA_DIR}/planning", split='train',
                                   num_problems=num_problems, seed=seed)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    def collate_fn(batch_list):
        data_list = [item[0] for item in batch_list]
        labels = torch.tensor([item[1] for item in batch_list])
        batched_data = Batch.from_data_list(data_list)
        return {
            'x': batched_data.x, 'edge_index': batched_data.edge_index,
            'edge_type': batched_data.edge_type,
            'edge_attr': getattr(batched_data, 'edge_attr', None),
            'batch': batched_data.batch, 'y': labels
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Model with variant-specific configuration
    model = NSMModel(
        node_features=64,
        num_relations=16,
        num_classes=2,
        num_bases=8,
        pool_ratio=0.5,
        task_type='classification',
        num_levels=3,
        use_dual_pass=use_dual_pass,
        fusion_mode=fusion_mode
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if use_dual_pass:
        print(f"Fusion mode: {fusion_mode}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    temp_scheduler = TemperatureScheduler(initial_temp=1.0, final_temp=0.3,
                                         decay_rate=0.9999, warmup_epochs=10)

    trainer = NSMTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        cycle_loss_weight=cycle_weight,
        gradient_clip=1.0,
        temp_scheduler=temp_scheduler,
        checkpoint_dir=str(checkpoint_path),
        log_interval=10,
        use_wandb=False,
        use_tensorboard=False
    )

    start_time = datetime.now()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        task_type='classification',
        compute_metrics=lambda p, l, t: compute_classification_metrics(p, l, t),
        early_stopping_patience=20,
        save_best_only=True
    )

    training_time = (datetime.now() - start_time).total_seconds()

    # Collect final metrics
    final_val_metrics = history['val'][-1]

    results = {
        'variant_name': variant_name,
        'config': {
            'use_dual_pass': use_dual_pass,
            'fusion_mode': fusion_mode,
            'cycle_weight': cycle_weight,
            'batch_size': batch_size
        },
        'epochs': epochs,
        'training_time_seconds': training_time,
        'final_train_loss': history['train'][-1]['total_loss'],
        'final_val_loss': final_val_metrics['total_loss'],
        'best_val_loss': trainer.best_val_loss,
        'final_metrics': final_val_metrics,
        # Key metrics for comparison
        'accuracy': final_val_metrics.get('accuracy', 0.0),
        'accuracy_class_0': final_val_metrics.get('accuracy_class_0', 0.0),
        'accuracy_class_1': final_val_metrics.get('accuracy_class_1', 0.0),
        'class_balance_delta': abs(
            final_val_metrics.get('accuracy_class_0', 0.0) -
            final_val_metrics.get('accuracy_class_1', 0.0)
        ),
        'cycle_loss': final_val_metrics.get('cycle_loss', 0.0)
    }

    # Save results
    with open(checkpoint_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"✅ {variant_name} COMPLETE")
    print(f"{'='*80}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"  Class 0: {results['accuracy_class_0']:.2%}")
    print(f"  Class 1: {results['accuracy_class_1']:.2%}")
    print(f"  Balance Δ: {results['class_balance_delta']:.2%}")
    print(f"Cycle Loss: {results['cycle_loss']:.4f}")
    print(f"Training Time: {training_time:.1f}s")
    print(f"{'='*80}\n")

    volume.commit()
    return results


@app.local_entrypoint()
def validate_all_variants():
    """Run all 4 variants in parallel and compare results."""
    import json
    from datetime import datetime

    print(f"\n{'='*80}")
    print("🚀 DUAL-PASS ARCHITECTURE VALIDATION")
    print(f"{'='*80}")
    print(f"Testing 4 variants on Planning domain (10 epochs each)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # Launch all variants in parallel
    jobs = {
        'baseline': train_variant.spawn(
            variant_name="baseline_single_pass",
            use_dual_pass=False,
            fusion_mode='equal',  # Ignored for single-pass
            cycle_weight=0.01
        ),
        'dual_equal': train_variant.spawn(
            variant_name="dual_pass_equal_fusion",
            use_dual_pass=True,
            fusion_mode='equal',
            cycle_weight=0.01
        ),
        'dual_learned': train_variant.spawn(
            variant_name="dual_pass_learned_fusion",
            use_dual_pass=True,
            fusion_mode='learned',
            cycle_weight=0.01
        ),
        'dual_no_cycle': train_variant.spawn(
            variant_name="dual_pass_no_cycle",
            use_dual_pass=True,
            fusion_mode='equal',
            cycle_weight=0.0  # Remove cycle loss constraint
        )
    }

    print("⏳ Waiting for all variants to complete...\n")

    # Collect results
    results = {}
    for variant_key, job in jobs.items():
        try:
            result = job.get(timeout=1800)
            results[variant_key] = result
            print(f"✅ {variant_key}: Completed")
        except Exception as e:
            results[variant_key] = {'status': 'failed', 'error': str(e)}
            print(f"❌ {variant_key}: Failed - {e}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("📊 RESULTS COMPARISON")
    print(f"{'='*80}\n")

    print(f"{'Variant':<30} {'Accuracy':<12} {'Class Δ':<12} {'Cycle Loss':<12} {'Time':<8}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")

    for variant_key, result in results.items():
        if 'status' in result and result['status'] == 'failed':
            print(f"{variant_key:<30} {'FAILED':<12}")
        else:
            print(f"{result['variant_name']:<30} "
                  f"{result['accuracy']:>10.2%}  "
                  f"{result['class_balance_delta']:>10.2%}  "
                  f"{result['cycle_loss']:>10.4f}  "
                  f"{result['training_time_seconds']:>6.0f}s")

    # Determine winner
    print(f"\n{'='*80}")
    print("🏆 WINNER ANALYSIS")
    print(f"{'='*80}\n")

    valid_results = {k: v for k, v in results.items() if 'accuracy' in v}

    if valid_results:
        # Best by accuracy
        best_acc = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"Best Accuracy: {best_acc[0]} ({best_acc[1]['accuracy']:.2%})")

        # Best by class balance
        best_balance = min(valid_results.items(), key=lambda x: x[1]['class_balance_delta'])
        print(f"Best Balance: {best_balance[0]} (Δ={best_balance[1]['class_balance_delta']:.2%})")

        # Check for improvement over baseline
        if 'baseline' in valid_results:
            baseline_acc = valid_results['baseline']['accuracy']
            baseline_delta = valid_results['baseline']['class_balance_delta']

            print(f"\nComparison to Baseline:")
            for variant_key, result in valid_results.items():
                if variant_key != 'baseline':
                    acc_improvement = result['accuracy'] - baseline_acc
                    balance_improvement = baseline_delta - result['class_balance_delta']

                    print(f"  {variant_key}:")
                    print(f"    Accuracy: {acc_improvement:+.2%}")
                    print(f"    Balance: {balance_improvement:+.2%} (lower is better)")

    # Save aggregated results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'variants': results,
        'winner': {
            'best_accuracy': best_acc[0] if valid_results else None,
            'best_balance': best_balance[0] if valid_results else None
        }
    }

    summary_path = Path('/tmp/dual_pass_validation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📄 Summary saved to: {summary_path}")
    print(f"\n{'='*80}\n")

    return summary
