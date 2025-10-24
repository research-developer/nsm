"""
Production training entrypoint for NSM Phase 1.5.

Run with:
    modal run experiments/modal_train_production.py
"""

import modal
import sys
from pathlib import Path

# Import the main app from modal_train
sys.path.insert(0, str(Path(__file__).parent))
from modal_train import app, train_planning, train_causal, train_kg


@app.local_entrypoint()
def train_production():
    """
    Full 100-epoch production training with optimizations.

    Optimizations applied:
    - Larger batch size (64) for A100 40GB VRAM
    - TF32 enabled for 20% matmul speedup
    - DataLoader pin_memory and prefetch for GPU feeding
    - Checkpoint every 10 epochs
    - Early stopping after 20 epochs without improvement
    - 2-hour timeout per domain
    """
    print("🚀 Starting production training (100 epochs, optimized for A100)...\n")
    print("Optimizations:")
    print("  - Batch size: 64 (vs 32 baseline)")
    print("  - TF32: Enabled (20% speedup on matmul)")
    print("  - DataLoader: pin_memory, persistent_workers, prefetch_factor=2")
    print("  - Checkpoints: Every 10 epochs")
    print("  - Early stopping: 20 epochs patience")
    print("  - Timeout: 2 hours per domain\n")

    # Launch all jobs with production settings
    jobs = {
        'planning': train_planning.spawn(
            epochs=100, num_problems=2858, batch_size=64,
            lr=1e-4, cycle_weight=0.01, use_amp=False, checkpoint_freq=10
        ),
        'causal': train_causal.spawn(
            epochs=100, num_scenarios=1000, batch_size=64,
            lr=1e-4, cycle_weight=0.01, checkpoint_freq=10
        ),
        'kg': train_kg.spawn(
            epochs=100, num_entities=200, num_triples=2500, batch_size=64,
            lr=1e-4, cycle_weight=0.05, checkpoint_freq=10
        )
    }

    print("⏳ Training in progress (check Modal dashboard for live logs)...\n")
    print("Dashboard: https://modal.com/apps/research-developer/main\n")

    # Collect results with error handling
    results = {}
    for domain, job in jobs.items():
        try:
            result = job.get()
            results[domain] = {'status': 'success', 'data': result}
            print(f"✅ {domain.upper()} complete!")
        except Exception as e:
            results[domain] = {'status': 'failed', 'error': str(e)}
            print(f"❌ {domain.upper()} failed: {e}")

    print("\n" + "="*80)
    print("🎉 Production Training Complete!")
    print("="*80)

    # Summary report
    for domain, result_data in results.items():
        if result_data['status'] == 'failed':
            print(f"\n{domain.upper()}: ❌ FAILED")
            print(f"  Error: {result_data['error']}")
            continue

        res = result_data['data']
        acc = res['final_metrics'].get('accuracy', 0.0)
        cycle = res['final_metrics'].get('cycle_loss', 0.0)
        time_min = res['training_time_seconds'] / 60
        acc_0 = res['final_metrics'].get('accuracy_class_0', 0.0)
        acc_1 = res['final_metrics'].get('accuracy_class_1', 0.0)

        print(f"\n{domain.upper()}: ✅ SUCCESS")
        print(f"  Final accuracy: {acc:.2%}")
        print(f"  Best val loss: {res['best_val_loss']:.4f}")
        print(f"  Cycle loss: {cycle:.4f}")
        print(f"  Training time: {time_min:.1f} min")
        print(f"  Epochs completed: {res['epochs']}")

        if acc_0 > 0 and acc_1 > 0:
            print(f"  ✅ No class collapse (C0: {acc_0:.2%}, C1: {acc_1:.2%})")
        else:
            print(f"  ⚠️  CLASS COLLAPSE (C0: {acc_0:.2%}, C1: {acc_1:.2%})")

    # Cost estimate (rough)
    total_time_hours = sum(
        r['data']['training_time_seconds'] / 3600
        for r in results.values()
        if r['status'] == 'success'
    )
    cost_estimate = total_time_hours * 1.10  # A100-40GB is ~$1.10/hr on Modal

    print(f"\n📊 Training Summary:")
    print(f"  Total GPU time: {total_time_hours:.2f} hours")
    print(f"  Estimated cost: ${cost_estimate:.2f}")

    return results
