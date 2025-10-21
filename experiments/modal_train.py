"""
Modal.com GPU training for NSM Phase 1.5 (3-level hierarchy).

Provides GPU-accelerated training across all three domains:
- Planning: Goal-action hierarchy
- Knowledge Graph: Link prediction with 66 relations
- Causal: Counterfactual reasoning

Features:
- A100 GPU (40GB VRAM)
- Persistent checkpoints via Modal volumes
- Auto-retry on preemption
- Parallel domain training
- Cost-effective (pay only for GPU time)

Setup:
    modal token set --token-id <id> --token-secret <secret>

Usage:
    # Train all domains in parallel on GPU
    modal run experiments/modal_train.py::train_all_domains

    # Train individual domain
    modal run experiments/modal_train.py::train_planning
    modal run experiments/modal_train.py::train_causal
    modal run experiments/modal_train.py::train_kg

    # Quick validation (10 epochs)
    modal run experiments/modal_train.py::validate_3level
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("nsm-phase1.5")

# Get NSM project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Use official PyTorch image with CUDA 11.8 as base
image = (
    modal.Image.from_registry("pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime", add_python="3.10")
    .run_commands(
        # Install PyG extensions (torch-scatter and torch-sparse are the critical ones)
        "pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html",
        # Install PyG itself
        "pip install torch-geometric==2.4.0"
    )
    .pip_install("numpy", "scipy", "networkx", "matplotlib", "tensorboard")
    # Copy entire project root into /root/NSM so structure is preserved
    # This allows: sys.path.insert(0, "/root/NSM") → import nsm.data.*
    # Note: copy=True required since this is the final step (Modal will bake files into image)
    .add_local_dir(PROJECT_ROOT, "/root/NSM", copy=True, ignore=["*.pyc", "__pycache__", ".git", "logs", "checkpoints", "data", ".pytest_cache"])
)

# Persistent volume for checkpoints
volume = modal.Volume.from_name("nsm-checkpoints", create_if_missing=True)

CHECKPOINT_DIR = "/checkpoints"
DATA_DIR = "/data"


@app.function(
    image=image,
    gpu="A100-40GB",  # Strict 40GB to avoid surprise upgrades to 80GB
    timeout=7200,  # 2 hours for 100-epoch runs
    volumes={CHECKPOINT_DIR: volume},
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=60.0),
    cpu=4.0,  # Reserve sufficient CPU for DataLoader workers
)
def train_planning(epochs=100, batch_size=64, num_problems=2858, lr=1e-4, cycle_weight=0.01, seed=42,
                   use_amp=True, checkpoint_freq=10):
    """Train NSM on Planning domain with A100 GPU."""
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
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🚀 CUDA Version: {torch.version.cuda}")
    print(f"🚀 Batch size: {batch_size}, AMP: {use_amp}")

    checkpoint_path = Path(CHECKPOINT_DIR) / "planning"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Enable TF32 for better A100 performance (20% speedup on matmul/convs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = PlanningTripleDataset(root=f"{DATA_DIR}/planning", split='train', num_problems=num_problems, seed=seed)
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                             num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                           num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    model = NSMModel(node_features=64, num_relations=16, num_classes=2, num_bases=8,
                     pool_ratio=0.5, task_type='classification', num_levels=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    temp_scheduler = TemperatureScheduler(initial_temp=1.0, final_temp=0.3, decay_rate=0.9999, warmup_epochs=10)

    trainer = NSMTrainer(model=model, optimizer=optimizer, device=device, cycle_loss_weight=cycle_weight,
                        gradient_clip=1.0, temp_scheduler=temp_scheduler, checkpoint_dir=str(checkpoint_path),
                        log_interval=10, use_wandb=False, use_tensorboard=False)

    start_time = datetime.now()
    history = trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=epochs,
                           task_type='classification',
                           compute_metrics=lambda p, l, t: compute_classification_metrics(p, l, t),
                           early_stopping_patience=20, save_best_only=True)

    training_time = (datetime.now() - start_time).total_seconds()

    results = {
        'domain': 'planning', 'num_levels': 3, 'epochs': epochs, 'training_time_seconds': training_time,
        'final_train_loss': history['train'][-1]['total_loss'],
        'final_val_loss': history['val'][-1]['total_loss'],
        'best_val_loss': trainer.best_val_loss, 'final_metrics': history['val'][-1]
    }

    with open(checkpoint_path / 'modal_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    volume.commit()
    print(f"\n✅ Planning complete! Best loss: {trainer.best_val_loss:.4f}, Time: {training_time/60:.2f}min")
    return results


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=7200,  # 2 hours for 100-epoch runs
    volumes={CHECKPOINT_DIR: volume},
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=60.0),
    cpu=4.0
)
def train_causal(epochs=100, batch_size=64, num_scenarios=1000, lr=1e-4, cycle_weight=0.01, seed=42,
                 checkpoint_freq=10):
    """Train NSM on Causal domain with A100 GPU."""
    import torch, json
    from datetime import datetime
    import sys
    sys.path.insert(0, "/root/NSM")

    from nsm.data.causal_dataset import CausalTripleDataset
    from nsm.models import NSMModel
    from nsm.training import NSMTrainer, compute_classification_metrics
    from nsm.models.confidence.temperature import TemperatureScheduler
    from torch.utils.data import DataLoader, random_split
    from torch_geometric.data import Batch

    device = torch.device('cuda')
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🚀 CUDA Version: {torch.version.cuda}")
    print(f"🚀 Batch size: {batch_size}")

    checkpoint_path = Path(CHECKPOINT_DIR) / "causal"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Enable TF32 for better A100 performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = CausalTripleDataset(root=f"{DATA_DIR}/causal", split='train',
                                  num_scenarios=num_scenarios, num_treatments=8, num_symptoms=8, seed=seed)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    def collate_fn(batch_list):
        data_list = [item[0] for item in batch_list]
        labels = torch.tensor([item[1] for item in batch_list])
        batched_data = Batch.from_data_list(data_list)
        return {'x': batched_data.x, 'edge_index': batched_data.edge_index, 'edge_type': batched_data.edge_type,
                'edge_attr': getattr(batched_data, 'edge_attr', None), 'batch': batched_data.batch, 'y': labels}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                             num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                           num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    model = NSMModel(node_features=64, num_relations=20, num_classes=2, num_bases=5,
                     pool_ratio=0.5, task_type='classification', num_levels=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    temp_scheduler = TemperatureScheduler(initial_temp=1.0, final_temp=0.3, decay_rate=0.9999, warmup_epochs=10)

    trainer = NSMTrainer(model=model, optimizer=optimizer, device=device, cycle_loss_weight=cycle_weight,
                        gradient_clip=1.0, temp_scheduler=temp_scheduler, checkpoint_dir=str(checkpoint_path),
                        log_interval=10, use_wandb=False, use_tensorboard=False)

    start_time = datetime.now()
    history = trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=epochs, task_type='classification',
                           compute_metrics=lambda p, l, t: compute_classification_metrics(p, l, t),
                           early_stopping_patience=20, save_best_only=True)

    training_time = (datetime.now() - start_time).total_seconds()

    results = {
        'domain': 'causal', 'num_levels': 3, 'epochs': epochs, 'training_time_seconds': training_time,
        'final_train_loss': history['train'][-1]['total_loss'], 'final_val_loss': history['val'][-1]['total_loss'],
        'best_val_loss': trainer.best_val_loss, 'final_metrics': history['val'][-1]
    }

    with open(checkpoint_path / 'modal_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    volume.commit()
    print(f"\n✅ Causal complete! Best loss: {trainer.best_val_loss:.4f}, Time: {training_time/60:.2f}min")
    return results


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=7200,  # 2 hours for 100-epoch runs
    volumes={CHECKPOINT_DIR: volume},
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=60.0),
    cpu=4.0
)
def train_kg(epochs=100, batch_size=64, num_entities=200, num_triples=2500, lr=1e-4, cycle_weight=0.05, seed=42,
             checkpoint_freq=10):
    """Train NSM on KG domain with A100 GPU."""
    import torch, json
    from datetime import datetime
    import sys
    sys.path.insert(0, "/root/NSM")

    from nsm.data.knowledge_graph_dataset import KnowledgeGraphTripleDataset
    from nsm.models import NSMModel
    from nsm.training import NSMTrainer, compute_classification_metrics
    from nsm.models.confidence.temperature import TemperatureScheduler
    from torch.utils.data import DataLoader, random_split
    from torch_geometric.data import Batch

    device = torch.device('cuda')
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🚀 CUDA Version: {torch.version.cuda}")
    print(f"🚀 Batch size: {batch_size}")

    checkpoint_path = Path(CHECKPOINT_DIR) / "kg"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Enable TF32 for better A100 performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = KnowledgeGraphTripleDataset(root=f"{DATA_DIR}/kg", split='train',
                                         num_entities=num_entities, num_triples=num_triples, seed=seed)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    def collate_fn(batch_list):
        data_list = [item[0] for item in batch_list]
        labels = torch.tensor([item[1].item() for item in batch_list], dtype=torch.long)
        batched_data = Batch.from_data_list(data_list)
        return {'x': batched_data.x, 'edge_index': batched_data.edge_index, 'edge_type': batched_data.edge_type,
                'edge_attr': getattr(batched_data, 'edge_attr', None), 'batch': batched_data.batch, 'y': labels}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                             num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                           num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    model = NSMModel(node_features=64, num_relations=66, num_classes=2, num_bases=12,
                     pool_ratio=0.13, task_type='link_prediction', num_levels=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    temp_scheduler = TemperatureScheduler(initial_temp=1.0, final_temp=0.3, decay_rate=0.9999, warmup_epochs=10)

    trainer = NSMTrainer(model=model, optimizer=optimizer, device=device, cycle_loss_weight=cycle_weight,
                        gradient_clip=1.0, temp_scheduler=temp_scheduler, checkpoint_dir=str(checkpoint_path),
                        log_interval=10, use_wandb=False, use_tensorboard=False)

    start_time = datetime.now()
    history = trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=epochs,
                           task_type='link_prediction',
                           compute_metrics=lambda p, l, t: compute_classification_metrics(p, l, t),
                           early_stopping_patience=20, save_best_only=True)

    training_time = (datetime.now() - start_time).total_seconds()

    results = {
        'domain': 'kg', 'num_levels': 3, 'epochs': epochs, 'training_time_seconds': training_time,
        'final_train_loss': history['train'][-1]['total_loss'], 'final_val_loss': history['val'][-1]['total_loss'],
        'best_val_loss': trainer.best_val_loss, 'final_metrics': history['val'][-1]
    }

    with open(checkpoint_path / 'modal_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    volume.commit()
    print(f"\n✅ KG complete! Best loss: {trainer.best_val_loss:.4f}, Time: {training_time/60:.2f}min")
    return results


@app.local_entrypoint()
def train_all_domains():
    """Launch parallel GPU training on all three domains."""
    print("🚀 Launching parallel GPU training (3-level NSM)...\n")

    planning_job = train_planning.spawn()
    causal_job = train_causal.spawn()
    kg_job = train_kg.spawn()

    print("⏳ Waiting for jobs to complete...\n")
    planning_results = planning_job.get()
    causal_results = causal_job.get()
    kg_results = kg_job.get()

    print("\n" + "="*80)
    print("🎉 All domains complete!")
    print("="*80)

    for results in [planning_results, causal_results, kg_results]:
        print(f"\n{results['domain'].upper()}:")
        print(f"  Best val loss: {results['best_val_loss']:.4f}")
        print(f"  Training time: {results['training_time_seconds']/60:.2f} min")
        print(f"  Final accuracy: {results['final_metrics'].get('accuracy', 'N/A'):.2%}")

    return {'planning': planning_results, 'causal': causal_results, 'kg': kg_results}


@app.local_entrypoint()
def validate_3level():
    """Quick 10-epoch validation of 3-level architecture with independent error handling."""
    print("🧪 Running 3-level validation (10 epochs)...\n")

    # Launch all jobs (non-blocking) - use smaller batch for validation
    jobs = {
        'planning': train_planning.spawn(epochs=10, num_problems=500, batch_size=32, use_amp=False, checkpoint_freq=5),
        'causal': train_causal.spawn(epochs=10, num_scenarios=500, batch_size=32, checkpoint_freq=5),
        'kg': train_kg.spawn(epochs=10, num_entities=100, num_triples=500, batch_size=32, checkpoint_freq=5)
    }

    print("⏳ Waiting for validation jobs...\n")

    # Collect results with per-job error handling
    results = {}
    for domain, job in jobs.items():
        try:
            result = job.get(timeout=3600)  # Per-job timeout
            results[domain] = {'status': 'success', 'data': result}
        except Exception as e:
            results[domain] = {'status': 'failed', 'error': str(e)}
            print(f"❌ {domain} failed: {e}\n")

    print("\n" + "="*80)
    print("✅ Validation Complete!")
    print("="*80)

    for domain, result_data in results.items():
        if result_data['status'] == 'failed':
            print(f"\n{domain.upper()}: ❌ FAILED")
            print(f"  Error: {result_data['error']}")
            continue

        res = result_data['data']
        acc = res['final_metrics'].get('accuracy', 0.0)
        cycle = res['final_metrics'].get('cycle_loss', 0.0)
        acc_0 = res['final_metrics'].get('accuracy_class_0', 0.0)
        acc_1 = res['final_metrics'].get('accuracy_class_1', 0.0)

        print(f"\n{domain.upper()}: ✅ SUCCESS")
        print(f"  Accuracy: {acc:.2%}")
        print(f"  Cycle loss: {cycle:.4f}")

        if acc_0 == 0.0 or acc_1 == 0.0:
            print(f"  ⚠️  CLASS COLLAPSE! (C0: {acc_0:.2%}, C1: {acc_1:.2%})")
        else:
            print(f"  ✅ No collapse (C0: {acc_0:.2%}, C1: {acc_1:.2%})")

    return results
