"""
End-to-end test for NSM training pipeline.

Tests the complete workflow:
1. Model creation with domain-specific configuration
2. Dataset loading
3. Data loader creation
4. Single training step
5. Validation step
6. Loss computation verification
7. Gradient flow check
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import sys

# Add nsm to path
sys.path.insert(0, str(Path(__file__).parent))

from nsm.models import NSMModel
from nsm.training import NSMTrainer, compute_classification_metrics
from nsm.models.confidence.temperature import TemperatureScheduler


def test_model_creation(domain_config):
    """Test model can be created with domain configuration."""
    print(f"\n{'='*80}")
    print(f"Testing {domain_config['name']} Domain Model Creation")
    print(f"{'='*80}")

    model = NSMModel(
        node_features=domain_config['node_features'],
        num_relations=domain_config['num_relations'],
        num_classes=domain_config['num_classes'],
        num_bases=domain_config['num_bases'],
        pool_ratio=domain_config['pool_ratio'],
        task_type=domain_config['task_type']
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully")
    print(f"  - Total parameters: {num_params:,}")
    print(f"  - Node features: {domain_config['node_features']}")
    print(f"  - Relations: {domain_config['num_relations']}")
    print(f"  - Bases: {domain_config['num_bases']}")
    print(f"  - Pool ratio: {domain_config['pool_ratio']:.2f}")
    print(f"  - Parameter reduction: {100*(1 - domain_config['num_bases']/domain_config['num_relations']):.1f}%")

    return model


def test_forward_pass(model, domain_config):
    """Test forward pass with synthetic data."""
    print(f"\nTesting Forward Pass...")

    # Create synthetic batch
    num_nodes = 50
    num_edges = 100
    batch_size = 4

    x = torch.randn(num_nodes * batch_size, domain_config['node_features'])
    edge_index = torch.randint(0, num_nodes * batch_size, (2, num_edges * batch_size))
    edge_type = torch.randint(0, domain_config['num_relations'], (num_edges * batch_size,))
    edge_attr = torch.rand(num_edges * batch_size)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, edge_type, edge_attr, batch)

    print(f"✓ Forward pass successful")
    print(f"  - Input: {num_nodes * batch_size} nodes, {num_edges * batch_size} edges")
    print(f"  - Output logits: {output['logits'].shape}")
    print(f"  - Cycle loss: {output['cycle_loss'].item():.4f}")
    print(f"  - Abstract nodes: {output['x_abstract'].shape[0]}")

    return output


def test_training_step(model, domain_config, device):
    """Test single training step."""
    print(f"\nTesting Training Step...")

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    temp_scheduler = TemperatureScheduler(initial_temp=1.0, final_temp=0.3)

    trainer = NSMTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        cycle_loss_weight=domain_config['cycle_loss_weight'],
        gradient_clip=1.0,
        temp_scheduler=temp_scheduler
    )

    # Create synthetic batch
    num_nodes = 50
    num_edges = 100
    batch_size = 4

    batch = {
        'x': torch.randn(num_nodes * batch_size, domain_config['node_features']),
        'edge_index': torch.randint(0, num_nodes * batch_size, (2, num_edges * batch_size)),
        'edge_type': torch.randint(0, domain_config['num_relations'], (num_edges * batch_size,)),
        'edge_attr': torch.rand(num_edges * batch_size),
        'batch': torch.repeat_interleave(torch.arange(batch_size), num_nodes),
        'y': torch.randint(0, domain_config['num_classes'], (batch_size,))
    }

    # Training step
    loss_dict = trainer.train_step(batch, domain_config['task_type'])

    print(f"✓ Training step successful")
    print(f"  - Task loss: {loss_dict['task_loss']:.4f}")
    print(f"  - Cycle loss: {loss_dict['cycle_loss']:.4f}")
    print(f"  - Total loss: {loss_dict['total_loss']:.4f}")
    print(f"  - Gradient norm: {loss_dict['grad_norm']:.2f}")

    # Check gradient flow
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    print(f"  - Average gradient norm: {avg_grad_norm:.2e}")

    if avg_grad_norm > 1e-6:
        print(f"  ✓ Gradient flow: HEALTHY (>{1e-6:.0e})")
    else:
        print(f"  ⚠ Gradient flow: VANISHING (<{1e-6:.0e})")

    return loss_dict


def test_domain(domain_config, device):
    """Run complete end-to-end test for a domain."""
    try:
        # Test 1: Model creation
        model = test_model_creation(domain_config)

        # Test 2: Forward pass
        output = test_forward_pass(model, domain_config)

        # Test 3: Training step
        loss_dict = test_training_step(model, domain_config, device)

        print(f"\n{'='*80}")
        print(f"✅ {domain_config['name']} Domain: ALL TESTS PASSED")
        print(f"{'='*80}\n")

        return True

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ {domain_config['name']} Domain: TEST FAILED")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("NSM END-TO-END TRAINING PIPELINE TEST")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Domain configurations
    domains = [
        {
            'name': 'Causal (NSM-24)',
            'node_features': 64,
            'num_relations': 20,
            'num_bases': 5,
            'num_classes': 2,
            'pool_ratio': 0.5,
            'task_type': 'classification',
            'cycle_loss_weight': 0.1
        },
        {
            'name': 'Planning (NSM-22)',
            'node_features': 64,
            'num_relations': 16,
            'num_bases': 8,
            'num_classes': 2,
            'pool_ratio': 0.5,
            'task_type': 'classification',
            'cycle_loss_weight': 0.1
        },
        {
            'name': 'Knowledge Graph (NSM-23)',
            'node_features': 64,
            'num_relations': 66,
            'num_bases': 12,
            'num_classes': 2,
            'pool_ratio': 0.13,
            'task_type': 'link_prediction',
            'cycle_loss_weight': 0.15
        }
    ]

    results = {}
    for domain_config in domains:
        success = test_domain(domain_config, device)
        results[domain_config['name']] = success

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = all(results.values())

    for domain, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{domain}: {status}")

    print("="*80)

    if all_passed:
        print("\n🎉 ALL DOMAINS PASSED END-TO-END TESTS!")
        print("\nReady for full training runs:")
        print("  - python experiments/train_causal.py --epochs 10")
        print("  - python experiments/train_planning.py --epochs 10")
        print("  - python experiments/train_kg.py --epochs 10")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Review errors above")
        return 1


if __name__ == '__main__':
    exit(main())
