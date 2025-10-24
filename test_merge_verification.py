"""
Quick verification test for merged dual-pass and single-pass modes.
"""
import torch
import torch.nn.functional as F
from nsm.models.hierarchical import NSMModel

def test_single_pass_mode():
    """Test default single-pass mode (3-level)."""
    print("Testing single-pass mode (3-level)...")

    model = NSMModel(
        node_features=64,
        num_relations=4,
        num_classes=2,
        task_type='classification',
        num_levels=3,
        use_dual_pass=False  # Single-pass mode
    )

    # Create dummy data
    num_nodes = 20
    x = torch.randn(num_nodes, 64)
    edge_index = torch.randint(0, num_nodes, (2, 40))
    edge_type = torch.randint(0, 4, (40,))
    edge_attr = torch.randn(40)  # 1D confidence weights
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Forward pass
    output = model(x, edge_index, edge_type, edge_attr, batch)

    # Verify output
    assert 'logits' in output, "Missing logits"
    assert 'cycle_loss' in output, "Missing cycle_loss"
    assert 'x_l2' in output, "Missing x_l2"
    assert 'x_l3' in output, "Missing x_l3"
    assert output['logits'].shape == (1, 2), f"Unexpected logits shape: {output['logits'].shape}"

    # Verify no dual-pass outputs
    assert 'logits_abstract' not in output, "Should not have dual-pass outputs in single-pass mode"

    print(f"✓ Single-pass mode works!")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Cycle loss: {output['cycle_loss'].item():.4f}")
    return True

def test_dual_pass_mode():
    """Test dual-pass mode with fusion."""
    print("\nTesting dual-pass mode (3-level with fusion)...")

    model = NSMModel(
        node_features=64,
        num_relations=4,
        num_classes=2,
        task_type='classification',
        num_levels=3,
        use_dual_pass=True,  # Dual-pass mode
        fusion_mode='equal'
    )

    # Create dummy data
    num_nodes = 20
    x = torch.randn(num_nodes, 64)
    edge_index = torch.randint(0, num_nodes, (2, 40))
    edge_type = torch.randint(0, 4, (40,))
    edge_attr = torch.randn(40)  # 1D confidence weights
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Forward pass
    output = model(x, edge_index, edge_type, edge_attr, batch)

    # Verify output
    assert 'logits' in output, "Missing logits"
    assert 'logits_abstract' in output, "Missing logits_abstract"
    assert 'logits_concrete' in output, "Missing logits_concrete"
    assert 'fusion_weights' in output, "Missing fusion_weights"
    assert 'cycle_loss' in output, "Missing cycle_loss"
    assert output['logits'].shape == (1, 2), f"Unexpected logits shape: {output['logits'].shape}"

    print(f"✓ Dual-pass mode works!")
    print(f"  Fused logits shape: {output['logits'].shape}")
    print(f"  Abstract logits shape: {output['logits_abstract'].shape}")
    print(f"  Concrete logits shape: {output['logits_concrete'].shape}")
    print(f"  Fusion weights: {output['fusion_weights']}")
    print(f"  Cycle loss: {output['cycle_loss'].item():.4f}")
    return True

def test_2level_backward_compat():
    """Test 2-level mode for backward compatibility."""
    print("\nTesting 2-level mode (backward compatibility)...")

    model = NSMModel(
        node_features=64,
        num_relations=4,
        num_classes=2,
        task_type='classification',
        num_levels=2  # 2-level mode
    )

    # Create dummy data
    num_nodes = 20
    x = torch.randn(num_nodes, 64)
    edge_index = torch.randint(0, num_nodes, (2, 40))
    edge_type = torch.randint(0, 4, (40,))
    edge_attr = torch.randn(40)  # 1D confidence weights
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Forward pass
    output = model(x, edge_index, edge_type, edge_attr, batch)

    # Verify output
    assert 'logits' in output, "Missing logits"
    assert 'cycle_loss' in output, "Missing cycle_loss"
    assert output['logits'].shape == (1, 2), f"Unexpected logits shape: {output['logits'].shape}"

    print(f"✓ 2-level mode works!")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Cycle loss: {output['cycle_loss'].item():.4f}")
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("Merge Verification Test Suite")
    print("=" * 60)

    try:
        test_single_pass_mode()
        test_dual_pass_mode()
        test_2level_backward_compat()

        print("\n" + "=" * 60)
        print("✓ All tests passed! Merge successful.")
        print("=" * 60)
        print("\nBoth single-pass and dual-pass modes are working correctly.")
        print("Ready to push and create PR!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
