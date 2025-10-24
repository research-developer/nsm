"""
Quick verification test for merged dual-pass and single-pass modes.

Note: As of Phase 1.5, dual-pass mode is the default (use_dual_pass=True).
Single-pass mode is now opt-out via use_dual_pass=False.
"""
import torch
import torch.nn.functional as F
from nsm.models.hierarchical import NSMModel

def test_dual_pass_default():
    """Test dual-pass mode as default (3-level)."""
    print("Testing dual-pass mode (default behavior)...")

    model = NSMModel(
        node_features=64,
        num_relations=4,
        num_classes=2,
        task_type='classification',
        num_levels=3
        # use_dual_pass defaults to True
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

    # Verify dual-pass outputs
    assert 'logits' in output, "Missing logits"
    assert 'logits_abstract' in output, "Missing logits_abstract (dual-pass should be default)"
    assert 'logits_concrete' in output, "Missing logits_concrete"
    assert 'fusion_weights' in output, "Missing fusion_weights"
    assert 'cycle_loss' in output, "Missing cycle_loss"
    assert output['logits'].shape == (1, 2), f"Unexpected logits shape: {output['logits'].shape}"

    print(f"✓ Dual-pass mode (default) works!")
    print(f"  Fused logits shape: {output['logits'].shape}")
    print(f"  Abstract logits shape: {output['logits_abstract'].shape}")
    print(f"  Concrete logits shape: {output['logits_concrete'].shape}")
    print(f"  Fusion weights: {output['fusion_weights']}")
    print(f"  Cycle loss: {output['cycle_loss'].item():.4f}")
    return True

def test_single_pass_mode():
    """Test single-pass mode (opt-out via use_dual_pass=False)."""
    print("\nTesting single-pass mode (opt-out)...")

    model = NSMModel(
        node_features=64,
        num_relations=4,
        num_classes=2,
        task_type='classification',
        num_levels=3,
        use_dual_pass=False  # Explicitly opt-out of dual-pass
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

    print(f"✓ Single-pass mode (opt-out) works!")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Cycle loss: {output['cycle_loss'].item():.4f}")
    return True

def test_dual_pass_learned_fusion():
    """Test dual-pass mode with learned fusion."""
    print("\nTesting dual-pass with learned fusion...")

    model = NSMModel(
        node_features=64,
        num_relations=4,
        num_classes=2,
        task_type='classification',
        num_levels=3,
        use_dual_pass=True,
        fusion_mode='learned'
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

    print(f"✓ Dual-pass with learned fusion works!")
    print(f"  Fused logits shape: {output['logits'].shape}")
    print(f"  Fusion weights (learned): {output['fusion_weights']}")
    print(f"  Cycle loss: {output['cycle_loss'].item():.4f}")
    return True

def test_2level_backward_compat():
    """Test 2-level mode for backward compatibility.

    Note: 2-level mode doesn't support dual-pass, so use_dual_pass=False is required.
    """
    print("\nTesting 2-level mode (backward compatibility)...")

    model = NSMModel(
        node_features=64,
        num_relations=4,
        num_classes=2,
        task_type='classification',
        num_levels=2,  # 2-level mode
        use_dual_pass=False  # 2-level doesn't support dual-pass
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
    print("Merge Verification Test Suite (Phase 1.5)")
    print("=" * 60)
    print("Note: Dual-pass mode is now the default (use_dual_pass=True)\n")

    try:
        test_dual_pass_default()
        test_dual_pass_learned_fusion()
        test_single_pass_mode()
        test_2level_backward_compat()

        print("\n" + "=" * 60)
        print("✓ All tests passed! Merge successful.")
        print("=" * 60)
        print("\n✅ Dual-pass mode is now the default")
        print("✅ Single-pass mode available via use_dual_pass=False")
        print("✅ Backward compatibility maintained with 2-level mode")
        print("\nReady to push and create PR!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
