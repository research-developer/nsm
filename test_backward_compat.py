#!/usr/bin/env python3
"""
Test backward compatibility with original dataset behavior.

Verifies that:
1. Default behavior (problems_per_split=False) still works
2. Split ratios are correct (70/15/15)
3. Old code still works with new implementation
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nsm.data.planning_dataset import PlanningTripleDataset


def test_backward_compatibility():
    """Test that original behavior is preserved."""
    print("=" * 80)
    print("BACKWARD COMPATIBILITY TEST")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n[1/3] Testing original behavior (problems_per_split=False)...")

        # Create dataset with old API (default behavior)
        train_dataset = PlanningTripleDataset(
            root=tmpdir,
            split='train',
            num_problems=1000,  # Total across all splits
            seed=42
        )

        val_dataset = PlanningTripleDataset(
            root=tmpdir,
            split='val',
            num_problems=1000,
            seed=42
        )

        test_dataset = PlanningTripleDataset(
            root=tmpdir,
            split='test',
            num_problems=1000,
            seed=42
        )

        print(f"      Train: {len(train_dataset)} problems")
        print(f"      Val:   {len(val_dataset)} problems")
        print(f"      Test:  {len(test_dataset)} problems")
        print(f"      Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} problems")

        # Verify split ratios
        expected_train = int(1000 * 0.7)
        expected_val = int(1000 * 0.15)
        expected_test = int(1000 * 0.15)

        assert len(train_dataset) == expected_train, \
            f"Expected {expected_train} train problems, got {len(train_dataset)}"
        assert len(val_dataset) == expected_val, \
            f"Expected {expected_val} val problems, got {len(val_dataset)}"
        assert len(test_dataset) == expected_test, \
            f"Expected {expected_test} test problems, got {len(test_dataset)}"

        print(f"      ✓ Split ratios correct (70/15/15)")

        print("\n[2/3] Testing new behavior (problems_per_split=True)...")

        # Use different directory to avoid cache collision
        import os
        tmpdir_new = os.path.join(tmpdir, 'new')
        os.makedirs(tmpdir_new, exist_ok=True)

        train_dataset_new = PlanningTripleDataset(
            root=tmpdir_new,
            split='train',
            num_problems=1000,
            problems_per_split=True,  # New flag
            seed=42
        )

        print(f"      Train: {len(train_dataset_new)} problems")

        assert len(train_dataset_new) == 1000, \
            f"Expected 1000 problems, got {len(train_dataset_new)}"

        print(f"      ✓ New flag works correctly")

        print("\n[3/3] Testing dataset access...")

        # Test that we can access problems
        for i in [0, 10, 100]:
            graph, label = train_dataset[i]
            assert graph.num_nodes > 0, f"Problem {i} has no nodes"
            assert label.item() in [0, 1], f"Problem {i} has invalid label"

        print(f"      ✓ Can access problems correctly")

        print(f"\n{'=' * 80}")
        print("BACKWARD COMPATIBILITY SUMMARY")
        print(f"{'=' * 80}")
        print(f"✓ Original behavior preserved (problems_per_split=False)")
        print(f"✓ Split ratios work correctly (70/15/15)")
        print(f"✓ New flag works correctly (problems_per_split=True)")
        print(f"✓ Dataset access works correctly")
        print(f"\n  Old code will continue to work with new implementation!")
        print(f"{'=' * 80}\n")

        return True


if __name__ == "__main__":
    try:
        success = test_backward_compatibility()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
