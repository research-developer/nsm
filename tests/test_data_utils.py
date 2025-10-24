"""
Regression tests for data utilities.

Tests critical edge cases discovered during NSM-33 10x validation,
specifically the train/val split logic that caused ZeroDivisionError
when validation sets became empty.
"""

import pytest
from nsm.data.utils import adaptive_train_val_split


class TestAdaptiveTrainValSplit:
    """Test train/val splitting with various edge cases."""

    def test_sufficient_data_exact_split(self):
        """Test normal case: enough data for requested split."""
        samples = list(range(25000))
        train, val = adaptive_train_val_split(
            all_samples=samples,
            train_size=20000,
            min_val_size=1000
        )

        assert len(train) == 20000, "Train set should be exactly requested size"
        assert len(val) == 5000, "Val set should be remainder"
        assert set(train + val) == set(samples), "No samples lost"
        assert len(set(train) & set(val)) == 0, "No overlap between train/val"

    def test_insufficient_data_adaptive_split(self):
        """Test adaptive mode: dataset smaller than train_size + min_val."""
        samples = list(range(16800))  # Real NSM-33 scenario
        train, val = adaptive_train_val_split(
            all_samples=samples,
            train_size=20000,
            min_val_size=1000,
            train_ratio=0.833
        )

        # Should use adaptive ratio
        expected_train = int(16800 * 0.833)
        expected_val = 16800 - expected_train

        assert len(train) == expected_train, f"Expected {expected_train} train samples"
        assert len(val) == expected_val, f"Expected {expected_val} val samples"
        assert len(train) + len(val) == 16800, "All samples used"

    def test_minimum_validation_size_enforced(self):
        """Test that min_val_size is enforced even with adaptive split."""
        samples = list(range(5000))
        train, val = adaptive_train_val_split(
            all_samples=samples,
            train_size=20000,
            min_val_size=1000,
            train_ratio=0.833
        )

        assert len(val) >= 1000, "Validation size should meet minimum"
        assert len(train) == 4000, "Train adjusted to maintain min val size"

    def test_tiny_dataset_below_minimum(self):
        """Test that tiny datasets raise informative error."""
        samples = list(range(100))  # Too small

        with pytest.raises(ValueError, match="Dataset too small"):
            adaptive_train_val_split(
                all_samples=samples,
                train_size=20000,
                min_val_size=1000
            )

    def test_edge_case_exact_minimum(self):
        """Test edge case: dataset exactly at minimum threshold."""
        samples = list(range(1010))  # Just above minimum (1000 + 10)
        train, val = adaptive_train_val_split(
            all_samples=samples,
            train_size=20000,
            min_val_size=1000
        )

        assert len(val) >= 1000, "Should maintain minimum val size"
        assert len(train) + len(val) == 1010, "All samples used"

    def test_zero_size_validation_prevented(self):
        """
        Regression test for NSM-33 bug: empty validation set.

        The original rigid split logic could create empty validation sets:
          train_graphs = all_graphs[:20000]
          val_graphs = all_graphs[20000:]  # Empty if len < 20000!

        This caused ZeroDivisionError: val_loss /= len(val_loader)
        """
        samples = list(range(18000))  # Less than requested train_size

        # Original buggy logic would do:
        # train = samples[:20000]  # Gets all 18000
        # val = samples[20000:]     # Gets nothing! []

        # Fixed logic should never create empty validation set
        train, val = adaptive_train_val_split(
            all_samples=samples,
            train_size=20000,
            min_val_size=1000
        )

        assert len(val) > 0, "CRITICAL: Validation set must never be empty"
        assert len(val) >= 1000, "Validation set must meet minimum size"
        assert len(train) > 0, "Train set must not be empty"

    def test_custom_train_ratio(self):
        """Test custom train ratio parameter."""
        samples = list(range(10000))
        train, val = adaptive_train_val_split(
            all_samples=samples,
            train_size=20000,
            min_val_size=500,
            train_ratio=0.9  # 90/10 split
        )

        # Should use 90/10 split since adaptive mode triggered
        assert len(train) == 9000, "Should use custom 90% ratio"
        assert len(val) == 1000, "Should be 10% remainder"

    def test_no_data_loss_in_split(self):
        """Ensure no samples are lost or duplicated during split."""
        samples = list(range(15000))
        train, val = adaptive_train_val_split(
            all_samples=samples,
            train_size=20000,
            min_val_size=1000
        )

        all_split = train + val
        assert len(all_split) == len(samples), "All samples accounted for"
        assert set(all_split) == set(samples), "No samples lost or duplicated"
        assert len(set(train) & set(val)) == 0, "No overlap"

    def test_reproducibility(self):
        """Test that split is deterministic given same inputs."""
        samples = list(range(12000))

        train1, val1 = adaptive_train_val_split(samples, 20000, 1000)
        train2, val2 = adaptive_train_val_split(samples, 20000, 1000)

        assert train1 == train2, "Train sets should be identical"
        assert val1 == val2, "Val sets should be identical"

    def test_large_scale_split(self):
        """Test with large-scale data similar to NSM Phase 2."""
        samples = list(range(100000))
        train, val = adaptive_train_val_split(
            all_samples=samples,
            train_size=80000,
            min_val_size=5000
        )

        assert len(train) == 80000
        assert len(val) == 20000
        assert len(set(train) & set(val)) == 0


class TestEdgeCasesFromNSM33:
    """
    Real edge cases discovered during NSM-33 validation runs.

    These tests document actual bugs that caused experiment failures
    and ensure they never regress.
    """

    def test_nsm33_original_failure_scenario(self):
        """
        Exact scenario from NSM-33 that caused ZeroDivisionError.

        Modal logs showed:
        - Requested: 24,000 problems
        - Materialized: 16,800 (70% train split)
        - Buggy split: train = [:20000] = all 16800, val = [20000:] = []
        - Result: ZeroDivisionError at val_loss /= len(val_loader)
        """
        # Simulate exact NSM-33 scenario
        full_dataset_size = 16800  # What actually materialized
        requested_train = 20000    # What was requested

        samples = list(range(full_dataset_size))

        # Buggy original logic (DO NOT USE):
        # train_graphs = all_graphs[:requested_train]  # Takes all 16800
        # val_graphs = all_graphs[requested_train:]     # Empty!

        # Fixed logic:
        train, val = adaptive_train_val_split(
            all_samples=samples,
            train_size=requested_train,
            min_val_size=1000
        )

        # Assertions that would have caught the bug
        assert len(val) > 0, "CRITICAL BUG: Empty validation set causes ZeroDivisionError"
        assert len(val) >= 1000, "Validation set too small for meaningful metrics"
        assert len(train) + len(val) == full_dataset_size, "Data loss detected"

        # Verify split ratios are reasonable
        val_ratio = len(val) / full_dataset_size
        assert 0.10 <= val_ratio <= 0.30, f"Val ratio {val_ratio:.2f} outside reasonable range"

    def test_nsm33_all_experiment_scenarios(self):
        """Test all NSM-33 experiment configurations."""
        scenarios = [
            ("10x_baseline", 16800, 20000),
            ("10x_adaptive", 16800, 20000),
            ("10x_fixed_temp", 16800, 20000),
        ]

        for name, dataset_size, requested_train in scenarios:
            samples = list(range(dataset_size))
            train, val = adaptive_train_val_split(samples, requested_train, 1000)

            assert len(val) >= 1000, f"{name}: Val size below minimum"
            assert len(train) > 0, f"{name}: Empty train set"
            assert len(val) > 0, f"{name}: Empty val set"
            assert len(train) + len(val) == dataset_size, f"{name}: Data loss"
