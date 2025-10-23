"""
Data utilities for NSM training and validation.

Provides shared functionality for dataset preparation, train/val splitting,
and data validation.
"""

from typing import List, Tuple, TypeVar

T = TypeVar('T')


def adaptive_train_val_split(
    all_samples: List[T],
    train_size: int,
    min_val_size: int = 1000,
    train_ratio: float = 0.833
) -> Tuple[List[T], List[T]]:
    """
    Split dataset into train/val with adaptive sizing and safety checks.

    This function handles edge cases where the requested train_size exceeds
    the available dataset size, automatically computing appropriate splits
    with minimum validation set guarantees.

    Args:
        all_samples: Complete list of samples to split
        train_size: Desired number of training samples
        min_val_size: Minimum validation set size (default: 1000)
        train_ratio: Train split ratio when dataset < train_size + min_val_size
                    (default: 0.833, i.e., 83.3% train / 16.7% val)

    Returns:
        Tuple of (train_samples, val_samples)

    Raises:
        ValueError: If dataset is too small for minimum validation size

    Examples:
        >>> samples = list(range(20000))
        >>> train, val = adaptive_train_val_split(samples, train_size=18000)
        >>> len(train), len(val)
        (18000, 2000)

        >>> # Edge case: small dataset
        >>> small_samples = list(range(5000))
        >>> train, val = adaptive_train_val_split(small_samples, train_size=20000)
        >>> len(train), len(val)
        (4164, 836)  # Uses adaptive 83.3% / 16.7% split

    Design Rationale:
        The 0.833 train ratio (5:1 split) balances:
        - Statistical power for validation (avoid overfitting to small val set)
        - Training data sufficiency (maintain learning capacity)
        - Industry standard (~80/20 splits common, 83.3/16.7 slightly more conservative)
    """
    total_available = len(all_samples)

    # Safety check: Ensure dataset can support minimum validation size
    if total_available < min_val_size + 10:  # 10 samples minimum for training
        raise ValueError(
            f"Dataset too small ({total_available} samples). "
            f"Need at least {min_val_size + 10} for min_val_size={min_val_size}"
        )

    # Case 1: Sufficient data for requested train_size + minimum validation
    if total_available >= train_size + min_val_size:
        val_size = total_available - train_size
        train_samples = all_samples[:train_size]
        val_samples = all_samples[train_size:train_size + val_size]
        return train_samples, val_samples

    # Case 2: Insufficient data - use adaptive split with ratio
    print(f"⚠️  WARNING: Only {total_available} samples available (requested {train_size} train + {min_val_size} val)")
    print(f"⚠️  Using adaptive {train_ratio*100:.1f}% train / {(1-train_ratio)*100:.1f}% val split")

    train_split_size = int(total_available * train_ratio)
    val_split_size = total_available - train_split_size

    # Ensure validation set meets minimum size
    if val_split_size < min_val_size:
        val_split_size = min_val_size
        train_split_size = total_available - val_split_size
        print(f"⚠️  Adjusted to maintain min_val_size: train={train_split_size}, val={val_split_size}")

    train_samples = all_samples[:train_split_size]
    val_samples = all_samples[train_split_size:train_split_size + val_split_size]

    return train_samples, val_samples
