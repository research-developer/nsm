"""
Abstract base class for confidence propagation semirings.

Defines the interface that all semiring implementations must satisfy:
- combine(): Sequential reasoning (multi-hop paths)
- aggregate(): Alternative paths (multiple routes to same conclusion)

Implementations include:
- ProductSemiring: Probabilistic independence (multiply)
- MinMaxSemiring: Conservative reasoning (min/max)
- LearnedSemiring: Neural aggregation (NSM-12 exploration)
"""

from abc import ABC, abstractmethod
import torch
from torch import Tensor
from typing import Optional, Dict, Any


class BaseSemiring(ABC):
    """Abstract interface for confidence propagation.

    All semiring implementations must provide:
    1. combine(): Combine confidences along a sequential path
    2. aggregate(): Aggregate confidences from alternative paths

    Mathematical properties (recommended but not enforced):
    - combine() should be associative: (a⊗b)⊗c = a⊗(b⊗c)
    - combine() should have identity element (typically 1.0)
    - aggregate() should be commutative: a⊕b = b⊕a
    - aggregate() should be idempotent: a⊕a = a

    Example Usage:
        >>> semiring = ProductSemiring(temperature=1.0)
        >>>
        >>> # Sequential reasoning: A→B→C
        >>> path_conf = torch.tensor([0.9, 0.8, 0.7])
        >>> combined = semiring.combine(path_conf)  # 0.504
        >>>
        >>> # Alternative paths to same goal
        >>> alt_paths = torch.tensor([0.6, 0.8, 0.7])
        >>> aggregated = semiring.aggregate(alt_paths)  # ~0.8 (best path)
    """

    @abstractmethod
    def combine(
        self,
        confidences: Tensor,
        dim: int = -1,
        **kwargs
    ) -> Tensor:
        """Combine confidences along sequential reasoning steps.

        Used for multi-hop inference chains where each step depends on previous:
        - Path A→B→C with confidences [c_AB, c_BC]
        - Final confidence for A→C = combine([c_AB, c_BC])

        Args:
            confidences (Tensor): Confidence values to combine.
                Shape: [..., num_steps] where num_steps is number of hops
            dim (int, optional): Dimension to combine along. Defaults to -1.
            **kwargs: Semiring-specific parameters (e.g., temperature)

        Returns:
            Tensor: Combined confidence. Shape: [...] (dim removed)

        Example:
            >>> # 3-hop path with decreasing confidence
            >>> conf = torch.tensor([0.9, 0.8, 0.7])
            >>> product_sem.combine(conf)  # 0.504 (multiply)
            >>> minmax_sem.combine(conf)   # 0.7 (minimum, bottleneck)
        """
        pass

    @abstractmethod
    def aggregate(
        self,
        confidences: Tensor,
        dim: int = -1,
        **kwargs
    ) -> Tensor:
        """Aggregate confidences from alternative reasoning paths.

        Used when multiple independent paths lead to same conclusion:
        - Path 1: A→B→C (confidence 0.7)
        - Path 2: A→D→C (confidence 0.8)
        - Path 3: A→E→C (confidence 0.6)
        - Final confidence = aggregate([0.7, 0.8, 0.6])

        Args:
            confidences (Tensor): Confidence values from different paths.
                Shape: [..., num_paths]
            dim (int, optional): Dimension to aggregate along. Defaults to -1.
            **kwargs: Semiring-specific parameters

        Returns:
            Tensor: Aggregated confidence. Shape: [...] (dim removed)

        Example:
            >>> # Multiple paths with varying confidence
            >>> paths = torch.tensor([0.7, 0.8, 0.6])
            >>> product_sem.aggregate(paths)  # ~0.8 (probabilistic sum)
            >>> minmax_sem.aggregate(paths)   # 0.8 (maximum, best path)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable name for logging and debugging.

        Returns:
            str: Semiring type name (e.g., "Product", "MinMax", "Learned")
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.get_name()}')"


class ConfidenceMetrics:
    """Utility class for computing confidence-related metrics.

    Useful for monitoring training and evaluating calibration.
    """

    @staticmethod
    def entropy(confidences: Tensor, eps: float = 1e-8) -> Tensor:
        """Compute entropy of confidence distribution.

        High entropy → uncertain (uniform distribution)
        Low entropy → confident (peaked distribution)

        Args:
            confidences (Tensor): Confidence values in [0, 1]
            eps (float): Small constant for numerical stability

        Returns:
            Tensor: Entropy values
        """
        # Binary entropy: -p*log(p) - (1-p)*log(1-p)
        p = torch.clamp(confidences, eps, 1 - eps)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    @staticmethod
    def expected_calibration_error(
        confidences: Tensor,
        correctness: Tensor,
        num_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        Measures how well predicted confidence matches actual accuracy.
        Lower ECE → better calibrated model.

        Args:
            confidences (Tensor): Predicted confidence values [0, 1]
            correctness (Tensor): Binary correctness (1=correct, 0=incorrect)
            num_bins (int): Number of bins for discretization

        Returns:
            float: ECE value (0 = perfect calibration)

        Reference:
            Guo et al. (2017) - "On Calibration of Modern Neural Networks"
        """
        device = confidences.device
        ece = 0.0
        total = len(confidences)

        for i in range(num_bins):
            bin_lower = i / num_bins
            bin_upper = (i + 1) / num_bins

            # Find samples in this bin
            in_bin = ((confidences > bin_lower) & (confidences <= bin_upper))
            bin_size = in_bin.sum().item()

            if bin_size > 0:
                # Average confidence in bin
                bin_conf = confidences[in_bin].mean().item()

                # Accuracy in bin
                bin_acc = correctness[in_bin].float().mean().item()

                # Weighted difference
                ece += (bin_size / total) * abs(bin_conf - bin_acc)

        return ece


def verify_semiring_properties(
    semiring: BaseSemiring,
    test_values: Optional[Tensor] = None,
    atol: float = 1e-5
) -> Dict[str, bool]:
    """Test mathematical properties of a semiring implementation.

    Checks:
    - Associativity of combine: (a⊗b)⊗c = a⊗(b⊗c)
    - Identity element for combine: a⊗1 = a
    - Commutativity of aggregate: a⊕b = b⊕a
    - Output range: [0, 1]

    Args:
        semiring (BaseSemiring): Semiring to test
        test_values (Tensor, optional): Custom test values. If None, uses random.
        atol (float): Absolute tolerance for equality checks

    Returns:
        Dict[str, bool]: Property name → passed (True/False)

    Example:
        >>> semiring = ProductSemiring()
        >>> results = verify_semiring_properties(semiring)
        >>> assert all(results.values()), f"Failed: {results}"
    """
    if test_values is None:
        test_values = torch.rand(5) * 0.5 + 0.3  # [0.3, 0.8]

    results = {}

    try:
        # Test 1: Associativity of combine: (a⊗b)⊗c = a⊗(b⊗c)
        a, b, c = test_values[:3]
        ab = semiring.combine(torch.stack([a, b]))
        left = semiring.combine(torch.stack([ab, c]))

        bc = semiring.combine(torch.stack([b, c]))
        right = semiring.combine(torch.stack([a, bc]))

        results['combine_associative'] = torch.allclose(left, right, atol=atol)

    except Exception as e:
        results['combine_associative'] = False
        results['combine_error'] = str(e)

    try:
        # Test 2: Identity element (typically 1.0)
        a = test_values[0]
        identity = torch.tensor(1.0)
        combined = semiring.combine(torch.stack([a, identity]))
        results['combine_identity'] = torch.allclose(combined, a, atol=atol)

    except Exception as e:
        results['combine_identity'] = False

    try:
        # Test 3: Commutativity of aggregate: a⊕b = b⊕a
        a, b = test_values[:2]
        agg_ab = semiring.aggregate(torch.stack([a, b]))
        agg_ba = semiring.aggregate(torch.stack([b, a]))
        results['aggregate_commutative'] = torch.allclose(agg_ab, agg_ba, atol=atol)

    except Exception as e:
        results['aggregate_commutative'] = False

    try:
        # Test 4: Output range [0, 1]
        combined = semiring.combine(test_values)
        aggregated = semiring.aggregate(test_values)

        in_range_combine = (0 <= combined <= 1).all()
        in_range_aggregate = (0 <= aggregated <= 1).all()
        results['output_range_valid'] = bool(in_range_combine and in_range_aggregate)

    except Exception as e:
        results['output_range_valid'] = False

    return results
