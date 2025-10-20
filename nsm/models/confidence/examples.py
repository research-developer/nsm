"""
Example semiring implementations for testing infrastructure.

These are reference implementations demonstrating the BaseSemiring interface.
Production versions for NSM-12 exploration branches will be more sophisticated.

Implementations:
- ProductSemiring: Probabilistic independence (multiply for combine)
- MinMaxSemiring: Conservative reasoning (min for combine, max for aggregate)
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import BaseSemiring


class ProductSemiring(BaseSemiring):
    """Product semiring for probabilistic confidence propagation.

    Semantics:
    - combine(): Product of independent probabilities
    - aggregate(): Probabilistic sum (1 - ∏(1-p_i))

    Suitable for:
    - Independent reasoning steps
    - Probabilistic inference
    - Bayesian-style confidence

    Mathematical formulation:
    - Combine: c_path = c_1 × c_2 × ... × c_n
    - Aggregate: c_total = 1 - ∏(1 - c_i)

    Args:
        temperature (float): Temperature for soft operations. Default: 1.0
        min_confidence (float): Minimum confidence threshold. Default: 1e-6

    Example:
        >>> semiring = ProductSemiring()
        >>>
        >>> # Path A→B→C with confidences [0.9, 0.8]
        >>> path = torch.tensor([0.9, 0.8])
        >>> combined = semiring.combine(path)  # 0.72
        >>>
        >>> # Three alternative paths [0.7, 0.8, 0.6]
        >>> paths = torch.tensor([0.7, 0.8, 0.6])
        >>> aggregated = semiring.aggregate(paths)  # 0.976 (very confident)
    """

    def __init__(self, temperature: float = 1.0, min_confidence: float = 1e-6):
        self.temperature = temperature
        self.min_confidence = min_confidence

    def combine(
        self,
        confidences: Tensor,
        dim: int = -1,
        **kwargs
    ) -> Tensor:
        """Combine via product (independent probabilities).

        Args:
            confidences (Tensor): Confidence values in [0, 1]
            dim (int): Dimension to combine along

        Returns:
            Tensor: Combined confidence (product)
        """
        # Clamp to avoid numerical issues
        conf = torch.clamp(confidences, self.min_confidence, 1.0)

        # Product of probabilities
        return torch.prod(conf, dim=dim)

    def aggregate(
        self,
        confidences: Tensor,
        dim: int = -1,
        **kwargs
    ) -> Tensor:
        """Aggregate via probabilistic sum.

        Uses formula: P(A ∪ B) = 1 - P(¬A) × P(¬B)

        For multiple paths: 1 - ∏(1 - c_i)

        Args:
            confidences (Tensor): Confidence values in [0, 1]
            dim (int): Dimension to aggregate along

        Returns:
            Tensor: Aggregated confidence
        """
        # Clamp to avoid numerical issues
        conf = torch.clamp(confidences, self.min_confidence, 1 - self.min_confidence)

        # Probabilistic sum: 1 - product of complements
        complements = 1.0 - conf
        product_complements = torch.prod(complements, dim=dim)

        return 1.0 - product_complements

    def get_name(self) -> str:
        return "Product"


class MinMaxSemiring(BaseSemiring):
    """Min-Max semiring for conservative confidence propagation.

    Semantics:
    - combine(): Minimum (bottleneck/weakest link)
    - aggregate(): Maximum (best alternative path)

    Suitable for:
    - Conservative reasoning
    - Worst-case analysis
    - Bottleneck identification

    Mathematical formulation:
    - Combine: c_path = min(c_1, c_2, ..., c_n)
    - Aggregate: c_total = max(c_1, c_2, ..., c_n)

    Soft versions (differentiable):
    - Soft min: weighted average with high weight on minimum
    - Soft max: log-sum-exp approximation

    Args:
        temperature (float): Temperature for soft min/max. Default: 1.0
            - High temperature → uniform average
            - Low temperature → approaches hard min/max
        use_soft (bool): Use soft (differentiable) versions. Default: True

    Example:
        >>> semiring = MinMaxSemiring(temperature=0.5)
        >>>
        >>> # Path A→B→C with confidences [0.9, 0.8, 0.7]
        >>> path = torch.tensor([0.9, 0.8, 0.7])
        >>> combined = semiring.combine(path)  # ~0.7 (bottleneck)
        >>>
        >>> # Three alternative paths [0.7, 0.8, 0.6]
        >>> paths = torch.tensor([0.7, 0.8, 0.6])
        >>> aggregated = semiring.aggregate(paths)  # ~0.8 (best path)
    """

    def __init__(self, temperature: float = 1.0, use_soft: bool = True):
        self.temperature = temperature
        self.use_soft = use_soft

    def combine(
        self,
        confidences: Tensor,
        dim: int = -1,
        **kwargs
    ) -> Tensor:
        """Combine via minimum (bottleneck).

        Uses soft minimum if use_soft=True:
            soft_min(x) = Σ w_i × x_i
            where w_i = softmax(-x_i / τ)

        Args:
            confidences (Tensor): Confidence values in [0, 1]
            dim (int): Dimension to combine along

        Returns:
            Tensor: Combined confidence (minimum or soft minimum)
        """
        if not self.use_soft or self.temperature < 0.01:
            # Hard minimum
            return torch.min(confidences, dim=dim).values

        # Soft minimum: weighted average with weights favoring small values
        # weights = softmax(-x / τ) gives higher weight to smaller values
        weights = F.softmax(-confidences / self.temperature, dim=dim)
        return torch.sum(weights * confidences, dim=dim)

    def aggregate(
        self,
        confidences: Tensor,
        dim: int = -1,
        **kwargs
    ) -> Tensor:
        """Aggregate via maximum (best path).

        Uses soft maximum if use_soft=True:
            soft_max(x) = τ × log(Σ exp(x_i / τ))

        Args:
            confidences (Tensor): Confidence values in [0, 1]
            dim (int): Dimension to aggregate along

        Returns:
            Tensor: Aggregated confidence (maximum or soft maximum)
        """
        if not self.use_soft or self.temperature < 0.01:
            # Hard maximum
            return torch.max(confidences, dim=dim).values

        # Soft maximum via log-sum-exp
        return self.temperature * torch.logsumexp(
            confidences / self.temperature, dim=dim
        )

    def set_temperature(self, temp: float) -> None:
        """Update temperature (for annealing during training).

        Args:
            temp (float): New temperature value
        """
        self.temperature = temp

    def get_name(self) -> str:
        return f"MinMax(τ={self.temperature:.3f})"


class HybridSemiring(BaseSemiring):
    """Hybrid semiring combining Product and MinMax.

    Uses Product for combine() and MinMax for aggregate(), or vice versa.
    Allows mixing semantic interpretations:
    - Product combine + MinMax aggregate: Probabilistic paths, best alternative
    - MinMax combine + Product aggregate: Conservative paths, probabilistic union

    Args:
        combine_type (str): 'product' or 'minmax' for combine()
        aggregate_type (str): 'product' or 'minmax' for aggregate()
        temperature (float): Temperature for soft operations

    Example:
        >>> # Probabilistic paths, best alternative
        >>> semiring = HybridSemiring('product', 'minmax')
        >>>
        >>> # Conservative paths, probabilistic union
        >>> semiring2 = HybridSemiring('minmax', 'product')
    """

    def __init__(
        self,
        combine_type: str = 'product',
        aggregate_type: str = 'minmax',
        temperature: float = 1.0
    ):
        assert combine_type in ['product', 'minmax']
        assert aggregate_type in ['product', 'minmax']

        self.combine_type = combine_type
        self.aggregate_type = aggregate_type
        self.temperature = temperature

        # Create sub-semirings
        self._product = ProductSemiring(temperature)
        self._minmax = MinMaxSemiring(temperature)

    def combine(self, confidences: Tensor, dim: int = -1, **kwargs) -> Tensor:
        if self.combine_type == 'product':
            return self._product.combine(confidences, dim, **kwargs)
        else:
            return self._minmax.combine(confidences, dim, **kwargs)

    def aggregate(self, confidences: Tensor, dim: int = -1, **kwargs) -> Tensor:
        if self.aggregate_type == 'product':
            return self._product.aggregate(confidences, dim, **kwargs)
        else:
            return self._minmax.aggregate(confidences, dim, **kwargs)

    def get_name(self) -> str:
        return f"Hybrid({self.combine_type}/{self.aggregate_type})"
