"""TriFold semiring and operators for neurosymbolic semantic triples.

Implements the recursive triadic confidence algebra described in the
project discussion:

- ``TriFoldSemiring`` operates in log-space over 4-tuples ``(s, p, o, c)``
  representing subject, predicate, object, and convergence scores.
- ``TriFoldFold`` implements the folding operator :math:`\Phi` that pushes
  loop evidence into the nexus.
- ``TriFoldUnfold`` implements the unfolding operator :math:`\Psi` that
  propagates nexus coherence back to each loop.
- ``TriFoldReasoner`` orchestrates iterative fold/unfold message passing and
  provides aggregated semantics for each graph in a batch.

The implementation keeps the semiring operations distributive by
encapsulating folding/unfolding as separate differentiable modules instead of
changing the semiring product.  This mirrors the specification from the
"TriFold" design document and allows seamless integration with the existing
neurosymbolic hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from .base import BaseSemiring


TRIFOLD_CHANNELS = 4  # subject, predicate, object, center


def _validate_trifold_shape(tensor: Tensor) -> None:
    """Ensure the final dimension encodes a tri-fold state."""

    if tensor.size(-1) != TRIFOLD_CHANNELS:
        raise ValueError(
            f"TriFold tensors must have last dimension of size {TRIFOLD_CHANNELS}, "
            f"got {tensor.size(-1)}"
        )


class TriFoldSemiring(BaseSemiring):
    """Tropical-style semiring over tri-fold log scores."""

    def __init__(self, zero: float = float("-inf"), one: float = 0.0):
        self.zero = zero
        self.one = one

    def combine(
        self,
        confidences: Tensor,
        dim: int = -2,
        mask: Optional[Tensor] = None,
        keepdim: bool = False,
        **kwargs,
    ) -> Tensor:
        """Sequential composition corresponds to addition in log-space."""

        _validate_trifold_shape(confidences)

        if mask is not None:
            mask = mask.to(confidences.dtype)
            while mask.dim() < confidences.dim() - 1:
                mask = mask.unsqueeze(-1)
            confidences = confidences * mask

        combined = torch.sum(confidences, dim=dim, keepdim=keepdim)
        return combined

    def aggregate(
        self,
        confidences: Tensor,
        dim: int = -2,
        mask: Optional[Tensor] = None,
        keepdim: bool = False,
        **kwargs,
    ) -> Tensor:
        """Aggregate competing hypotheses via component-wise maximum."""

        _validate_trifold_shape(confidences)

        values = confidences
        if mask is not None:
            mask = mask.to(confidences.device)
            while mask.dim() < confidences.dim() - 1:
                mask = mask.unsqueeze(-1)
            fill_value = torch.full_like(confidences, self.zero)
            values = torch.where(mask.bool(), confidences, fill_value)

        aggregated = torch.max(values, dim=dim, keepdim=keepdim).values
        return aggregated

    def element(
        self,
        subject: float,
        predicate: float,
        obj: float,
        center: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Create a tri-fold element tensor."""

        return torch.tensor([subject, predicate, obj, center], device=device, dtype=dtype)

    def get_name(self) -> str:
        return "TriFold"


@dataclass
class TriFoldOutput:
    """Container for tri-fold reasoning outputs."""

    states: Tensor
    aggregated: Tensor
    center: Tensor
    loops: Tensor
    fold_history: Tensor


class TriFoldFold(nn.Module):
    """Fold operator :math:`\Phi` that accumulates loop evidence."""

    def __init__(
        self,
        alpha: float = 1.0,
        reduction: str = "min",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

        reducers: Dict[str, Callable[[Tensor], Tensor]] = {
            "min": lambda x: torch.min(x, dim=-1).values,
            "mean": lambda x: torch.mean(x, dim=-1),
            "logsumexp": lambda x: torch.logsumexp(x, dim=-1),
        }

        if reduction not in reducers:
            raise ValueError(
                f"Unknown reduction '{reduction}'. Expected one of {list(reducers)}"
            )

        self._reduce = reducers[reduction]

    def forward(
        self,
        states: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        _validate_trifold_shape(states)

        loops, center = states[..., :3], states[..., 3:]
        fold_value = self._reduce(loops)

        if mask is not None:
            mask = mask.to(states.dtype)
            while mask.dim() < fold_value.dim():
                mask = mask.unsqueeze(-1)
            fold_value = fold_value * mask.squeeze(-1)

        center = center + self.alpha * fold_value.unsqueeze(-1)
        updated = torch.cat([loops, center], dim=-1)
        return updated, fold_value


class TriFoldUnfold(nn.Module):
    """Unfold operator :math:`\Psi` that redistributes nexus coherence."""

    def __init__(self, beta: float = 0.2) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        states: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        _validate_trifold_shape(states)

        loops, center = states[..., :3], states[..., 3:]
        delta = self.beta * center

        if mask is not None:
            mask = mask.to(states.dtype)
            while mask.dim() < delta.dim():
                mask = mask.unsqueeze(-1)
            delta = delta * mask

        loops = loops + delta
        return torch.cat([loops, center], dim=-1)


class TriFoldReasoner(nn.Module):
    """Iterative fold/unfold reasoning over tri-fold states."""

    def __init__(
        self,
        semiring: Optional[TriFoldSemiring] = None,
        alpha: float = 1.0,
        beta: float = 0.2,
        reduction: str = "min",
    ) -> None:
        super().__init__()
        self.semiring = semiring or TriFoldSemiring()
        self.fold = TriFoldFold(alpha=alpha, reduction=reduction)
        self.unfold = TriFoldUnfold(beta=beta)

    def forward(
        self,
        states: Tensor,
        batch: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        iterations: int = 1,
    ) -> TriFoldOutput:
        _validate_trifold_shape(states)

        updated = states
        history = []

        for _ in range(iterations):
            updated, fold_value = self.fold(updated, mask=mask)
            history.append(fold_value)
            updated = self.unfold(updated, mask=mask)

        if history:
            fold_history_tensor = torch.stack(history, dim=0)
        else:
            fold_history_tensor = torch.zeros(
                (0,) + updated.shape[:-1],
                device=updated.device,
                dtype=updated.dtype,
            )

        aggregated = self._aggregate(updated, batch=batch, mask=mask)
        center = aggregated[..., 3]
        loops = aggregated[..., :3]

        return TriFoldOutput(
            states=updated,
            aggregated=aggregated,
            center=center,
            loops=loops,
            fold_history=fold_history_tensor,
        )

    def _aggregate(
        self,
        states: Tensor,
        batch: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if batch is None:
            return self.semiring.aggregate(states, dim=-2 if states.dim() > 1 else 0, mask=mask)

        unique_batches = torch.unique(batch, sorted=True)
        aggregated_states = []
        for idx in unique_batches.tolist():
            batch_mask = batch == idx
            aggregated_states.append(
                self.semiring.aggregate(states[batch_mask], dim=0, mask=None)
            )

        return torch.stack(aggregated_states, dim=0)
