"""Tri-fold confidence semiring utilities.

This module provides a semiring where each element tracks four correlated
log-scores corresponding to subject, predicate, object, and a shared center
channel.  The :class:`TriFoldSemiring` composes multi-hop reasoning chains via
component-wise addition (log-space composition) and aggregates alternative paths
with component-wise maxima.  Helper functions convert between structured tuples
and packed tensors and expose differentiable folding operators (``Phi``) that
update the center channel together with an unfolding operator (``Psi``) that
broadcasts the center score back to the leaf channels.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor

from .base import BaseSemiring


TRIFOLD_CHANNELS = 4


def as_trifold(
    subject: Tensor,
    predicate: Tensor,
    obj: Tensor,
    center: Tensor,
) -> Tensor:
    """Stack individual channels into a tri-fold tensor.

    All channels are broadcast to a common shape before being stacked along the
    last dimension.  The resulting tensor always has a final dimension of size 4
    representing ``(subject, predicate, object, center)``.
    """

    subject, predicate, obj, center = torch.broadcast_tensors(
        subject, predicate, obj, center
    )
    return torch.stack((subject, predicate, obj, center), dim=-1)


def split_trifold(trifold: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split a tri-fold tensor into ``(subject, predicate, object, center)``."""

    if trifold.size(-1) != TRIFOLD_CHANNELS:
        raise ValueError(
            f"Expected last dimension of size {TRIFOLD_CHANNELS}, got {trifold.size(-1)}"
        )
    subject = trifold[..., 0]
    predicate = trifold[..., 1]
    obj = trifold[..., 2]
    center = trifold[..., 3]
    return subject, predicate, obj, center


def _ensure_trifold(confidences: Tensor) -> Tensor:
    if confidences.size(-1) != TRIFOLD_CHANNELS:
        raise ValueError(
            "TriFoldSemiring expects tensors with last dimension == 4. "
            f"Received shape {tuple(confidences.shape)}"
        )
    return confidences


class TriFoldSemiring(BaseSemiring):
    """Semiring for reasoning over tri-fold log-score tuples."""

    def combine(self, confidences: Tensor, dim: int = -2, **_: object) -> Tensor:
        """Compose sequential steps by component-wise addition."""

        confidences = _ensure_trifold(confidences)
        if confidences.ndim < 2:
            return confidences
        return torch.sum(confidences, dim=dim)

    def aggregate(self, confidences: Tensor, dim: int = -2, **_: object) -> Tensor:
        """Aggregate alternative paths with component-wise maxima."""

        confidences = _ensure_trifold(confidences)
        if confidences.ndim < 2:
            return confidences
        return torch.max(confidences, dim=dim).values

    def get_name(self) -> str:
        return "TriFold"

    def get_combine_identity(self, reference: Tensor) -> Tensor:
        """Return the additive identity (zero vector) for the semiring."""

        return torch.zeros_like(reference)


def _phi(trifold: Tensor, reducer: Callable[[Tensor], Tensor]) -> Tensor:
    """Apply a reducer over subject/predicate/object and update the center."""

    trifold = _ensure_trifold(trifold)
    subject, predicate, obj, _ = split_trifold(trifold)
    stacked = torch.stack((subject, predicate, obj), dim=-1)
    new_center = reducer(stacked)
    return as_trifold(subject, predicate, obj, new_center)


def Phi_min(trifold: Tensor) -> Tensor:
    """Fold the minimum of subject/predicate/object into the center channel."""

    return _phi(trifold, lambda x: torch.min(x, dim=-1).values)


def Phi_mean(trifold: Tensor) -> Tensor:
    """Fold the mean of subject/predicate/object into the center channel."""

    return _phi(trifold, lambda x: torch.mean(x, dim=-1))


def Phi_logsumexp(trifold: Tensor) -> Tensor:
    """Fold the log-sum-exp of subject/predicate/object into the center channel."""

    return _phi(trifold, lambda x: torch.logsumexp(x, dim=-1))


def Psi(trifold: Tensor) -> Tensor:
    """Broadcast the center channel back to subject/predicate/object channels."""

    trifold = _ensure_trifold(trifold)
    _, _, _, center = split_trifold(trifold)
    return as_trifold(center, center, center, center)

