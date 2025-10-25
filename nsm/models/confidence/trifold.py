"""Tri-fold semiring for subject/predicate/object reasoning chains.

This module introduces :class:`TriFoldSemiring`, a lightweight semiring whose
elements are four-channel log-score tuples ``(s, p, o, c)`` representing the
confidence of subject, predicate, object, and their shared centre context.

The semiring follows log-domain arithmetic:

* ``combine`` performs component-wise addition across sequential reasoning
  steps, matching multiplication in probability space while remaining stable
  for negative log-scores.
* ``aggregate`` selects the component-wise maximum across alternative paths
  (best path semantics in log-space).

Helper utilities are provided to pack/unpack tri-fold tensors and to perform
``fold``/``unfold`` operations (:math:`\Phi`/ :math:`\Psi`) that share signal
between the outer channels (subject/predicate/object) and the centre channel.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from .base import BaseSemiring

__all__ = [
    "TriFoldSemiring",
    "trifold_tensor",
    "split_trifold",
    "fold",
    "fold_min",
    "fold_mean",
    "fold_logsumexp",
    "Phi",
    "Phi_min",
    "Phi_mean",
    "Phi_logsumexp",
    "unfold",
    "Psi",
    "Psi_add",
    "Psi_replace",
    "Psi_max",
]


def _ensure_trifold(tensor: Tensor) -> Tensor:
    if tensor.size(-1) != 4:
        raise ValueError(
            f"Expected final dimension of size 4 for tri-fold tensor, got {tensor.size(-1)}"
        )
    return tensor


def _is_probability_tensor(tensor: Tensor) -> bool:
    if tensor.numel() == 0:
        return False
    bounds = (tensor >= 0) & (tensor <= 1)
    return bool(bounds.all().item())


def trifold_tensor(
    subject: Tensor,
    predicate: Tensor,
    obj: Tensor,
    center: Tensor | None = None,
) -> Tensor:
    """Stack four log-score channels into a tri-fold tensor.

    All inputs are broadcast to a common shape before stacking. When ``center``
    is omitted a zero log-score (``log(1)``) centre channel is used.
    """

    subject, predicate, obj = torch.broadcast_tensors(subject, predicate, obj)
    if center is None:
        center = torch.zeros_like(subject)
    else:
        center = center.expand_as(subject)
    return torch.stack((subject, predicate, obj, center), dim=-1)


def split_trifold(tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Unpack a tri-fold tensor into ``(subject, predicate, object, center)``."""

    tensor = _ensure_trifold(tensor)
    return tensor.unbind(dim=-1)


class TriFoldSemiring(BaseSemiring):
    """Semiring operating on log-score quadruples."""

    def combine(self, confidences: Tensor, dim: int = -2, **_: object) -> Tensor:
        """Component-wise addition along ``dim`` for tri-fold inputs.

        For compatibility with :func:`verify_semiring_properties` the method
        falls back to multiplicative behaviour when the tensor does not contain
        the tri-fold channel dimension and the values are in ``[0, 1]``.
        """

        if confidences.size(-1) == 4:
            reduce_dim = dim if dim >= 0 else confidences.dim() + dim
            channel_dim = confidences.dim() - 1
            if reduce_dim == channel_dim:
                raise ValueError("combine dimension cannot be the channel axis")
            return confidences.sum(dim=dim)

        if _is_probability_tensor(confidences):
            eps = torch.finfo(confidences.dtype).tiny
            logs = torch.log(confidences.clamp_min(eps))
            combined = logs.sum(dim=dim)
            return torch.exp(combined)

        return confidences.sum(dim=dim)

    def aggregate(self, confidences: Tensor, dim: int = -2, **_: object) -> Tensor:
        """Component-wise maximum along ``dim`` for tri-fold inputs."""

        if confidences.size(-1) == 4:
            reduce_dim = dim if dim >= 0 else confidences.dim() + dim
            channel_dim = confidences.dim() - 1
            if reduce_dim == channel_dim:
                raise ValueError("aggregate dimension cannot be the channel axis")
            return confidences.max(dim=dim).values

        return confidences.max(dim=dim).values

    def get_name(self) -> str:  # pragma: no cover - trivial accessor
        return "TriFold"


_FOLD_REDUCTIONS = {
    "min": torch.min,
    "mean": torch.mean,
    "logsumexp": torch.logsumexp,
}


def fold(tensor: Tensor, reduction: str = "logsumexp") -> Tensor:
    """Apply a fold (:math:`\Phi`) update on the centre channel.

    Args:
        tensor: Tri-fold log-score tensor.
        reduction: Reduction name (``"min"``, ``"mean"`` or ``"logsumexp"``).
    """

    tensor = _ensure_trifold(tensor)
    reduction = reduction.lower()
    if reduction not in _FOLD_REDUCTIONS:
        raise ValueError(f"Unsupported reduction '{reduction}'")

    outer = tensor[..., :3]
    reducer = _FOLD_REDUCTIONS[reduction]

    if reduction == "mean":
        center = reducer(outer, dim=-1)
    elif reduction == "min":
        center = reducer(outer, dim=-1).values
    else:  # logsumexp
        center = reducer(outer, dim=-1)

    return torch.cat((outer, center.unsqueeze(-1)), dim=-1)


def fold_min(tensor: Tensor) -> Tensor:
    return fold(tensor, reduction="min")


def fold_mean(tensor: Tensor) -> Tensor:
    return fold(tensor, reduction="mean")


def fold_logsumexp(tensor: Tensor) -> Tensor:
    return fold(tensor, reduction="logsumexp")


def Phi(tensor: Tensor, reduction: str = "logsumexp") -> Tensor:
    """Alias for :func:`fold` following the :math:`\Phi` notation."""

    return fold(tensor, reduction=reduction)


def Phi_min(tensor: Tensor) -> Tensor:
    return fold_min(tensor)


def Phi_mean(tensor: Tensor) -> Tensor:
    return fold_mean(tensor)


def Phi_logsumexp(tensor: Tensor) -> Tensor:
    return fold_logsumexp(tensor)


def unfold(tensor: Tensor, mode: str = "add") -> Tensor:
    """Broadcast the centre channel back to subject/predicate/object.

    Args:
        tensor: Tri-fold tensor.
        mode: Broadcast strategy - ``"add"`` (default), ``"replace"`` or
            ``"max"``.
    """

    tensor = _ensure_trifold(tensor)
    mode = mode.lower()
    outer = tensor[..., :3]
    center = tensor[..., 3].unsqueeze(-1)

    if mode == "add":
        updated = outer + center
    elif mode == "replace":
        updated = center.expand_as(outer)
    elif mode == "max":
        updated = torch.maximum(outer, center.expand_as(outer))
    else:
        raise ValueError(f"Unsupported unfold mode '{mode}'")

    return torch.cat((updated, center), dim=-1)


def Psi(tensor: Tensor, mode: str = "add") -> Tensor:
    """Alias for :func:`unfold` following the :math:`\Psi` notation."""

    return unfold(tensor, mode=mode)


def Psi_add(tensor: Tensor) -> Tensor:
    return unfold(tensor, mode="add")


def Psi_replace(tensor: Tensor) -> Tensor:
    return unfold(tensor, mode="replace")


def Psi_max(tensor: Tensor) -> Tensor:
    return unfold(tensor, mode="max")
