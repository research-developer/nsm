"""Confidence propagation infrastructure.

The package exposes classic scalar semirings (see :mod:`.examples`) alongside
multi-channel operators such as :class:`.trifold.TriFoldSemiring` for
subject/predicate/object reasoning. The tri-fold utilities (:func:`.trifold.Phi`
and :func:`.trifold.Psi`) allow differentiable folding of edge channels into a
centre context and broadcasting that context back out, enabling structured
log-domain confidence flows.
"""

from .base import BaseSemiring
from .temperature import TemperatureScheduler
from .examples import ProductSemiring, MinMaxSemiring
from .trifold import (
    TriFoldSemiring,
    trifold_tensor,
    split_trifold,
    fold,
    fold_min,
    fold_mean,
    fold_logsumexp,
    Phi,
    Phi_min,
    Phi_mean,
    Phi_logsumexp,
    unfold,
    Psi,
    Psi_add,
    Psi_replace,
    Psi_max,
)

__all__ = [
    'BaseSemiring',
    'TemperatureScheduler',
    'ProductSemiring',
    'MinMaxSemiring',
    'TriFoldSemiring',
    'trifold_tensor',
    'split_trifold',
    'fold',
    'fold_min',
    'fold_mean',
    'fold_logsumexp',
    'Phi',
    'Phi_min',
    'Phi_mean',
    'Phi_logsumexp',
    'unfold',
    'Psi',
    'Psi_add',
    'Psi_replace',
    'Psi_max',
]
