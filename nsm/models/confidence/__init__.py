"""Confidence propagation infrastructure.

The package exposes common semiring implementations together with operator
tooling for reasoning over multi-channel confidence scores.  The
``TriFoldSemiring`` models subject/predicate/object log-scores with a shared
center channel and ships with differentiable folding (``Phi``) and unfolding
(``Psi``) helpers for neural modules that need to align these channels during
training.
"""

from .base import BaseSemiring
from .temperature import TemperatureScheduler
from .examples import ProductSemiring, MinMaxSemiring
from .trifold import (
    TriFoldSemiring,
    as_trifold,
    split_trifold,
    Phi_min,
    Phi_mean,
    Phi_logsumexp,
    Psi,
)

__all__ = [
    'BaseSemiring',
    'TemperatureScheduler',
    'ProductSemiring',
    'MinMaxSemiring',
    'TriFoldSemiring',
    'as_trifold',
    'split_trifold',
    'Phi_min',
    'Phi_mean',
    'Phi_logsumexp',
    'Psi',
]
