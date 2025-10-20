"""
Evaluation metrics for NSM models.

Provides domain-specific metrics for different dataset types.
"""

from .kg_metrics import (
    compute_link_prediction_metrics,
    compute_analogical_reasoning_accuracy,
    compute_type_consistency_accuracy,
)

__all__ = [
    'compute_link_prediction_metrics',
    'compute_analogical_reasoning_accuracy',
    'compute_type_consistency_accuracy',
]
