"""
NSM evaluation and validation modules.

Provides metrics, preflight checks, process cleanup, and domain-specific evaluation utilities.
"""

from nsm.evaluation.preflight_checks import (
    run_preflight_checks,
    check_dataset_balance,
    check_cycle_loss_weight,
    check_learning_rate,
    check_pyg_extensions,
    check_model_architecture,
    check_class_weights,
    PreflightCheckError,
    PreflightCheckWarning
)

from nsm.evaluation.process_cleanup import (
    check_and_cleanup,
    find_training_processes,
    kill_process
)

from nsm.evaluation.kg_metrics import (
    compute_link_prediction_metrics,
    compute_analogical_reasoning_accuracy,
    compute_type_consistency_accuracy,
)

__all__ = [
    # Preflight checks
    'run_preflight_checks',
    'check_dataset_balance',
    'check_cycle_loss_weight',
    'check_learning_rate',
    'check_pyg_extensions',
    'check_model_architecture',
    'check_class_weights',
    'PreflightCheckError',
    'PreflightCheckWarning',
    # Process cleanup
    'check_and_cleanup',
    'find_training_processes',
    'kill_process',
    # KG-specific metrics
    'compute_link_prediction_metrics',
    'compute_analogical_reasoning_accuracy',
    'compute_type_consistency_accuracy',
]
