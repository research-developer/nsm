"""
NSM evaluation and validation modules.

Provides metrics, preflight checks, and domain-specific evaluation utilities.
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

__all__ = [
    'run_preflight_checks',
    'check_dataset_balance',
    'check_cycle_loss_weight',
    'check_learning_rate',
    'check_pyg_extensions',
    'check_model_architecture',
    'check_class_weights',
    'PreflightCheckError',
    'PreflightCheckWarning',
    'check_and_cleanup',
    'find_training_processes',
    'kill_process',
]
