"""
NSM evaluation and validation modules.

Provides metrics, preflight checks, and domain-specific evaluation utilities.
"""

# Preflight checks (from main branch)
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

# Planning-specific metrics (from dataset-planning branch)
from nsm.evaluation.planning_metrics import (
    PlanningMetrics,
    goal_achievement_rate,
    invalid_sequence_detection,
    temporal_ordering_accuracy,
    capability_coverage,
    decomposition_accuracy
)

# Process cleanup utilities (from main branch)
from nsm.evaluation.process_cleanup import (
    check_and_cleanup,
    find_training_processes,
    kill_process
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
    # Planning metrics
    'PlanningMetrics',
    'goal_achievement_rate',
    'invalid_sequence_detection',
    'temporal_ordering_accuracy',
    'capability_coverage',
    'decomposition_accuracy',
    # Process cleanup
    'check_and_cleanup',
    'find_training_processes',
    'kill_process',
]
