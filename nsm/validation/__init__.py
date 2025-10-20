"""
NSM Training Validation Framework.

Provides Pydantic-based validation for:
- Dataset configuration (label balance, size standardization)
- Hyperparameters (learned bounds from NSM-10)
- Training metrics (early stopping, failure detection)
- Domain-specific constraints

All lessons from NSM-10 debugging are encoded as validators.
"""

from .test_controller import TestViewController
from .training_models import (
    CausalTrainingRun,
    DatasetConfig,
    DomainType,
    EpochMetrics,
    HyperparametersConfig,
    KnowledgeGraphTrainingRun,
    PlanningTrainingRun,
    TrainingRun,
    TrainingStatus,
)

__all__ = [
    # Test Controller
    "TestViewController",
    # Training Models
    "CausalTrainingRun",
    "KnowledgeGraphTrainingRun",
    "PlanningTrainingRun",
    "TrainingRun",
    # Configuration
    "DatasetConfig",
    "HyperparametersConfig",
    "EpochMetrics",
    # Enums
    "DomainType",
    "TrainingStatus",
]
