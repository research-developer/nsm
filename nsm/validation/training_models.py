"""
Pydantic models for training validation and monitoring.

This module provides type-safe validation for NSM training runs with:
- Pre-flight checks (dataset validation, hyperparameter bounds)
- Checkpoint validation (metric bounds, early stopping criteria)
- Experiment tracking (JSONL logging of all runs)
- Domain-specific subclasses with learned constraints

Lessons learned from NSM-10 debugging are encoded as validators.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class TrainingStatus(str, Enum):
    """Training run status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EARLY_STOPPED = "early_stopped"


class DomainType(str, Enum):
    """NSM dataset domain types."""
    CAUSAL = "causal"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    PLANNING = "planning"


class DatasetConfig(BaseModel):
    """Dataset configuration with validation bounds."""

    domain: DomainType
    split: str = Field(default="train", pattern="^(train|val|test)$")

    # Dataset size (standardized across domains per NSM-10)
    total_size: int = Field(gt=0, description="Total dataset size")
    train_size: int = Field(gt=0, description="Training set size")
    val_size: int = Field(gt=0, description="Validation set size")

    # Label distribution (must be balanced per Bug #1)
    label_balance_class_0: float = Field(ge=0.4, le=0.6, description="Class 0 proportion")
    label_balance_class_1: float = Field(ge=0.4, le=0.6, description="Class 1 proportion")

    # Domain-specific parameters
    domain_params: Dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def is_balanced(self) -> bool:
        """Check if dataset is properly balanced (40-60% per class)."""
        return (0.4 <= self.label_balance_class_0 <= 0.6 and
                0.4 <= self.label_balance_class_1 <= 0.6)

    @field_validator("train_size")
    @classmethod
    def validate_train_size(cls, v: int, info) -> int:
        """Validate training size is standardized (NSM-10 Bug #2)."""
        # Target: 2,000 training examples ±10% tolerance
        if not (1800 <= v <= 2200):
            raise ValueError(
                f"Training size {v} outside standardized range [1800, 2200]. "
                "Per NSM-10 Bug #2, all domains should have ~2,000 train examples."
            )
        return v

    @model_validator(mode="after")
    def validate_label_balance_sum(self) -> "DatasetConfig":
        """Ensure label proportions sum to ~1.0."""
        total = self.label_balance_class_0 + self.label_balance_class_1
        if not (0.95 <= total <= 1.05):
            raise ValueError(
                f"Label proportions sum to {total:.3f}, should be ~1.0. "
                "Per NSM-10 Bug #1, check for degenerate label distributions."
            )
        return self

    @model_validator(mode="after")
    def validate_split_sizes(self) -> "DatasetConfig":
        """Ensure train + val doesn't exceed total."""
        if self.train_size + self.val_size > self.total_size:
            raise ValueError(
                f"train_size ({self.train_size}) + val_size ({self.val_size}) "
                f"> total_size ({self.total_size})"
            )
        return self


class HyperparametersConfig(BaseModel):
    """Training hyperparameters with learned bounds."""

    # Training loop
    epochs: int = Field(ge=1, le=500, default=100)
    batch_size: int = Field(ge=1, le=128, default=32)
    learning_rate: float = Field(gt=0.0, le=0.1, default=0.001)
    seed: int = Field(ge=0, default=42)

    # Loss weights
    cycle_loss_weight: float = Field(ge=0.0, le=1.0, default=0.1,
                                     description="Lambda for cycle consistency loss")

    # Early stopping
    patience: int = Field(ge=5, le=50, default=20,
                         description="Epochs without improvement before stopping")
    min_delta: float = Field(ge=0.0, default=0.001,
                            description="Minimum improvement to reset patience")

    # Gradient clipping
    grad_clip_norm: Optional[float] = Field(None, gt=0.0,
                                           description="Max gradient norm (None = no clipping)")

    # Domain-specific pool ratios (learned from NSM-10)
    pool_ratio: float = Field(gt=0.0, lt=1.0, default=0.5)

    @field_validator("cycle_loss_weight")
    @classmethod
    def validate_cycle_weight(cls, v: float) -> float:
        """Validate cycle loss weight based on NSM-10 findings."""
        if v > 0.2:
            raise ValueError(
                f"cycle_loss_weight={v} may be too high. Per NSM-10 analysis, "
                "high cycle loss can dominate task loss and cause class imbalance collapse. "
                "Recommended: 0.05-0.15"
            )
        return v


class EpochMetrics(BaseModel):
    """Metrics for a single epoch."""

    epoch: int = Field(ge=0)

    # Losses
    train_loss: float = Field(ge=0.0)
    val_loss: float = Field(ge=0.0)
    task_loss: float = Field(ge=0.0)
    cycle_loss: float = Field(ge=0.0)

    # Task-specific metrics
    val_accuracy: float = Field(ge=0.0, le=1.0)

    # Per-class accuracy (critical per Bug #1)
    accuracy_class_0: Optional[float] = Field(None, ge=0.0, le=1.0)
    accuracy_class_1: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Gradient flow (vanishing gradient detection)
    grad_norm_mean: Optional[float] = Field(None, ge=0.0)
    grad_norm_max: Optional[float] = Field(None, ge=0.0)

    # Training dynamics
    learning_rate: float = Field(gt=0.0)

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def has_class_imbalance(self) -> bool:
        """Detect class imbalance collapse (NSM-10 Bug #1)."""
        if self.accuracy_class_0 is None or self.accuracy_class_1 is None:
            return False

        # If one class has 0% accuracy and other has 100%, model collapsed
        return ((self.accuracy_class_0 < 0.01 and self.accuracy_class_1 > 0.99) or
                (self.accuracy_class_1 < 0.01 and self.accuracy_class_0 > 0.99))

    @computed_field
    @property
    def has_vanishing_gradients(self) -> bool:
        """Check for vanishing gradients."""
        if self.grad_norm_mean is None:
            return False
        return self.grad_norm_mean < 1e-6

    @field_validator("task_loss")
    @classmethod
    def validate_task_loss(cls, v: float, info) -> float:
        """Validate task loss is non-zero (NSM-10 Bug #3)."""
        if v < 1e-8:
            raise ValueError(
                f"task_loss={v:.8f} is essentially zero. Per NSM-10 Bug #3, "
                "this usually indicates a trivial task or dataset architecture bug. "
                "Check that dataset returns complete problems, not fragments."
            )
        return v


class TrainingRunBase(BaseModel):
    """Base class for training run validation."""

    # Identification
    run_id: str = Field(description="Unique identifier (e.g., 'causal_20251020_061321')")
    domain: DomainType
    status: TrainingStatus = Field(default=TrainingStatus.PENDING)

    # Configuration
    dataset_config: DatasetConfig
    hyperparameters: HyperparametersConfig

    # Process tracking
    pid: Optional[int] = Field(None, description="Process ID if running")
    log_path: Optional[Path] = Field(None, description="Path to training log")
    checkpoint_dir: Optional[Path] = Field(None, description="Checkpoint directory")

    # Metrics history
    metrics_history: List[EpochMetrics] = Field(default_factory=list)

    # Final results
    best_val_loss: Optional[float] = Field(None, ge=0.0)
    best_val_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    best_epoch: Optional[int] = Field(None, ge=0)

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Failure tracking
    error_message: Optional[str] = None

    @computed_field
    @property
    def current_epoch(self) -> int:
        """Current training epoch."""
        return len(self.metrics_history)

    @computed_field
    @property
    def is_stuck(self) -> bool:
        """Check if validation accuracy is stuck (not improving)."""
        if len(self.metrics_history) < 10:
            return False

        # Check if last 10 epochs have identical val_accuracy
        recent_accs = [m.val_accuracy for m in self.metrics_history[-10:]]
        return len(set(recent_accs)) == 1  # All identical

    @computed_field
    @property
    def should_early_stop(self) -> bool:
        """Check early stopping criteria."""
        if len(self.metrics_history) < self.hyperparameters.patience:
            return False

        if self.best_epoch is None:
            return False

        # Check if no improvement for `patience` epochs
        epochs_without_improvement = self.current_epoch - self.best_epoch
        return epochs_without_improvement >= self.hyperparameters.patience

    @computed_field
    @property
    def has_converged(self) -> bool:
        """Check if training has converged."""
        if len(self.metrics_history) < 5:
            return False

        # Check if val_loss has stopped decreasing
        recent_losses = [m.val_loss for m in self.metrics_history[-5:]]
        return all(abs(recent_losses[i] - recent_losses[i-1]) < 0.001
                  for i in range(1, len(recent_losses)))

    def add_epoch_metrics(self, metrics: EpochMetrics) -> None:
        """Add epoch metrics and update best values."""
        self.metrics_history.append(metrics)

        # Update best values
        if self.best_val_loss is None or metrics.val_loss < self.best_val_loss:
            self.best_val_loss = metrics.val_loss
            self.best_val_accuracy = metrics.val_accuracy
            self.best_epoch = metrics.epoch

    def get_summary(self) -> Dict[str, Any]:
        """Get human-readable summary."""
        return {
            "run_id": self.run_id,
            "domain": self.domain.value,
            "status": self.status.value,
            "current_epoch": self.current_epoch,
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_loss": self.best_val_loss,
            "is_stuck": self.is_stuck,
            "should_early_stop": self.should_early_stop,
            "dataset_balanced": self.dataset_config.is_balanced,
        }


class CausalTrainingRun(TrainingRunBase):
    """Causal domain training run with domain-specific validation."""

    domain: DomainType = Field(default=DomainType.CAUSAL, frozen=True)

    # Domain-specific metrics
    counterfactual_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    intervention_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_causal_config(self) -> "CausalTrainingRun":
        """Validate causal-specific configuration."""
        # Check domain params
        params = self.dataset_config.domain_params

        if "num_scenarios" in params:
            # Per NSM-10 standardization
            if not (2400 <= params["num_scenarios"] <= 2600):
                raise ValueError(
                    f"num_scenarios={params['num_scenarios']} outside standardized range. "
                    "Should be ~2500 for 2,000 train examples."
                )

        # Per NSM-10 Bug #1 fix: effectiveness range should be [0.2, 0.9]
        if "effectiveness_range" in params:
            eff_range = params["effectiveness_range"]
            if eff_range != [0.2, 0.9]:
                raise ValueError(
                    f"effectiveness_range={eff_range} may cause label imbalance. "
                    "Per NSM-10 Bug #1 fix, should be [0.2, 0.9]."
                )

        # Pool ratio learned from NSM-10
        if self.hyperparameters.pool_ratio != 0.25:
            raise ValueError(
                f"pool_ratio={self.hyperparameters.pool_ratio} suboptimal for Causal. "
                "Learned optimal: 0.25"
            )

        return self


class KnowledgeGraphTrainingRun(TrainingRunBase):
    """Knowledge graph domain training run with domain-specific validation."""

    domain: DomainType = Field(default=DomainType.KNOWLEDGE_GRAPH, frozen=True)

    # Domain-specific metrics
    hits_at_10: Optional[float] = Field(None, ge=0.0, le=1.0)
    mrr: Optional[float] = Field(None, ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    analogical_reasoning_acc: Optional[float] = Field(None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_kg_config(self) -> "KnowledgeGraphTrainingRun":
        """Validate KG-specific configuration."""
        params = self.dataset_config.domain_params

        if "num_entities" in params and "num_triples" in params:
            # Per NSM-10 standardization
            if not (2400 <= params["num_triples"] <= 2600):
                raise ValueError(
                    f"num_triples={params['num_triples']} outside standardized range. "
                    "Should be ~2500 for 2,000 train examples."
                )

            if not (180 <= params["num_entities"] <= 220):
                raise ValueError(
                    f"num_entities={params['num_entities']} outside learned range. "
                    "Should be ~200 per NSM-10 standardization."
                )

        # Per NSM-10 Bug #1 fix: must use negative sampling
        if "use_negative_sampling" in params and not params["use_negative_sampling"]:
            raise ValueError(
                "KG domain MUST use negative sampling. "
                "Per NSM-10 Bug #1, without it all labels are positive."
            )

        # Pool ratio learned from NSM-10
        if self.hyperparameters.pool_ratio != 0.13:
            raise ValueError(
                f"pool_ratio={self.hyperparameters.pool_ratio} suboptimal for KG. "
                "Learned optimal: 0.13"
            )

        return self

    @computed_field
    @property
    def has_task_mismatch(self) -> bool:
        """Detect task mismatch (NSM-10: good ranking, poor binary classification)."""
        if self.hits_at_10 is None or self.best_val_accuracy is None:
            return False

        # If Hits@10 is excellent but val_accuracy is poor, task mismatch
        return self.hits_at_10 > 0.9 and self.best_val_accuracy < 0.5


class PlanningTrainingRun(TrainingRunBase):
    """Planning domain training run with domain-specific validation."""

    domain: DomainType = Field(default=DomainType.PLANNING, frozen=True)

    # Domain-specific metrics
    goal_achievement_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    temporal_ordering_acc: Optional[float] = Field(None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_planning_config(self) -> "PlanningTrainingRun":
        """Validate planning-specific configuration."""
        params = self.dataset_config.domain_params

        if "num_problems" in params:
            # Per NSM-10 standardization
            if not (2700 <= params["num_problems"] <= 2900):
                raise ValueError(
                    f"num_problems={params['num_problems']} outside standardized range. "
                    "Should be ~2858 for 2,000 train examples."
                )

        # CRITICAL: Per NSM-10 Bug #3, dataset MUST return complete problems
        if "returns_complete_problems" in params and not params["returns_complete_problems"]:
            raise ValueError(
                "Planning dataset MUST return complete problems as graphs. "
                "Per NSM-10 Bug #3, returning individual triples causes zero task loss."
            )

        # Per NSM-10 Bug #1 fix: validity threshold should be 50%
        if "validity_threshold" in params and params["validity_threshold"] != 0.5:
            raise ValueError(
                f"validity_threshold={params['validity_threshold']} causes label imbalance. "
                "Per NSM-10 Bug #1 fix, should be 0.5 (50%)."
            )

        # Pool ratio learned from NSM-10
        if self.hyperparameters.pool_ratio != 0.5:
            raise ValueError(
                f"pool_ratio={self.hyperparameters.pool_ratio} suboptimal for Planning. "
                "Learned optimal: 0.5"
            )

        return self

    @field_validator("metrics_history")
    @classmethod
    def validate_no_zero_task_loss(cls, v: List[EpochMetrics]) -> List[EpochMetrics]:
        """Ensure task loss never zero (NSM-10 Bug #3 detector)."""
        for metrics in v:
            if metrics.task_loss < 1e-8:
                raise ValueError(
                    f"Epoch {metrics.epoch}: task_loss={metrics.task_loss:.8f} is zero. "
                    "Per NSM-10 Bug #3, this indicates dataset architecture bug. "
                    "Check that PlanningTripleDataset returns complete problems."
                )
        return v


# Type alias for all training runs
TrainingRun = Union[CausalTrainingRun, KnowledgeGraphTrainingRun, PlanningTrainingRun]
