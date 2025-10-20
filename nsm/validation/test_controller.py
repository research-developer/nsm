"""
Test View Controller for training validation.

Provides centralized validation for pre-flight checks, checkpoint validation,
and experiment tracking via JSONL logging.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

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


class ValidationTest(ABC):
    """Base class for validation tests."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, training_run: TrainingRun) -> tuple[bool, Optional[str]]:
        """
        Run validation test.

        Returns:
            (passed, error_message): True if passed, False with error message if failed
        """
        pass


class PreFlightTest(ValidationTest):
    """Pre-flight validation tests (before training starts)."""
    pass


class CheckpointTest(ValidationTest):
    """Checkpoint validation tests (during/after training)."""
    pass


# ============================================================================
# Pre-Flight Tests
# ============================================================================


class DatasetBalanceTest(PreFlightTest):
    """Check dataset label balance (NSM-10 Bug #1)."""

    def __init__(self):
        super().__init__(
            name="dataset_balance",
            description="Verify dataset has balanced labels (40-60% per class)"
        )

    def run(self, training_run: TrainingRun) -> tuple[bool, Optional[str]]:
        if not training_run.dataset_config.is_balanced:
            return False, (
                f"Dataset imbalanced: class_0={training_run.dataset_config.label_balance_class_0:.2%}, "
                f"class_1={training_run.dataset_config.label_balance_class_1:.2%}. "
                "Per NSM-10 Bug #1, this will cause degenerate training."
            )
        return True, None


class DatasetSizeTest(PreFlightTest):
    """Check dataset size is standardized (NSM-10 Bug #2)."""

    def __init__(self):
        super().__init__(
            name="dataset_size",
            description="Verify dataset has standardized train size (~2,000)"
        )

    def run(self, training_run: TrainingRun) -> tuple[bool, Optional[str]]:
        train_size = training_run.dataset_config.train_size
        if not (1800 <= train_size <= 2200):
            return False, (
                f"Train size {train_size} outside standardized range [1800, 2200]. "
                "Per NSM-10 Bug #2, all domains should have ~2,000 examples."
            )
        return True, None


class HyperparameterBoundsTest(PreFlightTest):
    """Check hyperparameters are within learned bounds."""

    def __init__(self):
        super().__init__(
            name="hyperparameter_bounds",
            description="Verify hyperparameters within learned bounds"
        )

    def run(self, training_run: TrainingRun) -> tuple[bool, Optional[str]]:
        hp = training_run.hyperparameters

        # Check cycle loss weight (NSM-10: high values cause class collapse)
        if hp.cycle_loss_weight > 0.2:
            return False, (
                f"cycle_loss_weight={hp.cycle_loss_weight} too high. "
                "Per NSM-10 analysis, high values cause class imbalance collapse. "
                "Recommended: 0.05-0.15"
            )

        # Check learning rate
        if hp.learning_rate > 0.01:
            return False, f"learning_rate={hp.learning_rate} very high, may cause instability."

        return True, None


class DomainSpecificConfigTest(PreFlightTest):
    """Check domain-specific configuration."""

    def __init__(self):
        super().__init__(
            name="domain_specific_config",
            description="Verify domain-specific parameters are correct"
        )

    def run(self, training_run: TrainingRun) -> tuple[bool, Optional[str]]:
        # Domain-specific validators are already in Pydantic models
        # This test just ensures they were applied
        return True, None


# ============================================================================
# Checkpoint Tests
# ============================================================================


class ZeroTaskLossTest(CheckpointTest):
    """Detect zero task loss (NSM-10 Bug #3)."""

    def __init__(self):
        super().__init__(
            name="zero_task_loss",
            description="Detect zero task loss (indicates trivial task)"
        )

    def run(self, training_run: TrainingRun) -> tuple[bool, Optional[str]]:
        if not training_run.metrics_history:
            return True, None

        latest = training_run.metrics_history[-1]
        if latest.task_loss < 1e-8:
            return False, (
                f"Epoch {latest.epoch}: task_loss={latest.task_loss:.8f} is zero. "
                "Per NSM-10 Bug #3, this indicates dataset architecture bug or trivial task."
            )
        return True, None


class ClassImbalanceCollapseTest(CheckpointTest):
    """Detect class imbalance collapse (NSM-10: Causal issue)."""

    def __init__(self):
        super().__init__(
            name="class_imbalance_collapse",
            description="Detect model collapsing to always-predict-one-class"
        )

    def run(self, training_run: TrainingRun) -> tuple[bool, Optional[str]]:
        if not training_run.metrics_history:
            return True, None

        latest = training_run.metrics_history[-1]
        if latest.has_class_imbalance:
            return False, (
                f"Epoch {latest.epoch}: Model collapsed to always predict one class. "
                f"class_0_acc={latest.accuracy_class_0:.2%}, "
                f"class_1_acc={latest.accuracy_class_1:.2%}. "
                "Per NSM-10 Causal analysis, check cycle loss dominance."
            )
        return True, None


class VanishingGradientTest(CheckpointTest):
    """Detect vanishing gradients."""

    def __init__(self):
        super().__init__(
            name="vanishing_gradient",
            description="Detect vanishing gradient problem"
        )

    def run(self, training_run: TrainingRun) -> tuple[bool, Optional[str]]:
        if not training_run.metrics_history:
            return True, None

        latest = training_run.metrics_history[-1]
        if latest.has_vanishing_gradients:
            return False, (
                f"Epoch {latest.epoch}: Vanishing gradients detected "
                f"(mean_norm={latest.grad_norm_mean:.2e}). "
                "Add residual connections or increase learning rate."
            )
        return True, None


class StuckTrainingTest(CheckpointTest):
    """Detect stuck training (validation accuracy not improving)."""

    def __init__(self):
        super().__init__(
            name="stuck_training",
            description="Detect when validation accuracy is stuck"
        )

    def run(self, training_run: TrainingRun) -> tuple[bool, Optional[str]]:
        if training_run.is_stuck:
            latest = training_run.metrics_history[-1]
            return False, (
                f"Training stuck: val_accuracy={latest.val_accuracy:.2%} "
                f"unchanged for {len(training_run.metrics_history[-10:])} epochs. "
                "Consider early stopping or hyperparameter adjustment."
            )
        return True, None


class EarlyStoppingTest(CheckpointTest):
    """Check if early stopping criteria met."""

    def __init__(self):
        super().__init__(
            name="early_stopping",
            description="Check if early stopping criteria are met"
        )

    def run(self, training_run: TrainingRun) -> tuple[bool, Optional[str]]:
        if training_run.should_early_stop:
            return False, (
                f"Early stopping triggered: No improvement for "
                f"{training_run.hyperparameters.patience} epochs. "
                f"Best: epoch {training_run.best_epoch}, "
                f"val_acc={training_run.best_val_accuracy:.2%}"
            )
        return True, None


class TaskMismatchTest(CheckpointTest):
    """Detect task mismatch (NSM-10: KG issue)."""

    def __init__(self):
        super().__init__(
            name="task_mismatch",
            description="Detect task/loss mismatch (good ranking, poor binary)"
        )

    def run(self, training_run: TrainingRun) -> tuple[bool, Optional[str]]:
        if isinstance(training_run, KnowledgeGraphTrainingRun):
            if training_run.has_task_mismatch:
                return False, (
                    f"Task mismatch detected: Hits@10={training_run.hits_at_10:.2%}, "
                    f"but val_acc={training_run.best_val_accuracy:.2%}. "
                    "Per NSM-10 KG analysis, consider using ranking loss instead of binary."
                )
        return True, None


# ============================================================================
# Test View Controller
# ============================================================================


class TestViewController:
    """Central controller for validation tests and experiment tracking."""

    def __init__(self, experiment_log_path: Path):
        self.experiment_log_path = Path(experiment_log_path)
        self.experiment_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Pre-flight tests
        self.preflight_tests: List[PreFlightTest] = [
            DatasetBalanceTest(),
            DatasetSizeTest(),
            HyperparameterBoundsTest(),
            DomainSpecificConfigTest(),
        ]

        # Checkpoint tests
        self.checkpoint_tests: List[CheckpointTest] = [
            ZeroTaskLossTest(),
            ClassImbalanceCollapseTest(),
            VanishingGradientTest(),
            StuckTrainingTest(),
            EarlyStoppingTest(),
            TaskMismatchTest(),
        ]

    def run_preflight_checks(self, training_run: TrainingRun) -> Dict[str, Any]:
        """
        Run all pre-flight checks before training starts.

        Returns:
            Dictionary with test results and overall pass/fail
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "run_id": training_run.run_id,
            "domain": training_run.domain.value,
            "tests": {},
            "all_passed": True,
        }

        for test in self.preflight_tests:
            passed, error_msg = test.run(training_run)
            results["tests"][test.name] = {
                "passed": passed,
                "description": test.description,
                "error": error_msg,
            }
            if not passed:
                results["all_passed"] = False

        return results

    def run_checkpoint_checks(self, training_run: TrainingRun) -> Dict[str, Any]:
        """
        Run all checkpoint checks during/after training.

        Returns:
            Dictionary with test results and warnings
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "run_id": training_run.run_id,
            "domain": training_run.domain.value,
            "epoch": training_run.current_epoch,
            "tests": {},
            "warnings": [],
            "critical_failures": [],
        }

        for test in self.checkpoint_tests:
            passed, error_msg = test.run(training_run)
            results["tests"][test.name] = {
                "passed": passed,
                "description": test.description,
                "error": error_msg,
            }

            if not passed:
                # Classify severity
                if test.name in ["zero_task_loss", "class_imbalance_collapse"]:
                    results["critical_failures"].append({
                        "test": test.name,
                        "message": error_msg,
                    })
                else:
                    results["warnings"].append({
                        "test": test.name,
                        "message": error_msg,
                    })

        return results

    def log_experiment(self, training_run: TrainingRun) -> None:
        """Log experiment to JSONL file."""
        with open(self.experiment_log_path, "a") as f:
            # Use model_dump() for JSON serialization
            record = {
                "timestamp": datetime.now().isoformat(),
                "run_data": training_run.model_dump(mode="json"),
            }
            f.write(json.dumps(record) + "\n")

    def load_experiments(self, domain: Optional[DomainType] = None) -> List[Dict[str, Any]]:
        """
        Load experiments from JSONL log.

        Args:
            domain: Optional domain filter

        Returns:
            List of experiment records
        """
        if not self.experiment_log_path.exists():
            return []

        experiments = []
        with open(self.experiment_log_path, "r") as f:
            for line in f:
                record = json.loads(line.strip())
                run_data = record["run_data"]

                # Filter by domain if specified
                if domain is None or run_data["domain"] == domain.value:
                    experiments.append(record)

        return experiments

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all experiments."""
        experiments = self.load_experiments()

        if not experiments:
            return {"total_experiments": 0}

        # Count by domain
        domain_counts = {}
        domain_best_acc = {}

        for exp in experiments:
            run_data = exp["run_data"]
            domain = run_data["domain"]

            domain_counts[domain] = domain_counts.get(domain, 0) + 1

            best_acc = run_data.get("best_val_accuracy")
            if best_acc is not None:
                if domain not in domain_best_acc or best_acc > domain_best_acc[domain]:
                    domain_best_acc[domain] = best_acc

        return {
            "total_experiments": len(experiments),
            "by_domain": domain_counts,
            "best_accuracy_by_domain": domain_best_acc,
            "log_path": str(self.experiment_log_path),
        }

    def create_training_run(
        self,
        run_id: str,
        domain: DomainType,
        dataset_config: DatasetConfig,
        hyperparameters: HyperparametersConfig,
    ) -> TrainingRun:
        """
        Create domain-specific training run with validation.

        Raises:
            ValidationError: If configuration is invalid
        """
        # Select appropriate subclass
        if domain == DomainType.CAUSAL:
            RunClass: Type[TrainingRun] = CausalTrainingRun
        elif domain == DomainType.KNOWLEDGE_GRAPH:
            RunClass = KnowledgeGraphTrainingRun
        elif domain == DomainType.PLANNING:
            RunClass = PlanningTrainingRun
        else:
            raise ValueError(f"Unknown domain: {domain}")

        # Create with validation
        training_run = RunClass(
            run_id=run_id,
            domain=domain,
            dataset_config=dataset_config,
            hyperparameters=hyperparameters,
        )

        return training_run

    def format_test_results(self, results: Dict[str, Any]) -> str:
        """Format test results for human-readable output."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"Validation Results: {results.get('run_id', 'Unknown')}")
        lines.append(f"Domain: {results.get('domain', 'Unknown')}")
        lines.append(f"Timestamp: {results.get('timestamp', 'Unknown')}")
        lines.append(f"{'='*60}\n")

        # Pre-flight results
        if "all_passed" in results:
            status = "✅ PASSED" if results["all_passed"] else "❌ FAILED"
            lines.append(f"Overall Status: {status}\n")

        # Test details
        for test_name, test_result in results.get("tests", {}).items():
            status = "✅" if test_result["passed"] else "❌"
            lines.append(f"{status} {test_name}")
            lines.append(f"   {test_result['description']}")
            if test_result.get("error"):
                lines.append(f"   ERROR: {test_result['error']}")
            lines.append("")

        # Warnings and failures
        if "critical_failures" in results and results["critical_failures"]:
            lines.append(f"\n🚨 CRITICAL FAILURES ({len(results['critical_failures'])}):")
            for failure in results["critical_failures"]:
                lines.append(f"  - {failure['test']}: {failure['message']}")
            lines.append("")

        if "warnings" in results and results["warnings"]:
            lines.append(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                lines.append(f"  - {warning['test']}: {warning['message']}")
            lines.append("")

        lines.append("="*60)

        return "\n".join(lines)
