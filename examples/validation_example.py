"""
Example usage of the NSM training validation framework.

Demonstrates:
1. Creating validated training runs
2. Running pre-flight checks
3. Adding epoch metrics with checkpoint validation
4. Experiment tracking via JSONL
5. Detecting NSM-10 bugs automatically
"""

from datetime import datetime
from pathlib import Path

from nsm.validation import (
    CausalTrainingRun,
    DatasetConfig,
    DomainType,
    EpochMetrics,
    HyperparametersConfig,
    TestViewController,
    TrainingStatus,
)


def example_1_valid_causal_run():
    """Example 1: Create a valid Causal training run."""
    print("\n" + "="*60)
    print("Example 1: Valid Causal Training Run")
    print("="*60)

    # Initialize controller
    controller = TestViewController(
        experiment_log_path=Path("experiments/training_log.jsonl")
    )

    # Create dataset config (matches NSM-10 standardization)
    dataset_config = DatasetConfig(
        domain=DomainType.CAUSAL,
        split="train",
        total_size=2500,
        train_size=2000,
        val_size=500,
        label_balance_class_0=0.50,
        label_balance_class_1=0.50,
        domain_params={
            "num_scenarios": 2500,
            "effectiveness_range": [0.2, 0.9],  # Per NSM-10 Bug #1 fix
        }
    )

    # Create hyperparameters
    hyperparameters = HyperparametersConfig(
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        cycle_loss_weight=0.1,
        patience=20,
        pool_ratio=0.25,  # Learned optimal for Causal
    )

    # Create training run (with automatic validation)
    try:
        training_run = controller.create_training_run(
            run_id="causal_valid_20251020",
            domain=DomainType.CAUSAL,
            dataset_config=dataset_config,
            hyperparameters=hyperparameters,
        )
        print("✅ Training run created successfully!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return

    # Run pre-flight checks
    preflight_results = controller.run_preflight_checks(training_run)
    print(controller.format_test_results(preflight_results))

    if preflight_results["all_passed"]:
        print("✅ All pre-flight checks passed! Safe to start training.")
    else:
        print("❌ Pre-flight checks failed. Fix issues before training.")


def example_2_detect_bug1_degenerate_labels():
    """Example 2: Detect NSM-10 Bug #1 (degenerate labels)."""
    print("\n" + "="*60)
    print("Example 2: Detect Bug #1 (Degenerate Labels)")
    print("="*60)

    controller = TestViewController(
        experiment_log_path=Path("experiments/training_log.jsonl")
    )

    # Create dataset with imbalanced labels (Bug #1)
    try:
        dataset_config = DatasetConfig(
            domain=DomainType.CAUSAL,
            split="train",
            total_size=2500,
            train_size=2000,
            val_size=500,
            label_balance_class_0=0.0,  # All positive! (Bug #1)
            label_balance_class_1=1.0,
            domain_params={}
        )
        print("❌ Should have caught degenerate labels!")
    except Exception as e:
        print(f"✅ Caught Bug #1: {e}")


def example_3_detect_bug2_dataset_size():
    """Example 3: Detect NSM-10 Bug #2 (inconsistent dataset size)."""
    print("\n" + "="*60)
    print("Example 3: Detect Bug #2 (Inconsistent Dataset Size)")
    print("="*60)

    controller = TestViewController(
        experiment_log_path=Path("experiments/training_log.jsonl")
    )

    # Create dataset with non-standard size (Bug #2)
    try:
        dataset_config = DatasetConfig(
            domain=DomainType.KNOWLEDGE_GRAPH,
            split="train",
            total_size=500,
            train_size=400,  # Too small! Should be ~2000
            val_size=100,
            label_balance_class_0=0.50,
            label_balance_class_1=0.50,
            domain_params={}
        )
        print("❌ Should have caught non-standard size!")
    except Exception as e:
        print(f"✅ Caught Bug #2: {e}")


def example_4_detect_bug3_zero_task_loss():
    """Example 4: Detect NSM-10 Bug #3 (zero task loss)."""
    print("\n" + "="*60)
    print("Example 4: Detect Bug #3 (Zero Task Loss)")
    print("="*60)

    controller = TestViewController(
        experiment_log_path=Path("experiments/training_log.jsonl")
    )

    # Create valid training run
    dataset_config = DatasetConfig(
        domain=DomainType.PLANNING,
        split="train",
        total_size=2858,
        train_size=2000,
        val_size=429,
        label_balance_class_0=0.50,
        label_balance_class_1=0.50,
        domain_params={
            "num_problems": 2858,
            "returns_complete_problems": True,
            "validity_threshold": 0.5,
        }
    )

    hyperparameters = HyperparametersConfig(
        epochs=100,
        batch_size=32,
        pool_ratio=0.5,  # Planning optimal
    )

    training_run = controller.create_training_run(
        run_id="planning_bug3_20251020",
        domain=DomainType.PLANNING,
        dataset_config=dataset_config,
        hyperparameters=hyperparameters,
    )

    # Simulate epoch with zero task loss (Bug #3)
    try:
        metrics = EpochMetrics(
            epoch=1,
            train_loss=0.0871,
            val_loss=0.0872,
            task_loss=0.0000,  # Zero! (Bug #3)
            cycle_loss=0.8710,
            val_accuracy=1.0,
            learning_rate=0.001,
        )
        training_run.add_epoch_metrics(metrics)
        print("❌ Should have caught zero task loss!")
    except Exception as e:
        print(f"✅ Caught Bug #3: {e}")


def example_5_detect_class_imbalance_collapse():
    """Example 5: Detect class imbalance collapse (NSM-10 Causal issue)."""
    print("\n" + "="*60)
    print("Example 5: Detect Class Imbalance Collapse")
    print("="*60)

    controller = TestViewController(
        experiment_log_path=Path("experiments/training_log.jsonl")
    )

    # Create valid Causal run
    dataset_config = DatasetConfig(
        domain=DomainType.CAUSAL,
        split="train",
        total_size=2500,
        train_size=2000,
        val_size=500,
        label_balance_class_0=0.50,
        label_balance_class_1=0.50,
        domain_params={
            "num_scenarios": 2500,
            "effectiveness_range": [0.2, 0.9],
        }
    )

    hyperparameters = HyperparametersConfig(
        epochs=100,
        batch_size=32,
        pool_ratio=0.25,
    )

    training_run = controller.create_training_run(
        run_id="causal_collapse_20251020",
        domain=DomainType.CAUSAL,
        dataset_config=dataset_config,
        hyperparameters=hyperparameters,
    )

    # Add epoch metrics showing collapse
    metrics = EpochMetrics(
        epoch=68,
        train_loss=0.7261,
        val_loss=0.7269,
        task_loss=0.6837,
        cycle_loss=0.7278,
        val_accuracy=0.6143,
        accuracy_class_0=0.0,  # Never predicts class 0!
        accuracy_class_1=1.0,  # Always predicts class 1!
        learning_rate=0.001,
    )

    training_run.add_epoch_metrics(metrics)

    # Run checkpoint tests
    checkpoint_results = controller.run_checkpoint_checks(training_run)
    print(controller.format_test_results(checkpoint_results))

    if checkpoint_results["critical_failures"]:
        print("✅ Detected class imbalance collapse!")


def example_6_experiment_tracking():
    """Example 6: Experiment tracking and summary."""
    print("\n" + "="*60)
    print("Example 6: Experiment Tracking")
    print("="*60)

    controller = TestViewController(
        experiment_log_path=Path("experiments/training_log.jsonl")
    )

    # Create and log multiple runs
    for domain in [DomainType.CAUSAL, DomainType.KNOWLEDGE_GRAPH, DomainType.PLANNING]:
        # Set correct pool ratio for each domain
        pool_ratios = {
            DomainType.CAUSAL: 0.25,
            DomainType.KNOWLEDGE_GRAPH: 0.13,
            DomainType.PLANNING: 0.5,
        }

        dataset_config = DatasetConfig(
            domain=domain,
            split="train",
            total_size=2500,
            train_size=2000,
            val_size=500,
            label_balance_class_0=0.50,
            label_balance_class_1=0.50,
            domain_params={}
        )

        hyperparameters = HyperparametersConfig(
            epochs=100,
            batch_size=32,
            pool_ratio=pool_ratios[domain],
        )

        training_run = controller.create_training_run(
            run_id=f"{domain.value}_example_20251020",
            domain=domain,
            dataset_config=dataset_config,
            hyperparameters=hyperparameters,
        )

        training_run.status = TrainingStatus.COMPLETED
        training_run.best_val_accuracy = 0.55 + (hash(domain.value) % 10) / 100

        # Log to JSONL
        controller.log_experiment(training_run)
        print(f"Logged experiment: {training_run.run_id}")

    # Get summary
    summary = controller.get_experiment_summary()
    print("\nExperiment Summary:")
    print(f"  Total experiments: {summary['total_experiments']}")
    print(f"  By domain: {summary['by_domain']}")
    print(f"  Best accuracy by domain: {summary['best_accuracy_by_domain']}")
    print(f"  Log path: {summary['log_path']}")


if __name__ == "__main__":
    # Run all examples
    example_1_valid_causal_run()
    example_2_detect_bug1_degenerate_labels()
    example_3_detect_bug2_dataset_size()
    example_4_detect_bug3_zero_task_loss()
    example_5_detect_class_imbalance_collapse()
    example_6_experiment_tracking()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
