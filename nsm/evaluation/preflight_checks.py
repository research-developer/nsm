"""
Preflight checks for NSM training to catch common issues early.

This module validates critical training prerequisites before expensive
training runs begin, catching issues that led to NSM-31 failures.

Usage:
    from nsm.evaluation.preflight_checks import run_preflight_checks

    run_preflight_checks(
        dataset=train_dataset,
        model=model,
        cycle_loss_weight=0.01,
        learning_rate=5e-4
    )
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List
import warnings


class PreflightCheckError(Exception):
    """Raised when a critical preflight check fails."""
    pass


class PreflightCheckWarning(UserWarning):
    """Issued when a non-critical preflight check fails."""
    pass


def check_dataset_balance(
    dataset: Dataset,
    max_samples: int = 1000,
    balance_threshold: float = 0.4
) -> Dict[str, Any]:
    """
    Check dataset class balance to prevent class collapse (NSM-31).

    Args:
        dataset: PyTorch Dataset with __getitem__ returning (data, label)
        max_samples: Maximum samples to check (for large datasets)
        balance_threshold: Minimum proportion for minority class (0.4 = 40%)

    Returns:
        dict: Class distribution statistics

    Raises:
        PreflightCheckError: If severe imbalance detected
    """
    print("🔍 Checking dataset class balance...")

    # Sample labels
    num_samples = min(len(dataset), max_samples)
    labels = []

    for i in range(num_samples):
        try:
            _, label = dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            labels.append(label)
        except Exception as e:
            warnings.warn(
                f"Failed to load sample {i}: {e}",
                PreflightCheckWarning
            )
            continue

    if not labels:
        raise PreflightCheckError("Could not load any samples from dataset!")

    # Count classes
    unique_labels = set(labels)
    class_counts = {label: labels.count(label) for label in unique_labels}
    total = len(labels)

    # Calculate proportions
    class_props = {label: count / total for label, count in class_counts.items()}

    print(f"  Total samples checked: {total}")
    print(f"  Class distribution:")
    for label in sorted(class_props.keys()):
        print(f"    Class {label}: {class_counts[label]} ({100*class_props[label]:.1f}%)")

    # Check for severe imbalance
    min_prop = min(class_props.values())
    if min_prop < balance_threshold:
        raise PreflightCheckError(
            f"Severe class imbalance detected! Minority class: {100*min_prop:.1f}% "
            f"(threshold: {100*balance_threshold:.1f}%)\n"
            f"This will cause class collapse during training (NSM-31).\n"
            f"Fix: Use class_weights in trainer or balance dataset."
        )
    elif min_prop < 0.45:
        warnings.warn(
            f"Moderate class imbalance: minority class {100*min_prop:.1f}%. "
            f"Consider using class_weights to prevent collapse.",
            PreflightCheckWarning
        )
    else:
        print(f"  ✅ Dataset is well-balanced (minority: {100*min_prop:.1f}%)")

    return {
        'class_counts': class_counts,
        'class_proportions': class_props,
        'is_balanced': min_prop >= 0.45,
        'minority_proportion': min_prop
    }


def check_cycle_loss_weight(
    cycle_loss_weight: float,
    max_recommended: float = 0.05,
    max_safe: float = 0.1
) -> Dict[str, Any]:
    """
    Check cycle loss weight to prevent gradient dominance (NSM-31).

    Args:
        cycle_loss_weight: Cycle consistency loss weight
        max_recommended: Recommended maximum (0.05 from NSM-31 analysis)
        max_safe: Safe maximum before critical issues (0.1 caused failures)

    Returns:
        dict: Validation results

    Raises:
        PreflightCheckError: If weight is dangerously high
    """
    print(f"🔍 Checking cycle loss weight ({cycle_loss_weight})...")

    if cycle_loss_weight > max_safe:
        raise PreflightCheckError(
            f"Cycle loss weight {cycle_loss_weight} is too high!\n"
            f"NSM-31 analysis showed weight 0.1 caused:\n"
            f"  - Cycle loss dominating task gradient (0.1 × 0.98 = 0.098)\n"
            f"  - Class collapse (model always predicts one class)\n"
            f"  - Poor accuracy (40-53% across all domains)\n"
            f"Recommended: {max_recommended}, Maximum safe: {max_safe}"
        )
    elif cycle_loss_weight > max_recommended:
        warnings.warn(
            f"Cycle loss weight {cycle_loss_weight} exceeds recommended {max_recommended}. "
            f"This may cause cycle loss to dominate task learning (NSM-31).",
            PreflightCheckWarning
        )
    else:
        print(f"  ✅ Cycle loss weight is safe ({cycle_loss_weight} ≤ {max_recommended})")

    return {
        'cycle_loss_weight': cycle_loss_weight,
        'is_safe': cycle_loss_weight <= max_recommended,
        'is_critical': cycle_loss_weight > max_safe
    }


def check_learning_rate(
    learning_rate: float,
    max_recommended: float = 5e-4,
    max_safe: float = 1e-3
) -> Dict[str, Any]:
    """
    Check learning rate for training stability (NSM-31).

    Args:
        learning_rate: Optimizer learning rate
        max_recommended: Recommended maximum (5e-4 from NSM-31)
        max_safe: Safe maximum (1e-3 caused instability)

    Returns:
        dict: Validation results

    Raises:
        PreflightCheckError: If learning rate is dangerously high
    """
    print(f"🔍 Checking learning rate ({learning_rate:.2e})...")

    if learning_rate > max_safe:
        raise PreflightCheckError(
            f"Learning rate {learning_rate:.2e} is too high!\n"
            f"NSM-31 analysis showed LR {max_safe:.2e} caused:\n"
            f"  - Unstable training with complex hierarchical model\n"
            f"  - Cycle loss not converging\n"
            f"  - High reconstruction error (0.78-0.98 vs target <0.2)\n"
            f"Recommended: {max_recommended:.2e}, Maximum safe: {max_safe:.2e}"
        )
    elif learning_rate > max_recommended:
        warnings.warn(
            f"Learning rate {learning_rate:.2e} exceeds recommended {max_recommended:.2e}. "
            f"May cause training instability with hierarchical architecture (NSM-31).",
            PreflightCheckWarning
        )
    else:
        print(f"  ✅ Learning rate is safe ({learning_rate:.2e} ≤ {max_recommended:.2e})")

    return {
        'learning_rate': learning_rate,
        'is_safe': learning_rate <= max_recommended,
        'is_critical': learning_rate > max_safe
    }


def check_pyg_extensions() -> Dict[str, Any]:
    """
    Check PyTorch Geometric extensions are working (NSM-31).

    Returns:
        dict: Extension availability status

    Note:
        NSM-31 investigation showed that torch-scatter/torch-sparse
        warnings are non-critical - PyG has pure PyTorch fallbacks
        that work correctly. This check verifies SAGPooling works.
    """
    print("🔍 Checking PyTorch Geometric extensions...")

    try:
        from torch_geometric.nn import SAGPooling

        # Test SAGPooling
        pool = SAGPooling(in_channels=32, ratio=0.5)
        x = torch.randn(10, 32)
        edge_index = torch.randint(0, 10, (2, 20))
        batch = torch.zeros(10, dtype=torch.long)

        x_pooled, _, _, _, _, _ = pool(x, edge_index, batch=batch)

        print(f"  ✅ SAGPooling working ({x.size(0)} → {x_pooled.size(0)} nodes)")

        return {
            'pyg_available': True,
            'sagpooling_works': True,
            'pooling_ratio': x_pooled.size(0) / x.size(0)
        }

    except Exception as e:
        raise PreflightCheckError(
            f"PyTorch Geometric pooling failed: {e}\n"
            f"WHY/WHAT operations require functional pooling.\n"
            f"Try reinstalling: pip install torch-geometric torch-scatter torch-sparse"
        )


def check_model_architecture(
    model: nn.Module,
    expected_components: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Check model has required NSM components.

    Args:
        model: NSM model instance
        expected_components: List of required component names

    Returns:
        dict: Architecture validation results
    """
    print("🔍 Checking model architecture...")

    if expected_components is None:
        expected_components = ['layer_1_2', 'classifier']

    missing = []
    for component in expected_components:
        if not hasattr(model, component):
            missing.append(component)

    if missing:
        warnings.warn(
            f"Model missing expected components: {missing}. "
            f"May not be a valid NSM model.",
            PreflightCheckWarning
        )
    else:
        print(f"  ✅ Model has all expected components")

    # Check for common issues
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")

    if num_trainable == 0:
        raise PreflightCheckError("Model has no trainable parameters!")

    return {
        'has_all_components': len(missing) == 0,
        'missing_components': missing,
        'num_parameters': num_params,
        'num_trainable': num_trainable
    }


def check_class_weights(
    class_weights: Optional[torch.Tensor],
    dataset_balance: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check if class weights are provided when dataset is imbalanced.

    Args:
        class_weights: Optional class weights tensor
        dataset_balance: Results from check_dataset_balance()

    Returns:
        dict: Validation results
    """
    print("🔍 Checking class weighting strategy...")

    is_balanced = dataset_balance.get('is_balanced', True)
    minority_prop = dataset_balance.get('minority_proportion', 0.5)

    if not is_balanced and class_weights is None:
        warnings.warn(
            f"Dataset is imbalanced (minority: {100*minority_prop:.1f}%) "
            f"but no class_weights provided!\n"
            f"This increases risk of class collapse (NSM-31).\n"
            f"Recommended: Pass class_weights to NSMTrainer.",
            PreflightCheckWarning
        )
        return {'has_weights': False, 'recommended': True}
    elif class_weights is not None:
        print(f"  ✅ Class weights provided: {class_weights.tolist()}")
        return {'has_weights': True, 'weights': class_weights.tolist()}
    else:
        print(f"  ✅ Dataset balanced, class weights not required")
        return {'has_weights': False, 'recommended': False}


def run_preflight_checks(
    dataset: Optional[Dataset] = None,
    model: Optional[nn.Module] = None,
    cycle_loss_weight: float = 0.1,
    learning_rate: float = 1e-3,
    class_weights: Optional[torch.Tensor] = None,
    strict: bool = True,
    check_processes: bool = True
) -> Dict[str, Any]:
    """
    Run all preflight checks before training.

    Args:
        dataset: Training dataset
        model: NSM model
        cycle_loss_weight: Cycle consistency loss weight
        learning_rate: Optimizer learning rate
        class_weights: Optional class weights for loss
        strict: If True, raise errors on failures. If False, only warn.
        check_processes: If True, check for orphaned training processes

    Returns:
        dict: All check results

    Raises:
        PreflightCheckError: If critical checks fail (strict=True)

    Example:
        >>> from nsm.evaluation.preflight_checks import run_preflight_checks
        >>>
        >>> results = run_preflight_checks(
        ...     dataset=train_dataset,
        ...     model=model,
        ...     cycle_loss_weight=0.01,
        ...     learning_rate=5e-4,
        ...     class_weights=torch.tensor([1.0, 1.0])
        ... )
        >>>
        >>> if results['all_passed']:
        ...     print("✅ All preflight checks passed!")
        ...     # Start training
    """
    print("\n" + "="*80)
    print("🚀 Running NSM Preflight Checks (NSM-31)")
    print("="*80 + "\n")

    # Check for orphaned processes first
    if check_processes:
        from nsm.evaluation.process_cleanup import check_and_cleanup
        check_and_cleanup(interactive=True, auto_kill=False)

    results = {}
    errors = []
    warnings_list = []

    # Capture warnings
    import warnings as warnings_module
    with warnings_module.catch_warnings(record=True) as w:
        warnings_module.simplefilter("always")

        try:
            # 1. Check PyG extensions (always required)
            results['pyg'] = check_pyg_extensions()

            # 2. Check cycle loss weight
            results['cycle_loss'] = check_cycle_loss_weight(cycle_loss_weight)

            # 3. Check learning rate
            results['learning_rate'] = check_learning_rate(learning_rate)

            # 4. Check dataset balance (if provided)
            if dataset is not None:
                results['dataset_balance'] = check_dataset_balance(dataset)

                # 5. Check class weights (requires dataset balance results)
                results['class_weights'] = check_class_weights(
                    class_weights,
                    results['dataset_balance']
                )

            # 6. Check model architecture (if provided)
            if model is not None:
                results['model'] = check_model_architecture(model)

            # Collect warnings
            for warning in w:
                if issubclass(warning.category, PreflightCheckWarning):
                    warnings_list.append(str(warning.message))

        except PreflightCheckError as e:
            errors.append(str(e))
            if strict:
                print("\n" + "="*80)
                print("❌ PREFLIGHT CHECK FAILED")
                print("="*80)
                raise

    # Summary
    print("\n" + "="*80)
    if errors:
        print("❌ PREFLIGHT CHECKS FAILED")
        for error in errors:
            print(f"\n{error}")
        results['all_passed'] = False
    elif warnings_list:
        print("⚠️  PREFLIGHT CHECKS PASSED WITH WARNINGS")
        for warning in warnings_list:
            print(f"\n{warning}")
        results['all_passed'] = True
        results['has_warnings'] = True
    else:
        print("✅ ALL PREFLIGHT CHECKS PASSED")
        results['all_passed'] = True
        results['has_warnings'] = False

    print("="*80 + "\n")

    return results


if __name__ == '__main__':
    # Self-test
    print("Running preflight checks self-test...\n")

    # Test with known-good parameters (NSM-31 Phase 1)
    try:
        results = run_preflight_checks(
            cycle_loss_weight=0.01,
            learning_rate=5e-4,
            strict=False
        )
        print("Self-test passed!" if results['all_passed'] else "Self-test failed!")
    except Exception as e:
        print(f"Self-test error: {e}")
