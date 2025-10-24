"""
Test script to verify the metrics fix for link_prediction task.

This tests the fix for the tensor shape mismatch that was causing:
    RuntimeError: The size of tensor a (2) must match the size of tensor b (100)

Run with: python tests/test_metrics_fix.py
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nsm.training.trainer import compute_classification_metrics


def test_link_prediction_with_logits():
    """Test link_prediction with [batch_size, 2] logits (KG domain)."""
    print("\n" + "="*80)
    print("Test 1: link_prediction with [batch_size, 2] logits")
    print("="*80)

    batch_size = 32
    preds = torch.randn(batch_size, 2)  # Two-class logits
    labels = torch.randint(0, 2, (batch_size,))  # Binary labels

    print(f"Input shapes:")
    print(f"  preds: {preds.shape}")
    print(f"  labels: {labels.shape}")

    try:
        metrics = compute_classification_metrics(preds, labels, 'link_prediction')
        print(f"\n✅ SUCCESS! Metrics computed:")
        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}")
        return True
    except Exception as e:
        print(f"\n❌ FAILED with error: {e}")
        return False


def test_link_prediction_with_single_prob():
    """Test link_prediction with [batch_size, 1] probabilities."""
    print("\n" + "="*80)
    print("Test 2: link_prediction with [batch_size, 1] probabilities")
    print("="*80)

    batch_size = 32
    preds = torch.randn(batch_size, 1)  # Single probability (will apply sigmoid)
    labels = torch.randint(0, 2, (batch_size,))

    print(f"Input shapes:")
    print(f"  preds: {preds.shape}")
    print(f"  labels: {labels.shape}")

    try:
        metrics = compute_classification_metrics(preds, labels, 'link_prediction')
        print(f"\n✅ SUCCESS! Metrics computed:")
        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}")
        return True
    except Exception as e:
        print(f"\n❌ FAILED with error: {e}")
        return False


def test_classification():
    """Test multi-class classification (should still work)."""
    print("\n" + "="*80)
    print("Test 3: classification with [batch_size, num_classes]")
    print("="*80)

    batch_size = 32
    num_classes = 2
    preds = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    print(f"Input shapes:")
    print(f"  preds: {preds.shape}")
    print(f"  labels: {labels.shape}")

    try:
        metrics = compute_classification_metrics(preds, labels, 'classification')
        print(f"\n✅ SUCCESS! Metrics computed:")
        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}")
        return True
    except Exception as e:
        print(f"\n❌ FAILED with error: {e}")
        return False


def test_per_class_accuracy():
    """Test that per-class accuracy is computed correctly."""
    print("\n" + "="*80)
    print("Test 4: Per-class accuracy (link_prediction)")
    print("="*80)

    # Create balanced dataset
    batch_size = 100
    preds = torch.zeros(batch_size, 2)
    labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()

    # Make first 40 class-0 correct, last 30 class-1 correct
    preds[:40, 0] = 10.0  # Correct for class 0
    preds[40:50, 1] = 10.0  # Incorrect for class 0 (predicted class 1)
    preds[50:70, 1] = 10.0  # Correct for class 1
    preds[70:, 0] = 10.0  # Incorrect for class 1 (predicted class 0)

    print(f"Input shapes:")
    print(f"  preds: {preds.shape}")
    print(f"  labels: {labels.shape}")
    print(f"Label distribution: {labels.unique(return_counts=True)}")

    metrics = compute_classification_metrics(preds, labels, 'link_prediction')
    print(f"\n✅ Metrics computed:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")

    # Verify expected accuracies
    expected_acc_0 = 40 / 50  # 0.8
    expected_acc_1 = 20 / 50  # 0.4
    expected_overall = 60 / 100  # 0.6

    print(f"\nExpected accuracies:")
    print(f"  accuracy_class_0: {expected_acc_0:.4f} (got {metrics['accuracy_class_0']:.4f})")
    print(f"  accuracy_class_1: {expected_acc_1:.4f} (got {metrics['accuracy_class_1']:.4f})")
    print(f"  accuracy: {expected_overall:.4f} (got {metrics['accuracy']:.4f})")

    assert abs(metrics['accuracy_class_0'] - expected_acc_0) < 1e-6
    assert abs(metrics['accuracy_class_1'] - expected_acc_1) < 1e-6
    assert abs(metrics['accuracy'] - expected_overall) < 1e-6

    print("\n✅ All assertions passed!")
    return True


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Testing Metrics Fix for Link Prediction Task")
    print("="*80)

    results = {
        'test_link_prediction_with_logits': test_link_prediction_with_logits(),
        'test_link_prediction_with_single_prob': test_link_prediction_with_single_prob(),
        'test_classification': test_classification(),
        'test_per_class_accuracy': test_per_class_accuracy()
    }

    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    if all_passed:
        print("\n🎉 All tests passed! Metrics fix is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review the fix.")
        sys.exit(1)
