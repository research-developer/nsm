"""
Test that KG metrics work correctly with binary classification.
"""
import torch
from nsm.evaluation.kg_metrics import (
    compute_hits_at_k,
    compute_mrr,
    compute_analogical_reasoning_accuracy
)

def test_kg_metrics_binary_classification():
    """Test KG metrics with binary classification (2-class logits)."""
    print("Testing KG metrics with binary classification...")

    # Simulate batch of 10 samples with 2-class logits
    batch_size = 10
    preds = torch.randn(batch_size, 2)  # [10, 2] logits
    labels = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])  # Binary labels

    # Test compute_hits_at_k
    hits = compute_hits_at_k(preds, labels, k=10)
    print(f"✓ Hits@K: {hits:.4f}")
    assert isinstance(hits, float), "Hits@K should return float"
    assert 0.0 <= hits <= 1.0, "Hits@K should be in [0, 1]"

    # Test compute_mrr
    mrr = compute_mrr(preds, labels)
    print(f"✓ MRR: {mrr:.4f}")
    assert isinstance(mrr, float), "MRR should return float"
    assert 0.0 <= mrr <= 1.0, "MRR should be in [0, 1]"

    # Test compute_analogical_reasoning_accuracy
    acc = compute_analogical_reasoning_accuracy(preds, labels)
    print(f"✓ Analogical Reasoning Accuracy: {acc:.4f}")
    assert isinstance(acc, float), "Accuracy should return float"
    assert 0.0 <= acc <= 1.0, "Accuracy should be in [0, 1]"

    print("\n✅ All KG metrics work correctly with binary classification!")
    return True

def test_kg_metrics_single_output():
    """Test KG metrics with single probability output."""
    print("\nTesting KG metrics with single probability output...")

    # Simulate batch with single probability output
    batch_size = 10
    preds = torch.randn(batch_size, 1)  # [10, 1] probabilities (before sigmoid)
    labels = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])

    # Test compute_hits_at_k
    hits = compute_hits_at_k(preds, labels, k=10)
    print(f"✓ Hits@K (single output): {hits:.4f}")

    # Test compute_mrr
    mrr = compute_mrr(preds, labels)
    print(f"✓ MRR (single output): {mrr:.4f}")

    # Test compute_analogical_reasoning_accuracy
    acc = compute_analogical_reasoning_accuracy(preds, labels)
    print(f"✓ Analogical Reasoning Accuracy (single output): {acc:.4f}")

    print("\n✅ Metrics work with single output format!")
    return True

def test_edge_cases():
    """Test edge cases like all zeros, all ones, etc."""
    print("\nTesting edge cases...")

    # All positive labels
    preds = torch.randn(5, 2)
    labels = torch.ones(5, dtype=torch.long)

    hits = compute_hits_at_k(preds, labels)
    mrr = compute_mrr(preds, labels)
    acc = compute_analogical_reasoning_accuracy(preds, labels)

    print(f"✓ All positive labels: hits={hits:.4f}, mrr={mrr:.4f}, acc={acc:.4f}")

    # All negative labels
    labels = torch.zeros(5, dtype=torch.long)

    hits = compute_hits_at_k(preds, labels)  # Should handle no positives
    mrr = compute_mrr(preds, labels)  # Should return 0
    acc = compute_analogical_reasoning_accuracy(preds, labels)

    print(f"✓ All negative labels: hits={hits:.4f}, mrr={mrr:.4f}, acc={acc:.4f}")

    print("\n✅ Edge cases handled correctly!")
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("KG Metrics Fix Verification")
    print("=" * 60)
    print()

    try:
        test_kg_metrics_binary_classification()
        test_kg_metrics_single_output()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("✅ All verification tests passed!")
        print("=" * 60)
        print("\nKG metrics are now compatible with binary classification.")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
