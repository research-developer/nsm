"""
Knowledge Graph Evaluation Metrics

Implements evaluation metrics for knowledge graph reasoning tasks:
- Link prediction (Hits@K, MRR)
- Analogical reasoning (A:B :: C:?)
- Type consistency checking

Mathematical Foundation:
    Link prediction: Given (h, r, ?), rank all entities by score
    - Hits@K: Fraction of correct entities in top K
    - MRR: Mean reciprocal rank = 1/N * Σ(1/rank_i)

    Analogical reasoning: Given A:B :: C:D, find D
    - Requires embedding-based similarity or graph traversal

    Type consistency: Verify (entity, type) consistency
    - Binary classification: consistent vs inconsistent
"""

from typing import List, Tuple, Dict, Optional, Set
import torch
from torch import Tensor
import numpy as np


def compute_link_prediction_metrics(
    predictions: Tensor,
    targets: Tensor,
    k_values: List[int] = [1, 3, 10],
) -> Dict[str, float]:
    """
    Compute link prediction metrics: Hits@K and MRR.

    Args:
        predictions: Predicted scores [batch_size, num_candidates]
                    Higher scores indicate more likely candidates
        targets: True entity indices [batch_size]
        k_values: Values of K for Hits@K computation

    Returns:
        Dictionary containing:
            - hits@1, hits@3, hits@10: Fraction of correct predictions in top K
            - mrr: Mean reciprocal rank
            - mean_rank: Average rank of correct entity

    Mathematical Foundation:
        Hits@K = (1/N) * Σ I[rank(correct) ≤ K]
        MRR = (1/N) * Σ (1 / rank(correct))

    Examples:
        >>> predictions = torch.tensor([[0.1, 0.9, 0.3], [0.8, 0.2, 0.7]])
        >>> targets = torch.tensor([1, 0])  # True indices
        >>> metrics = compute_link_prediction_metrics(predictions, targets)
        >>> print(f"Hits@1: {metrics['hits@1']:.3f}")
    """
    batch_size = predictions.size(0)
    num_candidates = predictions.size(1)

    # Get ranks of target entities
    # Sort predictions in descending order
    sorted_indices = torch.argsort(predictions, dim=1, descending=True)

    # Find rank of each target
    ranks = []
    for i in range(batch_size):
        target_idx = targets[i].item()
        # Find position of target in sorted list (1-indexed)
        rank = (sorted_indices[i] == target_idx).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)

    # Compute metrics
    metrics = {}

    # Hits@K for each K value
    for k in k_values:
        hits_at_k = (ranks_tensor <= k).float().mean().item()
        metrics[f'hits@{k}'] = hits_at_k

    # Mean Reciprocal Rank
    mrr = (1.0 / ranks_tensor).mean().item()
    metrics['mrr'] = mrr

    # Mean Rank
    mean_rank = ranks_tensor.mean().item()
    metrics['mean_rank'] = mean_rank

    # Median Rank
    median_rank = ranks_tensor.median().item()
    metrics['median_rank'] = median_rank

    return metrics


def compute_analogical_reasoning_accuracy(
    embeddings: Tensor,
    analogy_queries: List[Tuple[int, int, int, int]],
    k: int = 1,
) -> Dict[str, float]:
    """
    Compute accuracy on analogical reasoning: A:B :: C:D.

    Uses vector arithmetic: D ≈ C + (B - A)

    Args:
        embeddings: Entity embeddings [num_entities, embed_dim]
        analogy_queries: List of (A, B, C, D) entity index tuples
        k: Top-K accuracy threshold

    Returns:
        Dictionary containing:
            - accuracy@k: Fraction of queries where D is in top K predictions
            - average_rank: Average rank of correct D

    Mathematical Foundation:
        Given embeddings e_A, e_B, e_C, e_D:
        Find D' = argmax_{i} sim(e_i, e_C + e_B - e_A)
        where sim is cosine similarity or dot product.

    Examples:
        >>> embeddings = torch.randn(100, 64)
        >>> queries = [(0, 1, 2, 3), (4, 5, 6, 7)]  # A:B :: C:D
        >>> metrics = compute_analogical_reasoning_accuracy(embeddings, queries)
    """
    if len(analogy_queries) == 0:
        return {'accuracy@1': 0.0, 'average_rank': float('inf')}

    correct_count = 0
    ranks = []

    for a_idx, b_idx, c_idx, d_idx in analogy_queries:
        # Vector arithmetic: D ≈ C + (B - A)
        e_a = embeddings[a_idx]
        e_b = embeddings[b_idx]
        e_c = embeddings[c_idx]
        e_d = embeddings[d_idx]

        # Predicted D embedding
        predicted_d = e_c + (e_b - e_a)

        # Compute similarity to all entities
        similarities = torch.matmul(embeddings, predicted_d)

        # Mask out A, B, C to avoid trivial solutions
        similarities[a_idx] = float('-inf')
        similarities[b_idx] = float('-inf')
        similarities[c_idx] = float('-inf')

        # Get ranked entities
        sorted_indices = torch.argsort(similarities, descending=True)

        # Find rank of correct D
        rank = (sorted_indices == d_idx).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

        # Check if in top K
        if rank <= k:
            correct_count += 1

    accuracy = correct_count / len(analogy_queries)
    avg_rank = np.mean(ranks)

    return {
        f'accuracy@{k}': accuracy,
        'average_rank': avg_rank,
        'num_queries': len(analogy_queries),
    }


def compute_type_consistency_accuracy(
    predictions: Tensor,
    labels: Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute type consistency checking accuracy.

    Args:
        predictions: Predicted consistency scores [num_pairs]
                    Values in [0, 1] where 1 = consistent
        labels: Ground truth labels [num_pairs]
               1 = consistent, 0 = inconsistent
        threshold: Classification threshold (default 0.5)

    Returns:
        Dictionary containing:
            - accuracy: Overall classification accuracy
            - precision: Precision for positive class (consistent)
            - recall: Recall for positive class
            - f1: F1 score
            - true_positives, false_positives, true_negatives, false_negatives

    Mathematical Foundation:
        Binary classification metrics:
        - Accuracy = (TP + TN) / (TP + TN + FP + FN)
        - Precision = TP / (TP + FP)
        - Recall = TP / (TP + FN)
        - F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Examples:
        >>> predictions = torch.tensor([0.9, 0.3, 0.8, 0.1])
        >>> labels = torch.tensor([1, 0, 1, 0])
        >>> metrics = compute_type_consistency_accuracy(predictions, labels)
    """
    # Binary predictions
    binary_preds = (predictions >= threshold).float()

    # Compute confusion matrix elements
    tp = ((binary_preds == 1) & (labels == 1)).sum().float().item()
    tn = ((binary_preds == 0) & (labels == 0)).sum().float().item()
    fp = ((binary_preds == 1) & (labels == 0)).sum().float().item()
    fn = ((binary_preds == 0) & (labels == 1)).sum().float().item()

    total = tp + tn + fp + fn

    # Compute metrics
    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
    }


def compute_multi_hop_reasoning_accuracy(
    graph_queries: List[Dict],
    predicted_answers: List[Set[int]],
    ground_truth_answers: List[Set[int]],
) -> Dict[str, float]:
    """
    Compute accuracy for multi-hop reasoning queries.

    Args:
        graph_queries: List of query dictionaries with path information
        predicted_answers: List of sets of predicted entity indices
        ground_truth_answers: List of sets of correct entity indices

    Returns:
        Dictionary containing:
            - exact_match: Fraction with exact answer set match
            - hits@1: Fraction with at least one correct answer in top-1
            - hits@3: Fraction with at least one correct answer in top-3
            - average_precision: Mean precision across queries

    Mathematical Foundation:
        Multi-hop reasoning requires chaining relations:
        Given path (r1, r2, ..., rk) and start entity e0,
        Find {ek | ∃e1,...,ek-1: (e0,r1,e1), (e1,r2,e2), ..., (ek-1,rk,ek)}

    Examples:
        >>> queries = [{'path': ['born_in', 'located_in'], 'start': 0}]
        >>> predicted = [{5, 6}]
        >>> ground_truth = [{5, 7}]
        >>> metrics = compute_multi_hop_reasoning_accuracy(queries, predicted, ground_truth)
    """
    if len(predicted_answers) != len(ground_truth_answers):
        raise ValueError("Number of predictions must match ground truth")

    exact_matches = 0
    hits_1 = 0
    hits_3 = 0
    precisions = []

    for pred_set, true_set in zip(predicted_answers, ground_truth_answers):
        # Exact match
        if pred_set == true_set:
            exact_matches += 1

        # Hits@K - check if any predicted answer is correct
        if len(pred_set & true_set) > 0:
            hits_1 += 1  # At least one correct
            hits_3 += 1

        # Precision for this query
        if len(pred_set) > 0:
            precision = len(pred_set & true_set) / len(pred_set)
            precisions.append(precision)
        else:
            precisions.append(0.0)

    num_queries = len(predicted_answers)

    return {
        'exact_match': exact_matches / num_queries if num_queries > 0 else 0.0,
        'hits@1': hits_1 / num_queries if num_queries > 0 else 0.0,
        'hits@3': hits_3 / num_queries if num_queries > 0 else 0.0,
        'average_precision': np.mean(precisions) if precisions else 0.0,
        'num_queries': num_queries,
    }


def compute_calibration_error(
    confidences: Tensor,
    accuracies: Tensor,
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well predicted confidence scores match actual accuracy.

    Args:
        confidences: Predicted confidence scores [num_samples]
        accuracies: Binary correctness (1 = correct, 0 = incorrect) [num_samples]
        num_bins: Number of bins for calibration curve

    Returns:
        Dictionary containing:
            - ece: Expected Calibration Error
            - mce: Maximum Calibration Error
            - calibration_curve: List of (bin_confidence, bin_accuracy, bin_count) tuples

    Mathematical Foundation:
        ECE = Σ (|Bm| / N) * |acc(Bm) - conf(Bm)|
        where Bm is the set of samples in bin m,
        acc(Bm) is the accuracy in bin m,
        conf(Bm) is the average confidence in bin m.

    Examples:
        >>> confidences = torch.tensor([0.9, 0.8, 0.6, 0.3])
        >>> accuracies = torch.tensor([1.0, 1.0, 0.0, 0.0])
        >>> metrics = compute_calibration_error(confidences, accuracies)
    """
    confidences = confidences.cpu().numpy()
    accuracies = accuracies.cpu().numpy()

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    calibration_curve = []

    total_samples = len(confidences)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        # Handle last bin inclusively
        if bin_upper == 1.0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)

        bin_size = in_bin.sum()

        if bin_size > 0:
            # Average confidence in bin
            bin_confidence = confidences[in_bin].mean()

            # Average accuracy in bin
            bin_accuracy = accuracies[in_bin].mean()

            # Calibration error for this bin
            bin_error = abs(bin_accuracy - bin_confidence)

            # Weighted contribution to ECE
            ece += (bin_size / total_samples) * bin_error

            # Update MCE
            mce = max(mce, bin_error)

            calibration_curve.append((
                float(bin_confidence),
                float(bin_accuracy),
                int(bin_size)
            ))
        else:
            calibration_curve.append((0.0, 0.0, 0))

    return {
        'ece': float(ece),
        'mce': float(mce),
        'calibration_curve': calibration_curve,
        'num_bins': num_bins,
    }


def compute_kg_comprehensive_metrics(
    model: torch.nn.Module,
    dataset,
    device: str = 'cpu',
    num_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for KG dataset.

    Args:
        model: Trained NSM model
        dataset: KnowledgeGraphTripleDataset instance
        device: Device to run evaluation on
        num_samples: Number of samples to evaluate (None = all)

    Returns:
        Dictionary with all KG-specific metrics

    Note:
        This is a convenience function that orchestrates all KG metrics.
        Requires model to have methods: forward(), predict_link(), etc.
    """
    model.eval()
    model = model.to(device)

    metrics = {}

    # Sample dataset if needed
    if num_samples is not None:
        indices = torch.randperm(len(dataset))[:num_samples].tolist()
    else:
        indices = list(range(len(dataset)))

    # Collect predictions and targets
    all_predictions = []
    all_targets = []
    all_confidences = []

    with torch.no_grad():
        for idx in indices:
            graph, label = dataset[idx]
            graph = graph.to(device)

            # Get model prediction
            # This assumes model returns (output, confidence)
            # Actual implementation depends on model architecture
            # output, confidence = model(graph)
            # all_predictions.append(output)
            # all_targets.append(label)
            # all_confidences.append(confidence)
            pass

    # TODO: Implement once model architecture is defined
    # metrics['link_prediction'] = compute_link_prediction_metrics(...)
    # metrics['type_consistency'] = compute_type_consistency_accuracy(...)
    # metrics['calibration'] = compute_calibration_error(...)

    return metrics
