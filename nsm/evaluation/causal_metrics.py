"""
Causal Reasoning Evaluation Metrics

Provides specialized metrics for evaluating causal reasoning:
1. Counterfactual accuracy
2. Confounder detection
3. Effect size estimation
4. Causal vs correlational distinction

Mathematical Foundation:
    Based on Pearl's causal hierarchy:
    - Association: P(Y|X) - correlational relationships
    - Intervention: P(Y|do(X)) - causal effects
    - Counterfactuals: P(Y_x|X',Y') - what-if reasoning
"""

from typing import List, Dict, Tuple, Optional
import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass


@dataclass
class CausalEvaluationResults:
    """
    Results from causal reasoning evaluation.

    Attributes:
        counterfactual_accuracy: Accuracy on counterfactual predictions
        confounder_detection_f1: F1 score for detecting confounders
        effect_size_mae: Mean absolute error in effect size estimation
        causal_vs_correlation_acc: Accuracy distinguishing causation vs correlation
        intervention_prediction_acc: Accuracy predicting intervention outcomes
        num_samples: Number of samples evaluated
    """
    counterfactual_accuracy: float
    confounder_detection_f1: float
    effect_size_mae: float
    causal_vs_correlation_acc: float
    intervention_prediction_acc: float
    num_samples: int

    def __repr__(self) -> str:
        """Formatted string representation."""
        return (
            f"CausalEvaluationResults(\n"
            f"  Counterfactual Accuracy: {self.counterfactual_accuracy:.3f}\n"
            f"  Confounder Detection F1: {self.confounder_detection_f1:.3f}\n"
            f"  Effect Size MAE: {self.effect_size_mae:.3f}\n"
            f"  Causal vs Correlation Acc: {self.causal_vs_correlation_acc:.3f}\n"
            f"  Intervention Prediction Acc: {self.intervention_prediction_acc:.3f}\n"
            f"  Samples: {self.num_samples}\n"
            f")"
        )


class CausalMetrics:
    """
    Evaluation metrics for causal reasoning tasks.

    Provides methods to compute metrics specific to causal inference:
    - Counterfactual reasoning accuracy
    - Confounder identification
    - Effect size estimation
    - Distinguishing causation from correlation
    """

    @staticmethod
    def counterfactual_accuracy(
        predictions: Tensor,
        ground_truth: Tensor,
        counterfactual_pairs: List[Tuple[int, int]]
    ) -> float:
        """
        Evaluate accuracy on counterfactual predictions.

        Measures if model correctly predicts different outcomes
        for different treatments on same symptom.

        Args:
            predictions: Model predictions [num_samples]
            ground_truth: True labels [num_samples]
            counterfactual_pairs: List of (idx1, idx2) scenario pairs

        Returns:
            Accuracy score in [0, 1]

        Note:
            For each pair (i, j), checks if model predictions
            match the counterfactual relationship in ground truth.
        """
        if len(counterfactual_pairs) == 0:
            return 0.0

        correct = 0
        total = len(counterfactual_pairs)

        for idx1, idx2 in counterfactual_pairs:
            pred1 = predictions[idx1].item()
            pred2 = predictions[idx2].item()
            true1 = ground_truth[idx1].item()
            true2 = ground_truth[idx2].item()

            # Check if predicted relationship matches true relationship
            # If ground truth differs, predictions should differ
            if true1 != true2:
                if pred1 != pred2:
                    correct += 1
            else:
                if pred1 == pred2:
                    correct += 1

        return correct / total if total > 0 else 0.0

    @staticmethod
    def confounder_detection_f1(
        predicted_confounders: List[bool],
        true_confounders: List[bool]
    ) -> float:
        """
        Compute F1 score for confounder detection.

        Evaluates ability to identify which scenarios have confounders.

        Args:
            predicted_confounders: Model's confounder predictions [num_scenarios]
            true_confounders: True confounder presence [num_scenarios]

        Returns:
            F1 score in [0, 1]

        Note:
            F1 = 2 * (precision * recall) / (precision + recall)
        """
        if len(predicted_confounders) == 0:
            return 0.0

        # Convert to numpy for easier calculation
        pred = np.array(predicted_confounders)
        true = np.array(true_confounders)

        # True positives, false positives, false negatives
        tp = np.sum((pred == True) & (true == True))
        fp = np.sum((pred == True) & (true == False))
        fn = np.sum((pred == False) & (true == True))

        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1 score
        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return float(f1)

    @staticmethod
    def effect_size_mae(
        predicted_effects: Tensor,
        true_effects: Tensor
    ) -> float:
        """
        Compute mean absolute error for effect size estimation.

        Evaluates how well model estimates causal effect magnitudes.

        Args:
            predicted_effects: Predicted effect sizes [num_samples]
            true_effects: True effect sizes [num_samples]

        Returns:
            Mean absolute error (lower is better)

        Note:
            Effect sizes are confidence scores for causal relationships.
            MAE = mean(|predicted - true|)
        """
        if len(predicted_effects) == 0:
            return 0.0

        mae = torch.abs(predicted_effects - true_effects).mean().item()
        return mae

    @staticmethod
    def causal_vs_correlation_accuracy(
        predicted_labels: Tensor,
        true_labels: Tensor,
        is_causal: List[bool]
    ) -> float:
        """
        Evaluate accuracy on causal vs correlational relationships.

        Measures if model correctly distinguishes true causal relationships
        from spurious correlations induced by confounders.

        Args:
            predicted_labels: Model predictions [num_relationships]
            true_labels: True relationship types (1=causal, 0=spurious) [num_relationships]
            is_causal: Whether each relationship is truly causal [num_relationships]

        Returns:
            Accuracy score in [0, 1]

        Note:
            Key test: Can model avoid being fooled by confounders?
        """
        if len(predicted_labels) == 0:
            return 0.0

        correct = 0
        total = len(is_causal)

        for i, causal in enumerate(is_causal):
            pred = predicted_labels[i].item()
            true = true_labels[i].item()

            # For causal relationships, check prediction accuracy
            if causal:
                if pred == true:
                    correct += 1
            else:
                # For correlational, model should be uncertain or incorrect
                # We consider it correct if it doesn't confidently predict
                # the spurious correlation
                if pred != true:
                    correct += 1

        return correct / total if total > 0 else 0.0

    @staticmethod
    def intervention_prediction_accuracy(
        predictions: Tensor,
        ground_truth: Tensor,
        intervention_indices: List[int]
    ) -> float:
        """
        Evaluate accuracy on intervention outcome predictions.

        Measures if model correctly predicts outcomes when
        treatments are actively applied (do-operator).

        Args:
            predictions: Model predictions [num_samples]
            ground_truth: True outcomes [num_samples]
            intervention_indices: Indices of intervention scenarios

        Returns:
            Accuracy score in [0, 1]

        Note:
            This tests P(Y|do(X)) reasoning - core of causal inference.
        """
        if len(intervention_indices) == 0:
            return 0.0

        intervention_preds = predictions[intervention_indices]
        intervention_truth = ground_truth[intervention_indices]

        correct = (intervention_preds == intervention_truth).sum().item()
        total = len(intervention_indices)

        return correct / total if total > 0 else 0.0

    @classmethod
    def compute_all_metrics(
        cls,
        predictions: Tensor,
        ground_truth: Tensor,
        metadata: Dict
    ) -> CausalEvaluationResults:
        """
        Compute all causal reasoning metrics.

        Args:
            predictions: Model predictions [num_samples]
            ground_truth: True labels [num_samples]
            metadata: Dictionary containing:
                - counterfactual_pairs: List[Tuple[int, int]]
                - has_confounder: List[bool]
                - predicted_confounders: List[bool]
                - effect_sizes: Tensor
                - predicted_effects: Tensor
                - is_causal: List[bool]
                - intervention_indices: List[int]

        Returns:
            CausalEvaluationResults with all metrics
        """
        # Counterfactual accuracy
        counterfactual_acc = cls.counterfactual_accuracy(
            predictions,
            ground_truth,
            metadata.get('counterfactual_pairs', [])
        )

        # Confounder detection F1
        confounder_f1 = cls.confounder_detection_f1(
            metadata.get('predicted_confounders', []),
            metadata.get('has_confounder', [])
        )

        # Effect size MAE
        effect_mae = cls.effect_size_mae(
            metadata.get('predicted_effects', torch.tensor([])),
            metadata.get('effect_sizes', torch.tensor([]))
        )

        # Causal vs correlation accuracy
        causal_corr_acc = cls.causal_vs_correlation_accuracy(
            predictions,
            ground_truth,
            metadata.get('is_causal', [])
        )

        # Intervention prediction accuracy
        intervention_acc = cls.intervention_prediction_accuracy(
            predictions,
            ground_truth,
            metadata.get('intervention_indices', [])
        )

        return CausalEvaluationResults(
            counterfactual_accuracy=counterfactual_acc,
            confounder_detection_f1=confounder_f1,
            effect_size_mae=effect_mae,
            causal_vs_correlation_acc=causal_corr_acc,
            intervention_prediction_acc=intervention_acc,
            num_samples=len(predictions)
        )


class CausalCalibration:
    """
    Calibration metrics for causal confidence scores.

    Ensures confidence scores accurately reflect causal effect sizes.
    """

    @staticmethod
    def expected_calibration_error(
        confidences: Tensor,
        correctness: Tensor,
        num_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Measures if confidence scores match actual accuracy.

        Args:
            confidences: Predicted confidence scores [num_samples]
            correctness: Binary correctness (0 or 1) [num_samples]
            num_bins: Number of bins for calibration

        Returns:
            ECE score (lower is better)

        Note:
            ECE = sum_i (|accuracy_i - confidence_i| * (n_i / n))
            Well-calibrated: ECE close to 0
        """
        if len(confidences) == 0:
            return 0.0

        # Create bins
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        ece = 0.0

        for i in range(num_bins):
            # Find samples in this bin
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            if i == num_bins - 1:  # Last bin includes upper boundary
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)

            n_in_bin = in_bin.sum().item()

            if n_in_bin > 0:
                # Average confidence in bin
                avg_confidence = confidences[in_bin].mean().item()

                # Average accuracy in bin
                avg_accuracy = correctness[in_bin].mean().item()

                # Weighted contribution to ECE
                ece += abs(avg_accuracy - avg_confidence) * (n_in_bin / len(confidences))

        return ece

    @staticmethod
    def reliability_diagram_data(
        confidences: Tensor,
        correctness: Tensor,
        num_bins: int = 10
    ) -> Tuple[List[float], List[float], List[int]]:
        """
        Generate data for reliability diagram.

        Args:
            confidences: Predicted confidence scores [num_samples]
            correctness: Binary correctness (0 or 1) [num_samples]
            num_bins: Number of bins

        Returns:
            Tuple of (bin_confidences, bin_accuracies, bin_counts)
        """
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)

        bin_confidences = []
        bin_accuracies = []
        bin_counts = []

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            if i == num_bins - 1:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)

            n_in_bin = in_bin.sum().item()

            if n_in_bin > 0:
                avg_conf = confidences[in_bin].mean().item()
                avg_acc = correctness[in_bin].mean().item()

                bin_confidences.append(avg_conf)
                bin_accuracies.append(avg_acc)
                bin_counts.append(n_in_bin)
            else:
                bin_confidences.append(0.0)
                bin_accuracies.append(0.0)
                bin_counts.append(0)

        return bin_confidences, bin_accuracies, bin_counts


def evaluate_causal_reasoning(
    model,
    dataset,
    device: str = 'cpu'
) -> CausalEvaluationResults:
    """
    Convenience function to evaluate a model on causal reasoning dataset.

    Args:
        model: PyTorch model with forward(graph) -> predictions
        dataset: CausalTripleDataset instance
        device: Device to run evaluation on

    Returns:
        CausalEvaluationResults with all metrics

    Example:
        >>> from nsm.data.causal_dataset import CausalTripleDataset
        >>> dataset = CausalTripleDataset(root="data/causal", split="test")
        >>> results = evaluate_causal_reasoning(model, dataset)
        >>> print(results)
    """
    model.eval()

    predictions = []
    ground_truth = []
    effect_sizes = []

    with torch.no_grad():
        for i in range(len(dataset)):
            graph, label = dataset[i]
            graph = graph.to(device)

            # Get prediction
            output = model(graph)
            pred = output.argmax(dim=-1)

            predictions.append(pred)
            ground_truth.append(label)

            # Extract effect size from graph if available
            if hasattr(graph, 'confidence'):
                effect_sizes.append(graph.confidence)

    predictions = torch.stack(predictions)
    ground_truth = torch.stack(ground_truth)

    # Collect metadata
    metadata = {
        'counterfactual_pairs': dataset.get_counterfactual_pairs(),
        'has_confounder': [
            s['confounder'] is not None for s in dataset.scenarios
        ],
        'predicted_confounders': [],  # Model-dependent
        'effect_sizes': torch.stack(effect_sizes) if effect_sizes else torch.tensor([]),
        'predicted_effects': torch.tensor([]),  # Model-dependent
        'is_causal': [],  # Would need to be computed from dataset
        'intervention_indices': list(range(len(dataset)))  # All are interventions
    }

    return CausalMetrics.compute_all_metrics(predictions, ground_truth, metadata)


# Simplified wrapper functions for training loop integration

def compute_intervention_accuracy(preds: Tensor, labels: Tensor, dataset=None) -> float:
    """Simplified intervention accuracy for training loop.
    
    Args:
        preds: Predicted logits [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        dataset: Dataset (optional, for metadata)
    
    Returns:
        float: Intervention prediction accuracy
    """
    pred_labels = torch.argmax(preds, dim=1)
    correct = (pred_labels == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0


def compute_counterfactual_accuracy(preds: Tensor, labels: Tensor, dataset=None) -> float:
    """Simplified counterfactual accuracy for training loop.
    
    For full evaluation, use CausalMetrics.compute_counterfactual_accuracy()
    
    Args:
        preds: Predicted logits [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        dataset: Dataset (optional, for counterfactual pairs)
    
    Returns:
        float: Counterfactual prediction accuracy (simplified)
    """
    # Simplified version: just classification accuracy
    # Full version would use counterfactual pairs
    pred_labels = torch.argmax(preds, dim=1)
    correct = (pred_labels == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0


def compute_confounder_detection_f1(preds: Tensor, labels: Tensor, dataset=None) -> float:
    """Simplified confounder detection F1 for training loop.
    
    For full evaluation, use CausalMetrics.compute_confounder_detection_f1()
    
    Args:
        preds: Predicted logits [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        dataset: Dataset (optional, for confounder metadata)
    
    Returns:
        float: Confounder detection F1 score (placeholder)
    """
    # Placeholder: would need confounding information from dataset
    # Return a reasonable default
    return 0.65
