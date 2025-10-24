"""
Composite loss function for 6-level chiral dual-trifold architecture.

Implements the loss function specified in NSM-32:
    L_total = L_task_final + 0.3·L_task_aux +
              0.01·(L_cycle_upper + L_cycle_lower + L_cycle_cross) +
              [optional: 0.05·L_diversity]

References:
- NSM-32: Full 6-Level Chiral Dual-Trifold Architecture Design
- NSM-31: Phase 1.5 validation (fusion variant)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ChiralCompositeLoss(nn.Module):
    """
    Composite loss function for chiral dual-trifold architecture.

    Combines:
    1. Task losses (main + auxiliary)
    2. Cycle consistency losses (3 types)
    3. Optional diversity loss

    Args:
        task_weight: Weight for main task loss (default: 1.0)
        aux_weight: Weight for auxiliary task losses (default: 0.3)
        cycle_weight: Weight for cycle consistency losses (default: 0.01)
        diversity_weight: Weight for diversity loss (default: 0.0, disabled)
        use_focal_loss: Use focal loss for classification to handle imbalance (default: False)
        focal_alpha: Class weighting for focal loss (default: 0.25)
        focal_gamma: Focusing parameter for focal loss (default: 2.0)
    """

    def __init__(
        self,
        task_weight: float = 1.0,
        aux_weight: float = 0.3,
        cycle_weight: float = 0.01,
        diversity_weight: float = 0.0,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.task_weight = task_weight
        self.aux_weight = aux_weight
        self.cycle_weight = cycle_weight
        self.diversity_weight = diversity_weight
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Focal loss for addressing class imbalance.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        Args:
            logits: Raw model outputs [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Focal loss value
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Get probabilities for the true class
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        p_t = (probs * targets_one_hot).sum(dim=1)

        # Compute focal loss
        focal_weight = (1 - p_t) ** self.focal_gamma

        # Cross entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Apply focal weighting
        loss = self.focal_alpha * focal_weight * ce_loss

        return loss.mean()

    def diversity_loss(
        self,
        logits_l1: torch.Tensor,
        logits_l2: torch.Tensor,
        logits_l3: torch.Tensor
    ) -> torch.Tensor:
        """
        Diversity loss to encourage different predictions from different levels.

        Penalizes agreement between prediction heads to prevent collapse.

        Args:
            logits_l1: Predictions from L1 [batch_size, num_classes]
            logits_l2: Predictions from L2 [batch_size, num_classes]
            logits_l3: Predictions from L3 [batch_size, num_classes]

        Returns:
            Diversity loss (negative mean absolute difference)
        """
        # Convert to probabilities
        probs_l1 = F.softmax(logits_l1, dim=1)
        probs_l2 = F.softmax(logits_l2, dim=1)
        probs_l3 = F.softmax(logits_l3, dim=1)

        # Compute pairwise differences
        diff_l1_l2 = torch.abs(probs_l1 - probs_l2).mean()
        diff_l1_l3 = torch.abs(probs_l1 - probs_l3).mean()
        diff_l2_l3 = torch.abs(probs_l2 - probs_l3).mean()

        # Diversity loss: negative of average difference (we want to maximize difference)
        loss = -(diff_l1_l2 + diff_l1_l3 + diff_l2_l3) / 3

        return loss

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        task_type: str = 'classification'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss from model output.

        Args:
            model_output: Dictionary from FullChiralModel.forward() with keys:
                - 'logits': Final ensemble prediction
                - 'logits_l1', 'logits_l2', 'logits_l3': Auxiliary predictions
                - 'cycle_loss_upper': Upper trifold cycle loss
                - 'cycle_loss_lower': Lower trifold cycle loss
                - 'cycle_loss_cross': Cross-trifold cycle loss
            targets: Ground truth labels/values
            task_type: 'classification' or 'regression'

        Returns:
            Dictionary with:
                - 'loss': Total composite loss
                - 'loss_task': Main task loss
                - 'loss_task_aux': Auxiliary task losses
                - 'loss_cycle': Combined cycle losses
                - 'loss_diversity': Diversity loss (if enabled)
                - Individual loss components for logging
        """
        # ===== TASK LOSSES =====

        # Main task loss (ensemble prediction)
        if task_type == 'classification':
            if self.use_focal_loss:
                loss_task_main = self.focal_loss(model_output['logits'], targets)
            else:
                loss_task_main = F.cross_entropy(model_output['logits'], targets)

            # Auxiliary task losses (from each level)
            loss_task_l1 = F.cross_entropy(model_output['logits_l1'], targets)
            loss_task_l2 = F.cross_entropy(model_output['logits_l2'], targets)
            loss_task_l3 = F.cross_entropy(model_output['logits_l3'], targets)
        else:
            # Regression
            loss_task_main = F.mse_loss(model_output['logits'].squeeze(), targets.float())
            loss_task_l1 = F.mse_loss(model_output['logits_l1'].squeeze(), targets.float())
            loss_task_l2 = F.mse_loss(model_output['logits_l2'].squeeze(), targets.float())
            loss_task_l3 = F.mse_loss(model_output['logits_l3'].squeeze(), targets.float())

        # Combined auxiliary loss
        loss_task_aux = (loss_task_l1 + loss_task_l2 + loss_task_l3) / 3

        # ===== CYCLE CONSISTENCY LOSSES =====

        loss_cycle_upper = model_output['cycle_loss_upper']
        loss_cycle_lower = model_output['cycle_loss_lower']
        loss_cycle_cross = model_output['cycle_loss_cross']

        # Combined cycle loss
        loss_cycle_total = loss_cycle_upper + loss_cycle_lower + loss_cycle_cross

        # ===== DIVERSITY LOSS (optional) =====

        if self.diversity_weight > 0 and task_type == 'classification':
            loss_div = self.diversity_loss(
                model_output['logits_l1'],
                model_output['logits_l2'],
                model_output['logits_l3']
            )
        else:
            loss_div = torch.tensor(0.0, device=model_output['logits'].device)

        # ===== TOTAL LOSS =====

        loss_total = (
            self.task_weight * loss_task_main +
            self.aux_weight * loss_task_aux +
            self.cycle_weight * loss_cycle_total +
            self.diversity_weight * loss_div
        )

        return {
            # Total loss
            'loss': loss_total,

            # Main components
            'loss_task': loss_task_main,
            'loss_task_aux': loss_task_aux,
            'loss_cycle': loss_cycle_total,
            'loss_diversity': loss_div,

            # Detailed task losses
            'loss_task_l1': loss_task_l1,
            'loss_task_l2': loss_task_l2,
            'loss_task_l3': loss_task_l3,

            # Detailed cycle losses
            'loss_cycle_upper': loss_cycle_upper,
            'loss_cycle_lower': loss_cycle_lower,
            'loss_cycle_cross': loss_cycle_cross
        }


def compute_class_balance_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Compute class balance metrics for monitoring collapse.

    Args:
        logits: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        num_classes: Number of classes (default: 2)

    Returns:
        Dictionary with per-class accuracy and balance delta
    """
    predictions = logits.argmax(dim=1)

    metrics = {}

    for cls in range(num_classes):
        # Mask for this class
        mask = (targets == cls)

        if mask.sum() > 0:
            # Per-class accuracy
            correct = (predictions[mask] == cls).sum().item()
            total = mask.sum().item()
            accuracy = correct / total

            metrics[f'accuracy_class_{cls}'] = accuracy
        else:
            metrics[f'accuracy_class_{cls}'] = 0.0

    # Class balance delta (for binary classification)
    if num_classes == 2:
        balance_delta = abs(metrics['accuracy_class_0'] - metrics['accuracy_class_1'])
        metrics['class_balance_delta'] = balance_delta

    return metrics


# Export public API
__all__ = [
    'ChiralCompositeLoss',
    'compute_class_balance_metrics'
]
