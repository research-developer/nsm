"""
Fixed 6-level chiral architecture with CORRECTED temperature profile.

ROOT CAUSE FIX: Original architecture had inverted temperature profile
    L1 (concrete): HIGH diversity (0.40)
    L3 (abstract):  LOW diversity (0.13)  ← BACKWARDS!

This causes inherent instability (like plasma with hot edge, cold core).

FIX: Add diversity regularization to INCREASE temperature at higher levels
    - L3 should have HIGHEST diversity (abstract concepts)
    - L1 should have LOWEST diversity (concrete actions)
    - Enforce this via loss function

Alternative fix (not implemented yet): Reverse pooling direction entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from nsm.models.chiral import FullChiralModel, ChiralHingeExchange
from nsm.models.rgcn import ConfidenceWeightedRGCN
from nsm.models.pooling import SymmetricGraphPooling


class DiversityRegularization(nn.Module):
    """
    Enforce correct temperature profile: L1 < L2 < L3 in diversity.

    Adds loss penalty when temperature is inverted.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        x_l1: torch.Tensor,
        x_l2: torch.Tensor,
        x_l3: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute diversity regularization loss.

        Penalizes when:
        - T_L3 < T_L2 or T_L2 < T_L1
        - Encourages hierarchical diversity increase

        Args:
            x_l1, x_l2, x_l3: Level representations

        Returns:
            (loss, diagnostics)
        """
        # Compute temperatures (variances)
        T_L1 = x_l1.var(dim=0).mean()
        T_L2 = x_l2.var(dim=0).mean()
        T_L3 = x_l3.var(dim=0).mean()

        # Desired: T_L1 < T_L2 < T_L3
        # Penalize violations
        loss = 0.0

        # L2 should be hotter than L1
        if T_L2 < T_L1:
            loss += F.relu(T_L1 - T_L2)  # Penalize inversion

        # L3 should be hotter than L2
        if T_L3 < T_L2:
            loss += F.relu(T_L2 - T_L3)  # Penalize inversion

        # Also add bonus for correct ordering
        # Encourage gradient: T_L3 - T_L1 > 0.1
        gradient = T_L3 - T_L1
        target_gradient = 0.1

        if gradient < target_gradient:
            loss += F.relu(target_gradient - gradient)

        loss *= self.weight

        diagnostics = {
            'T_L1': T_L1.item(),
            'T_L2': T_L2.item(),
            'T_L3': T_L3.item(),
            'T_gradient': gradient.item(),
            'diversity_loss': loss.item()
        }

        return loss, diagnostics


class FixedTemperatureChiralModel(FullChiralModel):
    """
    6-level chiral model with corrected temperature profile via regularization.

    Inherits from FullChiralModel but adds diversity regularization.
    """

    def __init__(
        self,
        node_features: int,
        num_relations: int,
        num_classes: int,
        pool_ratio: float = 0.5,
        task_type: str = 'classification',
        dropout: float = 0.1,
        diversity_reg_weight: float = 0.1
    ):
        super().__init__(
            node_features=node_features,
            num_relations=num_relations,
            num_classes=num_classes,
            pool_ratio=pool_ratio,
            task_type=task_type,
            dropout=dropout
        )

        # Add diversity regularization
        self.diversity_regularizer = DiversityRegularization(weight=diversity_reg_weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        x_l6_prior: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with diversity regularization.

        Adds 'diversity_loss' to output dict.
        """
        # Get base model output
        output = super().forward(x, edge_index, edge_type, batch, x_l6_prior)

        # Compute diversity regularization
        div_loss, div_diag = self.diversity_regularizer(
            x_l1=output['x_l1'],
            x_l2=output['x_l2'],
            x_l3=output['x_l3']
        )

        # Add to output
        output['diversity_loss'] = div_loss
        output['diversity_diagnostics'] = div_diag

        return output


class FixedTemperatureChiralLoss(nn.Module):
    """
    Composite loss including diversity regularization for temperature profile.

    Extends ChiralCompositeLoss to include diversity_loss from model.
    """

    def __init__(
        self,
        task_weight: float = 1.0,
        aux_weight: float = 0.3,
        cycle_weight: float = 0.01,
        diversity_weight: float = 0.1,  # Now actually used!
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

        self.task_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        task_type: str = 'classification'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss with diversity regularization.

        L_total = L_task + L_aux + L_cycle + L_diversity_profile
        """

        # Task losses (main + auxiliary)
        if self.use_focal_loss:
            loss_task_main = self._focal_loss(model_output['logits'], targets)
            loss_task_l1 = self._focal_loss(model_output['logits_l1'], targets)
            loss_task_l2 = self._focal_loss(model_output['logits_l2'], targets)
            loss_task_l3 = self._focal_loss(model_output['logits_l3'], targets)
        else:
            loss_task_main = self.task_criterion(model_output['logits'], targets)
            loss_task_l1 = self.task_criterion(model_output['logits_l1'], targets)
            loss_task_l2 = self.task_criterion(model_output['logits_l2'], targets)
            loss_task_l3 = self.task_criterion(model_output['logits_l3'], targets)

        loss_task_aux = (loss_task_l1 + loss_task_l2 + loss_task_l3) / 3

        # Cycle consistency losses
        loss_cycle_upper = model_output['cycle_loss_upper']
        loss_cycle_lower = model_output['cycle_loss_lower']
        loss_cycle_cross = model_output['cycle_loss_cross']
        loss_cycle_total = loss_cycle_upper + loss_cycle_lower + loss_cycle_cross

        # Diversity regularization (temperature profile correction)
        loss_diversity = model_output.get('diversity_loss', torch.tensor(0.0, device=targets.device))

        # Total loss
        loss_total = (
            self.task_weight * loss_task_main +
            self.aux_weight * loss_task_aux +
            self.cycle_weight * loss_cycle_total +
            self.diversity_weight * loss_diversity
        )

        return {
            'loss': loss_total,
            'loss_task': loss_task_main,
            'loss_task_aux': loss_task_aux,
            'loss_cycle': loss_cycle_total,
            'loss_cycle_upper': loss_cycle_upper,
            'loss_cycle_lower': loss_cycle_lower,
            'loss_cycle_cross': loss_cycle_cross,
            'loss_diversity': loss_diversity
        }

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for hard examples."""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - p_t) ** self.focal_gamma * ce_loss
        return focal_loss.mean()


# Export public API
__all__ = [
    'DiversityRegularization',
    'FixedTemperatureChiralModel',
    'FixedTemperatureChiralLoss'
]
