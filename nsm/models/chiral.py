"""
Chiral Dual-Trifold Architecture with Bidirectional Exchange

This module implements the chiral architecture where two mirror-image
hierarchical flows (bottom-up WHY and top-down WHAT) meet at middle
layers and exchange information via hinge mechanisms.

Theoretical Foundation:
- Category Theory: Adjoint functors (WHY ⊣ WHAT)
- Chiral Symmetry: Mirror-image processes that interact
- BDI-HTN-HRL: Validated 6-level cognitive hierarchy

References:
- Mac Lane (1998): Categories for the Working Mathematician
- NSM-31: Chiral Dual-Trifold Architecture
- notes/CHIRAL_ARCHITECTURE.md: 3-level minimal design
- notes/FULL_CHIRAL_6LEVEL.md: 6-level complete specification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from torch_geometric.nn import global_mean_pool

from .rgcn import ConfidenceWeightedRGCN
from .pooling import SymmetricGraphPooling


class ChiralHingeExchange(nn.Module):
    """
    Bidirectional exchange mechanism using simple weighted fusion.

    Allows upper (bottom-up) and lower (top-down) flows to exchange
    information via learnable weighted combination. This is the simplest
    baseline approach for hinge exchange.

    Mechanism:
        x_upper_refined = alpha * x_upper + (1 - alpha) * transform(x_lower)
        x_lower_refined = beta * x_lower + (1 - beta) * transform(x_upper)

    Args:
        dim: Hidden dimension
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim

        # Learnable mixing weights (per-dimension)
        self.alpha = nn.Parameter(torch.ones(1, dim) * 0.5)  # Initialize to 0.5
        self.beta = nn.Parameter(torch.ones(1, dim) * 0.5)

        # Transform layers (project other flow before mixing)
        self.transform_lower_for_upper = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.transform_upper_for_lower = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x_upper: torch.Tensor,
        x_lower: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional weighted fusion at hinge.

        Args:
            x_upper: Upper flow representation [num_nodes, dim]
            x_lower: Lower flow representation [num_nodes, dim]

        Returns:
            (x_upper_refined, x_lower_refined): Fused representations
        """
        # Transform flows for cross-pollination
        lower_transformed = self.transform_lower_for_upper(x_lower)
        upper_transformed = self.transform_upper_for_lower(x_upper)

        # Weighted fusion with learnable mixing coefficients
        # Constrain alpha and beta to [0, 1] via sigmoid
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)

        x_upper_refined = alpha * x_upper + (1 - alpha) * lower_transformed
        x_lower_refined = beta * x_lower + (1 - beta) * upper_transformed

        return x_upper_refined, x_lower_refined


class MinimalChiralModel(nn.Module):
    """
    Minimal 3-level chiral architecture with fusion-based hinge exchange.

    Architecture:
        Upper Flow (WHY):  L1 → L2_up
                                  ↕ (HINGE EXCHANGE via weighted fusion)
        Lower Flow (WHAT): L3 → L2_down

        Prediction: From L2_chiral = hinge_exchange(L2_up, L2_down)

    This minimal version tests the core hypothesis: simultaneous bidirectional
    flows with L2 exchange can prevent class collapse.

    Args:
        node_features: Input node feature dimension
        num_relations: Number of relation types
        num_classes: Number of output classes
        num_bases: Number of basis matrices for R-GCN (default: num_relations // 4)
        pool_ratio: Fraction of nodes to keep when pooling (default: 0.5)
        task_type: 'classification' or 'regression'
    """

    def __init__(
        self,
        node_features: int,
        num_relations: int,
        num_classes: int,
        num_bases: Optional[int] = None,
        pool_ratio: float = 0.5,
        task_type: str = 'classification'
    ):
        super().__init__()
        self.node_features = node_features
        self.num_relations = num_relations
        self.num_classes = num_classes
        self.task_type = task_type
        self.pool_ratio = pool_ratio

        if num_bases is None:
            num_bases = max(1, num_relations // 4)

        # Upper flow: L1 → L2_up (bottom-up, WHY operation)
        self.rgcn_l1 = ConfidenceWeightedRGCN(
            in_channels=node_features,
            out_channels=node_features,
            num_relations=num_relations,
            num_bases=num_bases
        )
        self.pool_l1_to_l2 = SymmetricGraphPooling(
            in_channels=node_features,
            ratio=pool_ratio
        )

        # Lower flow: L3 → L2_down (top-down, WHAT operation)
        # L3 starts as a learned embedding (abstract "mission/capability" prior)
        self.l3_prior = nn.Parameter(torch.randn(1, node_features))
        self.unpool_l3_to_l2 = nn.Linear(node_features, node_features)

        # Hinge exchange at L2 (fusion-based)
        self.hinge_l2 = ChiralHingeExchange(
            dim=node_features,
            dropout=0.1
        )

        # Prediction head from L2_chiral
        if task_type == 'classification':
            self.predictor = nn.Sequential(
                nn.Linear(node_features, node_features // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(node_features // 2, num_classes)
            )
        else:
            self.predictor = nn.Sequential(
                nn.Linear(node_features, node_features // 2),
                nn.ReLU(),
                nn.Linear(node_features // 2, 1)
            )

        # Cycle reconstruction head (for cycle loss)
        self.reconstruct_l1 = nn.Linear(node_features, node_features)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with simultaneous bidirectional flows and L2 exchange.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_type: Edge types [num_edges]
            batch: Batch assignment [num_nodes] (optional)

        Returns:
            Dictionary with:
                'logits': Task predictions
                'x_l2_up': Upper flow L2 representation
                'x_l2_down': Lower flow L2 representation
                'x_l2_chiral': Exchanged L2 representation
                'cycle_loss': Reconstruction error
                'perm_l2': Pooling permutation indices
        """
        num_nodes = x.size(0)

        # Default batch if not provided
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

        # ===== UPPER FLOW: L1 → L2_up (WHY operation) =====
        # Message passing at L1
        x_l1 = self.rgcn_l1(x, edge_index, edge_type)

        # Pool to L2 (abstraction)
        x_l2_up, edge_index_l2, _, batch_l2, perm_l2, score_l2 = self.pool_l1_to_l2.why_operation(
            x_l1, edge_index, edge_attr=None, batch=batch
        )

        # ===== LOWER FLOW: L3 → L2_down (WHAT operation) =====
        # Start with L3 prior (broadcast to match L2 size)
        num_l2_nodes = x_l2_up.size(0)
        x_l3 = self.l3_prior.expand(num_l2_nodes, -1)  # [num_l2_nodes, node_features]

        # "Unpool" from L3 to L2 (concretization via linear transform)
        x_l2_down = self.unpool_l3_to_l2(x_l3)

        # ===== HINGE EXCHANGE AT L2 (CHIRAL INTERACTION) =====
        x_l2_up_refined, x_l2_down_refined = self.hinge_l2(x_l2_up, x_l2_down)

        # Fuse upper and lower for final L2 representation
        x_l2_chiral = (x_l2_up_refined + x_l2_down_refined) / 2

        # ===== PREDICTION FROM L2_CHIRAL =====
        # Global pooling to graph-level representation
        x_graph = global_mean_pool(x_l2_chiral, batch_l2)

        logits = self.predictor(x_graph)

        # ===== CYCLE CONSISTENCY (for training stability) =====
        # Reconstruct L1 from L2_chiral to ensure information preservation
        # Unpool L2 back to L1 size
        x_l1_reconstructed = torch.zeros_like(x_l1)
        x_l1_reconstructed[perm_l2] = self.reconstruct_l1(x_l2_chiral)

        cycle_loss = F.mse_loss(x_l1_reconstructed, x_l1)

        return {
            'logits': logits,
            'x_l2_up': x_l2_up,
            'x_l2_down': x_l2_down,
            'x_l2_chiral': x_l2_chiral,
            'x_l1_reconstructed': x_l1_reconstructed,
            'cycle_loss': cycle_loss,
            'perm_l2': perm_l2,
            'score_l2': score_l2
        }


class FullChiralModel(nn.Module):
    """
    Full 6-level chiral dual-trifold architecture (NSM-31 Stage 2).

    Architecture:
        Upper Trifold:  L1 → L2 → L3  (WHY: concrete → abstract)
                         ↓    ↓    ↓
                       Hinge Hinge Hinge  (Cross-attention)
                         ↓    ↓    ↓
        Lower Trifold:  L6 → L5 → L4  (WHAT: abstract → concrete, inverted)

        Exchanges:
        - L3 ↔ L4: Capability ↔ Beliefs
        - L2 ↔ L5: Behavior ↔ Identity
        - L1 ↔ L6: Environment ↔ Mission

    This full version implements all 3 hinges with normalization inversion
    to match scales between upper (increasing abstraction) and lower
    (decreasing abstraction) trifolds.

    Args:
        node_features: Input node feature dimension
        num_relations: Number of relation types
        num_classes: Number of output classes
        task_type: 'classification' or 'regression'
    """

    def __init__(
        self,
        node_features: int,
        num_relations: int,
        num_classes: int,
        task_type: str = 'classification'
    ):
        super().__init__()
        self.node_features = node_features
        self.num_relations = num_relations
        self.num_classes = num_classes
        self.task_type = task_type

        # TODO: Implement upper trifold (L1 → L2 → L3)
        # TODO: Implement lower trifold (L6 → L5 → L4)
        # TODO: Implement 3 hinge exchanges
        # TODO: Implement normalization inversion
        # TODO: Implement multi-level prediction heads

        raise NotImplementedError("FullChiralModel needs implementation")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        x_l6_prior: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dual trifolds and triple hinge exchange.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_type: Edge types [num_edges]
            batch: Batch assignment [num_nodes] (optional)
            x_l6_prior: Mission/purpose prior [num_nodes, node_features] (optional)

        Returns:
            Dictionary with all level representations and predictions
        """
        # TODO: Implement forward pass
        raise NotImplementedError("FullChiralModel.forward needs implementation")


# Export public API
__all__ = [
    'ChiralHingeExchange',
    'MinimalChiralModel',
    'FullChiralModel'
]
