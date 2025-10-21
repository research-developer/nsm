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
    Full 6-level chiral dual-trifold architecture (NSM-32).

    Architecture:
        Upper Trifold:  L1 → L2 → L3  (WHY: concrete → abstract, bottom-up)
                         ↕    ↕    ↕
                      Hinge1 Hinge2 Hinge3  (Fusion-based exchange)
                         ↕    ↕    ↕
        Lower Trifold:  L6 → L5 → L4  (WHAT: abstract → concrete, top-down)

    Exchanges (fusion-based):
        - Hinge 1: L1 ↔ L6 (Environment ↔ Mission)
        - Hinge 2: L2 ↔ L5 (Behavior ↔ Identity)
        - Hinge 3: L3 ↔ L4 (Capability ↔ Beliefs)

    This implements the validated fusion mechanism from Phase 1.5 with size
    alignment and scale normalization for cross-trifold exchange.

    Args:
        node_features: Input node feature dimension
        num_relations: Number of relation types
        num_classes: Number of output classes
        num_bases: Number of basis matrices for R-GCN (default: num_relations // 4)
        pool_ratio: Fraction of nodes to keep when pooling (default: 0.5)
        task_type: 'classification' or 'regression'
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        node_features: int,
        num_relations: int,
        num_classes: int,
        num_bases: Optional[int] = None,
        pool_ratio: float = 0.5,
        task_type: str = 'classification',
        dropout: float = 0.1
    ):
        super().__init__()
        self.node_features = node_features
        self.num_relations = num_relations
        self.num_classes = num_classes
        self.task_type = task_type
        self.pool_ratio = pool_ratio
        self.dropout = dropout

        if num_bases is None:
            num_bases = max(1, num_relations // 4)

        # ===== UPPER TRIFOLD (WHY: bottom-up, concrete → abstract) =====
        # L1: Environment/Perception (most concrete, ~1000 nodes)
        self.rgcn_l1 = ConfidenceWeightedRGCN(
            in_channels=node_features,
            out_channels=node_features,
            num_relations=num_relations,
            num_bases=num_bases
        )

        # L1 → L2 pooling (abstraction)
        self.pool_l1_to_l2 = SymmetricGraphPooling(
            in_channels=node_features,
            ratio=pool_ratio  # Reduces to ~500 nodes
        )

        # L2: Actions/Behavior
        self.rgcn_l2 = ConfidenceWeightedRGCN(
            in_channels=node_features,
            out_channels=node_features,
            num_relations=num_relations,
            num_bases=num_bases
        )

        # L2 → L3 pooling (further abstraction)
        self.pool_l2_to_l3 = SymmetricGraphPooling(
            in_channels=node_features,
            ratio=pool_ratio  # Reduces to ~250 nodes
        )

        # L3: Capabilities/Skills (most abstract in upper trifold)
        self.rgcn_l3 = ConfidenceWeightedRGCN(
            in_channels=node_features,
            out_channels=node_features,
            num_relations=num_relations,
            num_bases=num_bases
        )

        # ===== LOWER TRIFOLD (WHAT: top-down, abstract → concrete) =====
        # L6: Purpose/Mission (most abstract, learned prior)
        self.l6_prior = nn.Parameter(torch.randn(1, node_features))

        # L6 → L5 unpooling (initial concretization)
        self.unpool_l6_to_l5 = nn.Linear(node_features, node_features)

        # L5: Goals/Identity
        self.rgcn_l5 = ConfidenceWeightedRGCN(
            in_channels=node_features,
            out_channels=node_features,
            num_relations=num_relations,
            num_bases=num_bases
        )

        # L5 → L4 unpooling (further concretization)
        self.unpool_l5_to_l4 = nn.Linear(node_features, node_features)

        # L4: Plans/Beliefs
        self.rgcn_l4 = ConfidenceWeightedRGCN(
            in_channels=node_features,
            out_channels=node_features,
            num_relations=num_relations,
            num_bases=num_bases
        )

        # ===== FUSION HINGES (size-aligned, scale-normalized) =====
        # Hinge 1: L1 ↔ L6 (Environment ↔ Mission)
        self.hinge_l1_l6 = ChiralHingeExchange(dim=node_features, dropout=dropout)

        # Hinge 2: L2 ↔ L5 (Behavior ↔ Identity)
        self.hinge_l2_l5 = ChiralHingeExchange(dim=node_features, dropout=dropout)

        # Hinge 3: L3 ↔ L4 (Capability ↔ Beliefs)
        self.hinge_l3_l4 = ChiralHingeExchange(dim=node_features, dropout=dropout)

        # ===== SIZE ALIGNMENT LAYERS =====
        # For L1 ↔ L6: L1 has ~1000 nodes, L6 might be smaller
        # Use adaptive pooling to match sizes
        self.align_l6_to_l1 = nn.Linear(node_features, node_features)

        # For L3 ↔ L4: Both should be ~250 nodes (aligned naturally)
        # No special alignment needed

        # ===== MULTI-LEVEL PREDICTION HEADS =====
        # Auxiliary head from L1 (most concrete)
        if task_type == 'classification':
            self.predictor_l1 = nn.Sequential(
                nn.Linear(node_features, node_features // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(node_features // 2, num_classes)
            )

            # Auxiliary head from L2 (intermediate)
            self.predictor_l2 = nn.Sequential(
                nn.Linear(node_features, node_features // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(node_features // 2, num_classes)
            )

            # Main head from L3 (most abstract)
            self.predictor_l3 = nn.Sequential(
                nn.Linear(node_features, node_features // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(node_features // 2, num_classes)
            )
        else:
            self.predictor_l1 = nn.Sequential(
                nn.Linear(node_features, node_features // 2),
                nn.ReLU(),
                nn.Linear(node_features // 2, 1)
            )
            self.predictor_l2 = nn.Sequential(
                nn.Linear(node_features, node_features // 2),
                nn.ReLU(),
                nn.Linear(node_features // 2, 1)
            )
            self.predictor_l3 = nn.Sequential(
                nn.Linear(node_features, node_features // 2),
                nn.ReLU(),
                nn.Linear(node_features // 2, 1)
            )

        # ===== CYCLE RECONSTRUCTION HEADS =====
        # Upper trifold: L1 → L3 → L1
        self.reconstruct_l1_from_l3 = nn.Linear(node_features, node_features)

        # Lower trifold: L6 → L4 → L6
        self.reconstruct_l6_from_l4 = nn.Linear(node_features, node_features)

        # Cross-trifold: L1 ↔ L6 consistency
        self.reconstruct_l1_from_l6 = nn.Linear(node_features, node_features)
        self.reconstruct_l6_from_l1 = nn.Linear(node_features, node_features)

    def _normalize_features(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize features to [0, 1] range for scale-invariant exchange.

        Args:
            x: Input features [num_nodes, dim]

        Returns:
            (x_normalized, min_val, max_val): Normalized features and scale params
        """
        min_val = x.min(dim=0, keepdim=True)[0]
        max_val = x.max(dim=0, keepdim=True)[0]

        # Avoid division by zero
        scale = max_val - min_val
        scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)

        x_normalized = (x - min_val) / scale

        return x_normalized, min_val, max_val

    def _denormalize_features(
        self,
        x_normalized: torch.Tensor,
        min_val: torch.Tensor,
        max_val: torch.Tensor
    ) -> torch.Tensor:
        """
        Denormalize features back to original scale.

        Args:
            x_normalized: Normalized features [num_nodes, dim]
            min_val: Minimum values from normalization
            max_val: Maximum values from normalization

        Returns:
            x: Denormalized features
        """
        scale = max_val - min_val
        scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)

        return x_normalized * scale + min_val

    def _align_sizes(
        self,
        x_small: torch.Tensor,
        x_large: torch.Tensor,
        perm_large: torch.Tensor
    ) -> torch.Tensor:
        """
        Align smaller tensor to match larger tensor's size via interpolation.

        Args:
            x_small: Smaller tensor [num_small, dim]
            x_large: Larger tensor [num_large, dim]
            perm_large: Permutation indices from pooling [num_large]

        Returns:
            x_aligned: Small tensor aligned to large size [num_large, dim]
        """
        num_small = x_small.size(0)
        num_large = x_large.size(0)
        dim = x_small.size(1)

        if num_small == num_large:
            return x_small

        # Broadcast smaller to match larger via learned transform + interpolation
        x_aligned = torch.zeros(num_large, dim, device=x_small.device, dtype=x_small.dtype)

        # Map each large node to nearest small node (simple nearest neighbor)
        indices = (torch.arange(num_large, device=x_small.device).float() * (num_small / num_large)).long()
        indices = torch.clamp(indices, 0, num_small - 1)

        x_aligned = x_small[indices]

        return x_aligned

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
            x_l6_prior: Optional custom L6 prior [1, node_features]

        Returns:
            Dictionary with:
                'logits': Final ensemble prediction
                'logits_l1', 'logits_l2', 'logits_l3': Auxiliary predictions
                'cycle_loss_upper': L1 → L3 → L1 reconstruction
                'cycle_loss_lower': L6 → L4 → L6 reconstruction
                'cycle_loss_cross': L1 ↔ L6 consistency
                All intermediate level representations
        """
        num_nodes = x.size(0)

        # Default batch if not provided
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

        # ===== UPPER TRIFOLD: L1 → L2 → L3 (WHY operation) =====
        # L1: Message passing
        x_l1 = self.rgcn_l1(x, edge_index, edge_type)

        # L1 → L2: Pool (abstraction)
        x_l2_up, edge_index_l2, edge_type_l2, batch_l2, perm_l2, score_l2 = self.pool_l1_to_l2.why_operation(
            x_l1, edge_index, edge_attr=edge_type, batch=batch
        )

        # L2: Message passing
        x_l2_up = self.rgcn_l2(x_l2_up, edge_index_l2, edge_type_l2)

        # L2 → L3: Pool (further abstraction)
        x_l3_up, edge_index_l3, edge_type_l3, batch_l3, perm_l3, score_l3 = self.pool_l2_to_l3.why_operation(
            x_l2_up, edge_index_l2, edge_attr=edge_type_l2, batch=batch_l2
        )

        # L3: Message passing
        x_l3_up = self.rgcn_l3(x_l3_up, edge_index_l3, edge_type_l3)

        # ===== LOWER TRIFOLD: L6 → L5 → L4 (WHAT operation) =====
        # L6: Start with prior (or custom if provided)
        num_l3_nodes = x_l3_up.size(0)  # Match L3 size for hinge 3

        if x_l6_prior is not None:
            x_l6 = x_l6_prior.expand(num_l3_nodes, -1)
        else:
            x_l6 = self.l6_prior.expand(num_l3_nodes, -1)

        # L6 → L5: Unpool (initial concretization)
        num_l2_nodes = x_l2_up.size(0)  # Match L2 size for hinge 2
        x_l5_down = self.unpool_l6_to_l5(x_l6)

        # Broadcast L5 to match L2 size
        if x_l5_down.size(0) < num_l2_nodes:
            x_l5_down = self._align_sizes(x_l5_down, x_l2_up, perm_l2)

        # L5: Message passing (on L2 graph structure)
        x_l5_down = self.rgcn_l5(x_l5_down, edge_index_l2, edge_type_l2)

        # L5 → L4: Unpool (further concretization)
        x_l4_down = self.unpool_l5_to_l4(x_l5_down)

        # Broadcast L4 to match L3 size (should already match)
        if x_l4_down.size(0) != num_l3_nodes:
            x_l4_down = self._align_sizes(x_l4_down, x_l3_up, perm_l3)

        # L4: Message passing (on L3 graph structure)
        x_l4_down = self.rgcn_l4(x_l4_down, edge_index_l3, edge_type_l3)

        # ===== HINGE EXCHANGES (with scale normalization) =====

        # Hinge 3: L3 ↔ L4 (Capability ↔ Beliefs)
        x_l3_norm, min_l3, max_l3 = self._normalize_features(x_l3_up)
        x_l4_norm, min_l4, max_l4 = self._normalize_features(x_l4_down)

        x_l3_refined_norm, x_l4_refined_norm = self.hinge_l3_l4(x_l3_norm, x_l4_norm)

        x_l3_refined = self._denormalize_features(x_l3_refined_norm, min_l3, max_l3)
        x_l4_refined = self._denormalize_features(x_l4_refined_norm, min_l4, max_l4)

        # Hinge 2: L2 ↔ L5 (Behavior ↔ Identity)
        x_l2_norm, min_l2, max_l2 = self._normalize_features(x_l2_up)
        x_l5_norm, min_l5, max_l5 = self._normalize_features(x_l5_down)

        x_l2_refined_norm, x_l5_refined_norm = self.hinge_l2_l5(x_l2_norm, x_l5_norm)

        x_l2_refined = self._denormalize_features(x_l2_refined_norm, min_l2, max_l2)
        x_l5_refined = self._denormalize_features(x_l5_refined_norm, min_l5, max_l5)

        # Hinge 1: L1 ↔ L6 (Environment ↔ Mission)
        # Need to align L6 to L1 size
        num_l1_nodes = x_l1.size(0)
        x_l6_aligned = self._align_sizes(x_l6, x_l1, perm_l2)

        x_l1_norm, min_l1, max_l1 = self._normalize_features(x_l1)
        x_l6_aligned_norm, min_l6_aligned, max_l6_aligned = self._normalize_features(x_l6_aligned)

        x_l1_refined_norm, x_l6_refined_norm = self.hinge_l1_l6(x_l1_norm, x_l6_aligned_norm)

        x_l1_refined = self._denormalize_features(x_l1_refined_norm, min_l1, max_l1)
        x_l6_refined = self._denormalize_features(x_l6_refined_norm, min_l6_aligned, max_l6_aligned)

        # ===== MULTI-LEVEL PREDICTIONS =====
        # Global pooling at each level
        x_l1_graph = global_mean_pool(x_l1_refined, batch)
        x_l2_graph = global_mean_pool(x_l2_refined, batch_l2)
        x_l3_graph = global_mean_pool(x_l3_refined, batch_l3)

        # Predictions from each level
        logits_l1 = self.predictor_l1(x_l1_graph)
        logits_l2 = self.predictor_l2(x_l2_graph)
        logits_l3 = self.predictor_l3(x_l3_graph)

        # Ensemble prediction (average of all 3 heads)
        logits_ensemble = (logits_l1 + logits_l2 + logits_l3) / 3

        # ===== CYCLE CONSISTENCY LOSSES =====

        # Upper trifold cycle: L1 → L3 → L1
        x_l1_reconstructed_from_l3 = torch.zeros_like(x_l1)
        # Unpool L3 back through L2 to L1
        x_l3_to_l2 = torch.zeros(num_l2_nodes, self.node_features, device=x_l1.device)
        x_l3_to_l2[perm_l3] = x_l3_refined

        x_l2_to_l1 = torch.zeros_like(x_l1)
        x_l2_to_l1[perm_l2] = self.reconstruct_l1_from_l3(x_l3_to_l2)

        cycle_loss_upper = F.mse_loss(x_l2_to_l1, x_l1)

        # Lower trifold cycle: L6 → L4 → L6
        x_l6_reconstructed_from_l4 = self.reconstruct_l6_from_l4(x_l4_refined)
        cycle_loss_lower = F.mse_loss(x_l6_reconstructed_from_l4, x_l6)

        # Cross-trifold cycle: L1 ↔ L6
        x_l1_reconstructed_from_l6 = self.reconstruct_l1_from_l6(x_l6_refined)
        x_l6_reconstructed_from_l1 = self.reconstruct_l6_from_l1(x_l1_refined[:x_l6.size(0)])  # Trim to L6 size

        cycle_loss_cross = (
            F.mse_loss(x_l1_reconstructed_from_l6, x_l1) +
            F.mse_loss(x_l6_reconstructed_from_l1, x_l6)
        ) / 2

        return {
            # Final predictions
            'logits': logits_ensemble,
            'logits_l1': logits_l1,
            'logits_l2': logits_l2,
            'logits_l3': logits_l3,

            # Cycle losses
            'cycle_loss_upper': cycle_loss_upper,
            'cycle_loss_lower': cycle_loss_lower,
            'cycle_loss_cross': cycle_loss_cross,

            # Level representations (for analysis)
            'x_l1': x_l1_refined,
            'x_l2': x_l2_refined,
            'x_l3': x_l3_refined,
            'x_l4': x_l4_refined,
            'x_l5': x_l5_refined,
            'x_l6': x_l6_refined,

            # Pooling info (for unpooling if needed)
            'perm_l2': perm_l2,
            'perm_l3': perm_l3,
            'batch_l2': batch_l2,
            'batch_l3': batch_l3
        }


# Export public API
__all__ = [
    'ChiralHingeExchange',
    'MinimalChiralModel',
    'FullChiralModel'
]
