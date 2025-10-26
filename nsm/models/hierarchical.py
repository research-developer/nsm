"""
Symmetric hierarchical layer implementing WHY/WHAT operations.

Integrates R-GCN message passing, coupling layers, confidence propagation,
and graph pooling into a unified architecture for 2-level hierarchy reasoning.

WHY operation (abstraction): Concrete (L1) → Abstract (L2)
WHAT operation (concretization): Abstract (L2) → Concrete (L1)

Key constraint: ||WHY(WHAT(x)) - x||² < 0.2 (20% reconstruction error)

Architecture:
1. R-GCN message passing at concrete level
2. Coupling transformation (information-preserving)
3. Graph pooling to abstract level
4. Inverse coupling back to concrete representation
5. Cycle consistency verification

References:
    NSM-20: Phase 1 Foundation architectural blueprint
    NSM-5: Adjoint functors for WHY ⊣ WHAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, Any

from .rgcn import ConfidenceWeightedRGCN, HierarchicalRGCN
from .coupling import AffineCouplingLayer, MultiLayerCoupling
from .pooling import SymmetricGraphPooling
from .confidence.base import BaseSemiring
from .confidence.examples import ProductSemiring


class SymmetricHierarchicalLayer(nn.Module):
    """Symmetric WHY/WHAT operations for 2-level hierarchy.

    Implements adjoint functors: WHY ⊣ WHAT
    - WHY: Abstraction (concrete → abstract) via pooling
    - WHAT: Concretization (abstract → concrete) via unpooling

    Architecture:
    - R-GCN for relational message passing
    - Coupling layers for invertible transformations
    - Graph pooling for hierarchical coarsening
    - Semiring for confidence propagation

    Args:
        node_features (int): Node feature dimensionality
        num_relations (int): Number of edge types (domain-specific)
        num_bases (int, optional): R-GCN basis count for parameter reduction
        pool_ratio (float): Fraction of nodes to keep at L2 (default: 0.5)
        coupling_layers (int): Number of coupling layers (default: 3)
        hidden_dim (int): Hidden dimension for coupling/R-GCN
        semiring (BaseSemiring, optional): Confidence propagation semiring
        dropout (float): Dropout rate for regularization

    Example:
        >>> layer = SymmetricHierarchicalLayer(
        ...     node_features=64,
        ...     num_relations=16,
        ...     pool_ratio=0.5
        ... )
        >>>
        >>> # WHY operation: Abstract to goals/mechanisms
        >>> x_abstract, edge_index_abstract, perm, score = layer.why_operation(
        ...     x, edge_index, edge_type, edge_attr, batch
        ... )
        >>>
        >>> # WHAT operation: Concretize back to actions/observations
        >>> x_concrete = layer.what_operation(
        ...     x_abstract, perm, batch, original_num_nodes=x.size(0)
        ... )
        >>>
        >>> # Verify cycle consistency
        >>> recon_error = F.mse_loss(x_concrete, x)
        >>> assert recon_error < 0.2, f"Reconstruction error {recon_error} > 20%"
    """

    def __init__(
        self,
        node_features: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        pool_ratio: float = 0.5,
        coupling_layers: int = 3,
        hidden_dim: int = 128,
        semiring: Optional[BaseSemiring] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        self.node_features = node_features
        self.num_relations = num_relations
        self.pool_ratio = pool_ratio

        # Default: num_bases ≈ 25-30% of num_relations
        if num_bases is None:
            num_bases = max(4, num_relations // 3)

        self.num_bases = num_bases

        # R-GCN for message passing at concrete level (L1)
        self.rgcn_l1 = ConfidenceWeightedRGCN(
            in_channels=node_features,
            out_channels=node_features,
            num_relations=num_relations,
            num_bases=num_bases,
            aggr='mean',
            bias=True
        )

        # R-GCN for message passing at abstract level (L2)
        self.rgcn_l2 = ConfidenceWeightedRGCN(
            in_channels=node_features,
            out_channels=node_features,
            num_relations=num_relations,  # Same relations at both levels
            num_bases=num_bases,
            aggr='mean',
            bias=True
        )

        # Coupling layers for information-preserving transformations
        self.coupling_forward = MultiLayerCoupling(
            features=node_features,
            num_layers=coupling_layers,
            hidden_dim=hidden_dim,
            activation='relu'
        )

        self.coupling_inverse = MultiLayerCoupling(
            features=node_features,
            num_layers=coupling_layers,
            hidden_dim=hidden_dim,
            activation='relu'
        )

        # Graph pooling for WHY operation
        self.pooling = SymmetricGraphPooling(
            in_channels=node_features,
            ratio=pool_ratio
        )

        # Confidence propagation
        self.semiring = semiring or ProductSemiring(temperature=1.0)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Layer normalization for stability
        self.norm_l1 = nn.LayerNorm(node_features)
        self.norm_l2 = nn.LayerNorm(node_features)

    def why_operation(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor]:
        """WHY operation: Abstract from concrete (L1) to abstract (L2).

        Steps:
        1. R-GCN message passing at L1
        2. Coupling transformation (forward)
        3. Graph pooling to L2
        4. R-GCN message passing at L2

        Args:
            x (Tensor): Node features [num_nodes, node_features]
            edge_index (Tensor): Edge connectivity [2, num_edges]
            edge_type (Tensor): Edge types [num_edges]
            edge_attr (Tensor, optional): Edge confidence [num_edges]
            batch (Tensor, optional): Batch assignment [num_nodes]

        Returns:
            Tuple containing:
            - x_abstract (Tensor): Abstract node features [num_pooled, node_features]
            - edge_index_abstract (Tensor): Abstract edge index [2, num_pooled_edges]
            - edge_attr_abstract (Tensor, optional): Abstract edge attributes
            - perm (Tensor): Pooling indices
            - score (Tensor): Node selection scores
        """
        # Step 1: R-GCN message passing at L1 (concrete level)
        x_l1 = self.rgcn_l1(x, edge_index, edge_type, edge_attr)
        x_l1 = self.norm_l1(x_l1)
        x_l1 = F.relu(x_l1)
        x_l1 = self.dropout(x_l1)

        # Step 2: Coupling transformation (information-preserving)
        x_coupled = self.coupling_forward.forward(x_l1)

        # Step 3: Graph pooling to L2 (abstract level)
        x_abstract, edge_index_abstract, edge_attr_abstract, batch_abstract, perm, score = \
            self.pooling.why_operation(x_coupled, edge_index, edge_attr, batch)

        # Step 4: R-GCN message passing at L2 (refine abstract representation)
        if edge_index_abstract.size(1) > 0:  # Check if any edges remain
            # Need edge types for abstract level - reindex from original
            # For now, use placeholder edge types (all same type)
            edge_type_abstract = torch.zeros(
                edge_index_abstract.size(1),
                dtype=torch.long,
                device=edge_index_abstract.device
            )

            x_abstract = self.rgcn_l2(
                x_abstract,
                edge_index_abstract,
                edge_type_abstract,
                edge_attr_abstract
            )
            x_abstract = self.norm_l2(x_abstract)
            x_abstract = F.relu(x_abstract)

        return x_abstract, edge_index_abstract, edge_attr_abstract, perm, score

    def what_operation(
        self,
        x_abstract: Tensor,
        perm: Tensor,
        batch: Optional[Tensor] = None,
        original_num_nodes: Optional[int] = None
    ) -> Tensor:
        """WHAT operation: Concretize from abstract (L2) to concrete (L1).

        Steps:
        1. Unpool from L2 to L1
        2. Inverse coupling transformation

        Args:
            x_abstract (Tensor): Abstract features [num_pooled, node_features]
            perm (Tensor): Pooling indices (from why_operation)
            batch (Tensor, optional): Original batch assignment
            original_num_nodes (int, optional): Original node count

        Returns:
            Tensor: Reconstructed concrete features [num_nodes, node_features]
        """
        # Step 1: Unpool to L1
        x_unpooled = self.pooling.what_operation(
            x_abstract,
            perm,
            batch,
            original_num_nodes
        )

        # Step 2: Inverse coupling transformation
        x_concrete = self.coupling_inverse.inverse(x_unpooled)

        return x_concrete

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        return_cycle_loss: bool = False
    ) -> Dict[str, Any]:
        """Full forward pass with optional cycle consistency computation.

        Args:
            x (Tensor): Input node features
            edge_index (Tensor): Edge connectivity
            edge_type (Tensor): Edge types
            edge_attr (Tensor, optional): Edge attributes
            batch (Tensor, optional): Batch assignment
            return_cycle_loss (bool): Whether to compute cycle consistency loss

        Returns:
            Dict containing:
            - x_abstract: Abstract representations
            - x_reconstructed: Reconstructed concrete features (if return_cycle_loss)
            - cycle_loss: Reconstruction error (if return_cycle_loss)
            - perm: Pooling indices
            - score: Node selection scores
        """
        original_num_nodes = x.size(0)

        # WHY operation
        x_abstract, edge_index_abstract, edge_attr_abstract, perm, score = \
            self.why_operation(x, edge_index, edge_type, edge_attr, batch)

        result = {
            'x_abstract': x_abstract,
            'edge_index_abstract': edge_index_abstract,
            'edge_attr_abstract': edge_attr_abstract,
            'perm': perm,
            'score': score
        }

        if return_cycle_loss:
            # WHAT operation
            x_reconstructed = self.what_operation(
                x_abstract,
                perm,
                batch,
                original_num_nodes
            )

            # Compute cycle consistency loss
            cycle_loss = self.pooling.cycle_loss(x, x_reconstructed)

            result['x_reconstructed'] = x_reconstructed
            result['cycle_loss'] = cycle_loss

        return result

    def get_reconstruction_error(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None
    ) -> Tensor:
        """Compute reconstruction error: ||WHY(WHAT(x)) - x||² / ||x||².

        This is the key metric for WHY/WHAT symmetry.
        Target: < 0.2 (20% error)

        Args:
            x (Tensor): Original features
            edge_index, edge_type, edge_attr, batch: Graph data

        Returns:
            Tensor: Normalized reconstruction error (scalar)
        """
        result = self.forward(
            x, edge_index, edge_type, edge_attr, batch,
            return_cycle_loss=True
        )

        # Normalized by original norm
        x_norm = torch.norm(x, p=2)
        error_norm = torch.norm(result['x_reconstructed'] - x, p=2)

        normalized_error = error_norm / (x_norm + 1e-8)

        return normalized_error

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  node_features={self.node_features},\n'
                f'  num_relations={self.num_relations},\n'
                f'  num_bases={self.num_bases},\n'
                f'  pool_ratio={self.pool_ratio:.2f},\n'
                f'  semiring={self.semiring.get_name()}\n'
                f')')


class NSMModel(nn.Module):
    """Full Neural Symbolic Model for Phase 1.5 (3-level hierarchy).

    Integrates all components:
    - Two SymmetricHierarchicalLayers for L1↔L2↔L3
    - Dual-pass prediction (abstract + concrete fusion) by default
    - Task-specific prediction heads
    - Confidence-aware output

    Args:
        node_features (int): Node feature dimensionality
        num_relations (int): Number of edge types
        num_classes (int): Number of output classes for task
        num_bases (int, optional): R-GCN basis count
        pool_ratio (float): Pooling ratio for each level
        task_type (str): 'classification', 'regression', or 'link_prediction'
        num_levels (int): Number of hierarchy levels (2 or 3, default 3)
        use_dual_pass (bool): Use dual-pass prediction (default True)
        fusion_mode (str): Fusion strategy for dual-pass ('equal', 'learned', 'abstract_only', 'concrete_only')

    Example:
        >>> # Dual-pass mode (default)
        >>> model = NSMModel(
        ...     node_features=64,
        ...     num_relations=16,
        ...     num_classes=2,
        ...     task_type='classification',
        ...     num_levels=3
        ... )
        >>>
        >>> # Single-pass mode (opt-out)
        >>> model = NSMModel(
        ...     node_features=64,
        ...     num_relations=16,
        ...     num_classes=2,
        ...     task_type='classification',
        ...     num_levels=3,
        ...     use_dual_pass=False
        ... )
        >>>
        >>> # Forward pass
        >>> output = model(x, edge_index, edge_type, edge_attr, batch)
        >>> logits = output['logits']
        >>> cycle_loss = output['cycle_loss']
        >>>
        >>> # Training loss
        >>> task_loss = F.cross_entropy(logits, labels)
        >>> total_loss = task_loss + 0.01 * cycle_loss
    """

    def __init__(
        self,
        node_features: int,
        num_relations: int,
        num_classes: int,
        num_bases: Optional[int] = None,
        pool_ratio: float = 0.5,
        task_type: str = 'classification',
        num_levels: int = 3,
        use_dual_pass: bool = True,
        fusion_mode: str = 'equal'
    ):
        super().__init__()

        self.node_features = node_features
        self.num_relations = num_relations
        self.num_classes = num_classes
        self.task_type = task_type
        self.num_levels = num_levels
        self.use_dual_pass = use_dual_pass
        self.fusion_mode = fusion_mode

        # L1 ↔ L2 hierarchical layer
        self.layer_1_2 = SymmetricHierarchicalLayer(
            node_features=node_features,
            num_relations=num_relations,
            num_bases=num_bases,
            pool_ratio=pool_ratio
        )

        # L2 ↔ L3 hierarchical layer (only if num_levels == 3)
        if num_levels >= 3:
            self.layer_2_3 = SymmetricHierarchicalLayer(
                node_features=node_features,
                num_relations=num_relations,
                num_bases=num_bases,
                pool_ratio=pool_ratio
            )
        else:
            self.layer_2_3 = None

        # Task-specific prediction head
        if task_type == 'classification':
            self.predictor = nn.Sequential(
                nn.Linear(node_features, node_features // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(node_features // 2, num_classes)
            )
        elif task_type == 'regression':
            self.predictor = nn.Sequential(
                nn.Linear(node_features, node_features // 2),
                nn.ReLU(),
                nn.Linear(node_features // 2, 1)
            )
        elif task_type == 'link_prediction':
            # Binary classification for edge existence
            self.predictor = nn.Sequential(
                nn.Linear(node_features, node_features // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(node_features // 2, num_classes)
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        # Dual-pass prediction heads (only if use_dual_pass=True)
        if use_dual_pass:
            if task_type in ['classification', 'link_prediction']:
                self.predictor_abstract = nn.Sequential(
                    nn.Linear(node_features, node_features // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(node_features // 2, num_classes)
                )
                self.predictor_concrete = nn.Sequential(
                    nn.Linear(node_features, node_features // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(node_features // 2, num_classes)
                )
            elif task_type == 'regression':
                self.predictor_abstract = nn.Sequential(
                    nn.Linear(node_features, node_features // 2),
                    nn.ReLU(),
                    nn.Linear(node_features // 2, 1)
                )
                self.predictor_concrete = nn.Sequential(
                    nn.Linear(node_features, node_features // 2),
                    nn.ReLU(),
                    nn.Linear(node_features // 2, 1)
                )

            # Learned fusion weights (if fusion_mode='learned')
            if fusion_mode == 'learned':
                self.fusion_attention = nn.Sequential(
                    nn.Linear(node_features * 2, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2),
                    nn.Softmax(dim=-1)
                )
        else:
            self.predictor_abstract = None
            self.predictor_concrete = None
            self.fusion_attention = None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None
    ) -> Dict[str, Any]:
        """Full forward pass with task prediction and cycle loss.

        For 3-level hierarchy:
        L1 (concrete) → WHY → L2 (mid) → WHY → L3 (abstract)
        L3 (abstract) → WHAT → L2 (mid) → WHAT → L1 (concrete)

        For 2-level hierarchy:
        L1 (concrete) → WHY → L2 (abstract) → WHAT → L1 (concrete)

        Args:
            x, edge_index, edge_type, edge_attr, batch: Graph data

        Returns:
            Dict containing:
            - logits: Task predictions
            - cycle_loss: Total reconstruction error across all levels
            - x_l2: L2 representations
            - x_l3: L3 representations (if num_levels == 3)
        """
        original_num_nodes = x.size(0)

        if self.num_levels == 2:
            # 2-level hierarchy (backwards compatible)
            result = self.layer_1_2.forward(
                x, edge_index, edge_type, edge_attr, batch,
                return_cycle_loss=True
            )

            # Task prediction from L2 (abstract)
            x_abstract = result['x_abstract']
            perm_l2 = result['perm']

        else:  # num_levels == 3
            # L1 → L2 (WHY operation)
            result_l2 = self.layer_1_2.why_operation(
                x, edge_index, edge_type, edge_attr, batch
            )

            x_l2 = result_l2[0]
            edge_index_l2 = result_l2[1]
            edge_attr_l2 = result_l2[2]
            perm_l2 = result_l2[3]
            score_l2 = result_l2[4]

            # Determine batch_l2 for L2 level
            if batch is not None:
                batch_l2 = batch[perm_l2]
            else:
                batch_l2 = None

            # Determine edge types for L2 level (placeholder for now)
            if edge_index_l2.size(1) > 0:
                edge_type_l2 = torch.zeros(
                    edge_index_l2.size(1),
                    dtype=torch.long,
                    device=edge_index_l2.device
                )
            else:
                edge_type_l2 = torch.tensor([], dtype=torch.long, device=x.device)

            # L2 → L3 (WHY operation)
            result_l3 = self.layer_2_3.why_operation(
                x_l2, edge_index_l2, edge_type_l2, edge_attr_l2, batch_l2
            )

            x_l3 = result_l3[0]
            edge_index_l3 = result_l3[1]
            edge_attr_l3 = result_l3[2]
            perm_l3 = result_l3[3]
            score_l3 = result_l3[4]

            # Determine batch_l3 for L3 level
            if batch_l2 is not None:
                batch_l3 = batch_l2[perm_l3]
            else:
                batch_l3 = None

            # L3 → L2 (WHAT operation)
            x_l2_reconstructed = self.layer_2_3.what_operation(
                x_l3, perm_l3, batch_l2, original_num_nodes=x_l2.size(0)
            )

            # L2 → L1 (WHAT operation)
            x_l1_reconstructed = self.layer_1_2.what_operation(
                x_l2_reconstructed, perm_l2, batch, original_num_nodes=original_num_nodes
            )

            # Compute 3-level cycle consistency loss
            # L1 cycle: L1 → L2 → L3 → L2 → L1
            cycle_loss_l1 = self.layer_1_2.pooling.cycle_loss(x, x_l1_reconstructed)

            # L2 cycle: L2 → L3 → L2
            cycle_loss_l2 = self.layer_2_3.pooling.cycle_loss(x_l2, x_l2_reconstructed)

            # Total cycle loss (weighted average)
            cycle_loss = 0.7 * cycle_loss_l1 + 0.3 * cycle_loss_l2

            # DUAL-PASS MODE: Make predictions from both abstract and concrete levels
            if self.use_dual_pass:
                # Pass 1 prediction: From L3 (abstract, after bottom-up)
                from torch_geometric.nn import global_mean_pool
                x_graph_abstract = global_mean_pool(x_l3, batch_l3) if batch is not None else x_l3.mean(dim=0, keepdim=True)
                logits_abstract = self.predictor_abstract(x_graph_abstract)

                # Pass 2 prediction: From L1' (concrete, after top-down reconstruction)
                x_graph_concrete = global_mean_pool(x_l1_reconstructed, batch) if batch is not None else x_l1_reconstructed.mean(dim=0, keepdim=True)
                logits_concrete = self.predictor_concrete(x_graph_concrete)

                # Fusion of predictions
                if self.fusion_mode == 'equal':
                    # Equal weighting
                    logits_fused = 0.5 * logits_abstract + 0.5 * logits_concrete
                    fusion_weights = (0.5, 0.5)
                elif self.fusion_mode == 'learned':
                    # Learned attention-based fusion
                    fusion_input = torch.cat([x_graph_abstract, x_graph_concrete], dim=-1)
                    weights = self.fusion_attention(fusion_input)  # [batch, 2]
                    alpha, beta = weights[:, 0:1], weights[:, 1:2]
                    logits_fused = alpha * logits_abstract + beta * logits_concrete
                    fusion_weights = (alpha.mean().item(), beta.mean().item())
                elif self.fusion_mode == 'abstract_only':
                    # Ablation: only use abstract prediction
                    logits_fused = logits_abstract
                    fusion_weights = (1.0, 0.0)
                elif self.fusion_mode == 'concrete_only':
                    # Ablation: only use concrete prediction
                    logits_fused = logits_concrete
                    fusion_weights = (0.0, 1.0)
                else:
                    raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

                # Store all predictions for multi-task loss
                result = {
                    'x_l2': x_l2,
                    'x_l3': x_l3,
                    'x_l1_reconstructed': x_l1_reconstructed,
                    'x_l2_reconstructed': x_l2_reconstructed,
                    'cycle_loss': cycle_loss,
                    'cycle_loss_l1': cycle_loss_l1,
                    'cycle_loss_l2': cycle_loss_l2,
                    'perm_l2': perm_l2,
                    'perm_l3': perm_l3,
                    'logits': logits_fused,  # Fused prediction is the main output
                    'logits_abstract': logits_abstract,
                    'logits_concrete': logits_concrete,
                    'fusion_weights': fusion_weights
                }

                # Use fused prediction for backward compatibility
                x_abstract = x_l3  # For later graph pooling (not used in dual-pass)
                perm_abstract = perm_l3

            else:
                # SINGLE-PASS MODE (original behavior)
                # Task prediction from L3 (most abstract)
                x_abstract = x_l3
                perm_abstract = perm_l3

                # Store results for analysis
                result = {
                    'x_l2': x_l2,
                    'x_l3': x_l3,
                    'x_l1_reconstructed': x_l1_reconstructed,
                    'x_l2_reconstructed': x_l2_reconstructed,
                    'cycle_loss': cycle_loss,
                    'cycle_loss_l1': cycle_loss_l1,
                    'cycle_loss_l2': cycle_loss_l2,
                    'perm_l2': perm_l2,
                    'perm_l3': perm_l3
                }

        # Task prediction from most abstract level (only if NOT using dual-pass)
        if not self.use_dual_pass:
            if self.task_type in ['classification', 'regression']:
                # Graph-level prediction: global pooling
                if batch is not None:
                    from torch_geometric.nn import global_mean_pool
                    if self.num_levels == 3:
                        batch_abstract = batch_l3
                    else:
                        batch_abstract = batch[perm_l2]
                    x_graph = global_mean_pool(x_abstract, batch_abstract)
                else:
                    # Single graph: mean pooling
                    x_graph = x_abstract.mean(dim=0, keepdim=True)

                logits = self.predictor(x_graph)

            elif self.task_type == 'link_prediction':
                # Graph-level binary prediction (edge exists/doesn't exist)
                if batch is not None:
                    from torch_geometric.nn import global_mean_pool
                    if self.num_levels == 3:
                        batch_abstract = batch_l3
                    else:
                        batch_abstract = batch[perm_l2]
                    x_graph = global_mean_pool(x_abstract, batch_abstract)
                else:
                    # Single graph: mean pooling
                    x_graph = x_abstract.mean(dim=0, keepdim=True)

                logits = self.predictor(x_graph)

            result['logits'] = logits

        # Add x_abstract to result for both modes
        result['x_abstract'] = x_abstract

        return result

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  node_features={self.node_features},\n'
                f'  num_relations={self.num_relations},\n'
                f'  num_classes={self.num_classes},\n'
                f'  num_levels={self.num_levels},\n'
                f'  task_type={self.task_type}\n'
                f')')
