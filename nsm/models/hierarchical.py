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
    """Full Neural Symbolic Model for Phase 1 (2-level hierarchy).

    Integrates all components:
    - SymmetricHierarchicalLayer for WHY/WHAT
    - Task-specific prediction heads
    - Confidence-aware output

    Args:
        node_features (int): Node feature dimensionality
        num_relations (int): Number of edge types
        num_classes (int): Number of output classes for task
        num_bases (int, optional): R-GCN basis count
        pool_ratio (float): Pooling ratio
        task_type (str): 'classification', 'regression', or 'link_prediction'

    Example:
        >>> model = NSMModel(
        ...     node_features=64,
        ...     num_relations=16,
        ...     num_classes=2,
        ...     task_type='classification'
        ... )
        >>>
        >>> # Forward pass
        >>> output = model(x, edge_index, edge_type, edge_attr, batch)
        >>> logits = output['logits']
        >>> cycle_loss = output['cycle_loss']
        >>>
        >>> # Training loss
        >>> task_loss = F.cross_entropy(logits, labels)
        >>> total_loss = task_loss + 0.1 * cycle_loss
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

        # Core hierarchical layer
        self.hierarchical = SymmetricHierarchicalLayer(
            node_features=node_features,
            num_relations=num_relations,
            num_bases=num_bases,
            pool_ratio=pool_ratio
        )

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
            self.predictor = nn.Sequential(
                nn.Linear(node_features * 2, node_features),
                nn.ReLU(),
                nn.Linear(node_features, 1)
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None
    ) -> Dict[str, Any]:
        """Full forward pass with task prediction and cycle loss.

        Args:
            x, edge_index, edge_type, edge_attr, batch: Graph data

        Returns:
            Dict containing:
            - logits: Task predictions
            - cycle_loss: Reconstruction error
            - x_abstract: Abstract representations (for analysis)
        """
        # Hierarchical encoding
        result = self.hierarchical.forward(
            x, edge_index, edge_type, edge_attr, batch,
            return_cycle_loss=True
        )

        # Task prediction from abstract representations
        x_abstract = result['x_abstract']

        if self.task_type in ['classification', 'regression']:
            # Graph-level prediction: global pooling
            if batch is not None:
                # Batch-wise global pooling
                from torch_geometric.nn import global_mean_pool
                batch_abstract = batch[result['perm']]
                x_graph = global_mean_pool(x_abstract, batch_abstract)
            else:
                # Single graph: mean pooling
                x_graph = x_abstract.mean(dim=0, keepdim=True)

            logits = self.predictor(x_graph)

        elif self.task_type == 'link_prediction':
            # Edge-level prediction (placeholder - needs edge pairs)
            logits = None  # Requires specific edge pairs for prediction

        result['logits'] = logits

        return result

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  node_features={self.node_features},\n'
                f'  num_relations={self.num_relations},\n'
                f'  num_classes={self.num_classes},\n'
                f'  task_type={self.task_type}\n'
                f')')
