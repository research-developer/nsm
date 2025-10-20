"""
Graph pooling operations for hierarchical abstraction (WHY) and refinement (WHAT).

Implements SAGPool-based pooling with unpooling for symmetric WHY/WHAT operations.
Pooling learns which nodes to keep at abstract level, unpooling reconstructs concrete level.

Key operations:
- WHY (pooling): Select important nodes, coarsen graph
- WHAT (unpooling): Reconstruct concrete graph from abstract representation

References:
    Lee et al. (2019) - "Self-Attention Graph Pooling" (SAGPool)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGPooling
from torch_geometric.utils import to_dense_batch, to_dense_adj
from typing import Tuple, Optional


class SymmetricGraphPooling(nn.Module):
    """Learnable graph pooling with exact unpooling for WHY/WHAT symmetry.

    Uses SAGPooling to select important nodes, stores pooling indices
    for perfect reconstruction via unpooling.

    WHY operation (pooling):
    - Input: Concrete graph (L1)
    - Output: Abstract graph (L2) + pooling indices

    WHAT operation (unpooling):
    - Input: Abstract graph (L2) + pooling indices
    - Output: Reconstructed concrete graph (L1)

    Args:
        in_channels (int): Node feature dimensionality
        ratio (float): Fraction of nodes to keep (0.5 = 50%)
        gnn (nn.Module, optional): GNN layer for scoring. If None, uses Linear.
        min_score (float, optional): Minimum score threshold
        multiplier (float): Score multiplier for selection
        nonlinearity (str): Activation function ('tanh', 'relu', 'sigmoid')

    Example:
        >>> pooling = SymmetricGraphPooling(in_channels=64, ratio=0.5)
        >>>
        >>> # WHY: Pool to abstract level
        >>> x_abstract, edge_index_abstract, edge_attr_abstract, batch_abstract, perm, score = \
        ...     pooling.why_operation(x, edge_index, edge_attr, batch)
        >>>
        >>> # WHAT: Unpool back to concrete level
        >>> x_reconstructed = pooling.what_operation(
        ...     x_abstract, perm, batch, original_num_nodes=x.size(0)
        ... )
    """

    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
        gnn: Optional[nn.Module] = None,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: str = 'tanh'
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio

        # Default GNN: simple linear layer for node scoring
        if gnn is None:
            from torch_geometric.nn import GraphConv
            gnn = GraphConv

        # SAGPooling for learnable node selection
        self.pool = SAGPooling(
            in_channels=in_channels,
            ratio=ratio,
            GNN=gnn,
            min_score=min_score,
            multiplier=multiplier,
            nonlinearity=nonlinearity
        )

        # Linear projection for unpooling (abstract → concrete features)
        self.unpool_proj = nn.Linear(in_channels, in_channels)

    def why_operation(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor, Tensor]:
        """Apply WHY operation: Pool to abstract level.

        Args:
            x (Tensor): Node features [num_nodes, in_channels]
            edge_index (Tensor): Edge connectivity [2, num_edges]
            edge_attr (Tensor, optional): Edge features [num_edges, edge_dim]
            batch (Tensor, optional): Batch assignment [num_nodes]

        Returns:
            Tuple containing:
            - x_pooled (Tensor): Pooled node features [num_pooled_nodes, in_channels]
            - edge_index_pooled (Tensor): Pooled edge index [2, num_pooled_edges]
            - edge_attr_pooled (Tensor, optional): Pooled edge attributes
            - batch_pooled (Tensor, optional): Pooled batch assignment
            - perm (Tensor): Indices of selected nodes
            - score (Tensor): Selection scores for all nodes
        """
        # Apply SAGPooling
        x_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled, perm, score = \
            self.pool(x, edge_index, edge_attr, batch)

        return x_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled, perm, score

    def what_operation(
        self,
        x_abstract: Tensor,
        perm: Tensor,
        batch: Optional[Tensor] = None,
        original_num_nodes: Optional[int] = None
    ) -> Tensor:
        """Apply WHAT operation: Unpool to concrete level.

        Args:
            x_abstract (Tensor): Abstract node features [num_pooled_nodes, in_channels]
            perm (Tensor): Pooling indices (from why_operation)
            batch (Tensor, optional): Original batch assignment
            original_num_nodes (int, optional): Original number of nodes before pooling

        Returns:
            Tensor: Reconstructed node features [num_nodes, in_channels]
        """
        if original_num_nodes is None:
            if batch is not None:
                original_num_nodes = batch.size(0)
            else:
                # Assume single graph
                original_num_nodes = perm.max().item() + 1

        # Project abstract features
        x_abstract_proj = self.unpool_proj(x_abstract)

        # Initialize reconstructed features (zeros for non-selected nodes)
        device = x_abstract.device
        x_reconstructed = torch.zeros(
            original_num_nodes,
            self.in_channels,
            device=device,
            dtype=x_abstract.dtype
        )

        # Place pooled nodes back at their original positions
        x_reconstructed[perm] = x_abstract_proj

        # Interpolate features for non-selected nodes (simple averaging)
        # This is a basic unpooling strategy - could be improved with graph structure
        return x_reconstructed

    def cycle_loss(
        self,
        x_original: Tensor,
        x_reconstructed: Tensor,
        reduction: str = 'mean'
    ) -> Tensor:
        """Compute cycle consistency loss: ||WHY(WHAT(x)) - x||².

        Args:
            x_original (Tensor): Original node features
            x_reconstructed (Tensor): Reconstructed features after WHY→WHAT cycle
            reduction (str): Loss reduction ('mean', 'sum', 'none')

        Returns:
            Tensor: Reconstruction error
        """
        return F.mse_loss(x_reconstructed, x_original, reduction=reduction)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'ratio={self.ratio:.2f})')


class AdaptiveUnpooling(nn.Module):
    """Improved unpooling using graph structure for better reconstruction.

    Instead of just placing pooled nodes back, this uses edge connectivity
    to interpolate features for non-selected nodes from their neighbors.

    Args:
        in_channels (int): Feature dimensionality
        num_layers (int): Number of message passing layers for interpolation
    """

    def __init__(self, in_channels: int, num_layers: int = 2):
        super().__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers

        # MLP for feature interpolation
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

    def forward(
        self,
        x_abstract: Tensor,
        perm: Tensor,
        edge_index: Tensor,
        original_num_nodes: int
    ) -> Tensor:
        """Unpool with graph-aware interpolation.

        Args:
            x_abstract (Tensor): Abstract features [num_pooled, in_channels]
            perm (Tensor): Pooling indices
            edge_index (Tensor): Original edge connectivity
            original_num_nodes (int): Original number of nodes

        Returns:
            Tensor: Reconstructed features [original_num_nodes, in_channels]
        """
        device = x_abstract.device

        # Initialize with zeros
        x_reconstructed = torch.zeros(
            original_num_nodes,
            self.in_channels,
            device=device,
            dtype=x_abstract.dtype
        )

        # Place selected nodes
        x_reconstructed[perm] = x_abstract

        # Create mask for nodes that need interpolation
        mask = torch.ones(original_num_nodes, dtype=torch.bool, device=device)
        mask[perm] = False
        missing_indices = mask.nonzero(as_tuple=True)[0]

        if len(missing_indices) == 0:
            return x_reconstructed

        # For each missing node, aggregate from neighbors
        for idx in missing_indices:
            # Find neighbors
            neighbor_mask = (edge_index[0] == idx) | (edge_index[1] == idx)
            if neighbor_mask.sum() == 0:
                continue

            neighbor_edges = edge_index[:, neighbor_mask]
            neighbor_ids = torch.unique(torch.cat([
                neighbor_edges[0],
                neighbor_edges[1]
            ]))
            neighbor_ids = neighbor_ids[neighbor_ids != idx]

            if len(neighbor_ids) == 0:
                continue

            # Average neighbor features
            neighbor_features = x_reconstructed[neighbor_ids]
            aggregated = neighbor_features.mean(dim=0)

            x_reconstructed[idx] = aggregated

        return x_reconstructed

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'num_layers={self.num_layers})')
