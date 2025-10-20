"""
Relational Graph Convolutional Network (R-GCN) with Confidence Weighting.

Implements R-GCN message passing with:
- Basis decomposition for parameter efficiency (70% reduction)
- Confidence-weighted messages for uncertainty-aware propagation
- Multi-layer stacking with residual connections

References:
    Schlichtkrull et al. (2018) - "Modeling Relational Data with Graph
    Convolutional Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional


class ConfidenceWeightedRGCN(MessagePassing):
    """R-GCN layer with basis decomposition and confidence weighting.

    Key properties:
    - Handles multiple relation types via shared basis matrices
    - Parameter reduction: num_bases × d_in × d_out + num_relations × num_bases
      vs. num_relations × d_in × d_out (70% reduction typical)
    - Confidence values from edge attributes weight message contributions
    - Self-loops with identity relation for node self-information

    Mathematical formulation:
        h_i' = σ(Σ_{r,j∈N_i^r} (1/c_i^r) * W_r * h_j * conf_ij + W_0 * h_i)

        where:
        - W_r = Σ_b a_{rb} * V_b (basis decomposition)
        - c_i^r = |{j : (j,r,i) ∈ E}| (normalization constant)
        - conf_ij ∈ [0,1] (edge confidence/causal strength)
        - W_0 = self-loop weights

    Args:
        in_channels (int): Input feature dimensionality
        out_channels (int): Output feature dimensionality
        num_relations (int): Number of relation types (predicates)
        num_bases (int, optional): Number of basis matrices. Defaults to 30.
            Recommended: min(num_relations, 30-50) for large relation sets
        aggr (str, optional): Aggregation method. Defaults to 'mean'.
        bias (bool, optional): Whether to add bias. Defaults to True.

    Example:
        >>> # Planning domain: 16 relations
        >>> rgcn = ConfidenceWeightedRGCN(64, 64, num_relations=16, num_bases=8)
        >>>
        >>> # Knowledge graph: 66 relations (high diversity)
        >>> rgcn_kg = ConfidenceWeightedRGCN(64, 64, num_relations=66, num_bases=12)
        >>>
        >>> # Forward pass
        >>> x = torch.randn(100, 64)  # 100 nodes, 64 features
        >>> edge_index = torch.randint(0, 100, (2, 500))  # 500 edges
        >>> edge_type = torch.randint(0, 16, (500,))  # Relation types
        >>> edge_attr = torch.rand(500)  # Confidence values [0,1]
        >>>
        >>> out = rgcn(x, edge_index, edge_type, edge_attr)  # [100, 64]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: int = 30,
        aggr: str = 'mean',
        bias: bool = True,
        **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)  # Can't have more bases than relations

        # Basis matrices: [num_bases, in_channels, out_channels]
        self.basis = nn.Parameter(
            torch.Tensor(self.num_bases, in_channels, out_channels)
        )

        # Relation-specific basis coefficients: [num_relations, num_bases]
        # Each relation is a linear combination of bases
        self.att = nn.Parameter(
            torch.Tensor(num_relations, self.num_bases)
        )

        # Self-loop transformation (root node update)
        self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Glorot uniform initialization."""
        # Xavier/Glorot uniform for basis matrices
        nn.init.xavier_uniform_(self.basis)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.root)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with confidence-weighted message passing.

        Args:
            x (Tensor): Node features [num_nodes, in_channels]
            edge_index (LongTensor): Graph connectivity [2, num_edges]
            edge_type (LongTensor): Edge relation types [num_edges]
            edge_attr (Tensor, optional): Edge confidence values [num_edges].
                If None, all confidences default to 1.0

        Returns:
            Tensor: Updated node features [num_nodes, out_channels]
        """
        # Default confidence to 1.0 if not provided
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), device=x.device)

        # Ensure edge_attr is 1D
        if edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze()

        # Compute relation-specific weight matrices via basis decomposition
        # Shape: [num_relations, num_bases] @ [num_bases, in_channels, out_channels]
        #     -> [num_relations, in_channels, out_channels]
        weight = torch.einsum('rb,bio->rio', self.att, self.basis)

        # Add self-loops for self-information retention
        # Each node receives message from itself with identity relation
        num_nodes = x.size(0)
        edge_index, edge_type = self._add_self_loops(
            edge_index, edge_type, num_nodes
        )

        # Extend edge_attr with confidence 1.0 for self-loops
        self_loop_attr = torch.ones(num_nodes, device=x.device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        # Propagate messages
        out = self.propagate(
            edge_index,
            x=x,
            edge_type=edge_type,
            edge_attr=edge_attr,
            weight=weight
        )

        # Root node transformation (self-loop)
        out = out + x @ self.root

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(
        self,
        x_j: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: torch.Tensor,
        weight: torch.Tensor
    ) -> torch.Tensor:
        """Compute messages from neighbors with confidence weighting.

        Args:
            x_j (Tensor): Source node features [num_edges, in_channels]
            edge_type (LongTensor): Relation types [num_edges]
            edge_attr (Tensor): Confidence values [num_edges]
            weight (Tensor): Relation weights [num_relations, in_channels, out_channels]

        Returns:
            Tensor: Messages [num_edges, out_channels]
        """
        # Select weight matrix for each edge based on its type
        # weight[edge_type] -> [num_edges, in_channels, out_channels]
        w = weight[edge_type]

        # Apply relation-specific transformation: x_j @ W_r
        # [num_edges, in_channels] @ [num_edges, in_channels, out_channels]
        #   -> [num_edges, out_channels]
        msg = torch.einsum('ei,eio->eo', x_j, w)

        # Weight messages by confidence
        # [num_edges, out_channels] * [num_edges, 1]
        msg = msg * edge_attr.unsqueeze(-1)

        return msg

    def _add_self_loops(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_nodes: int
    ):
        """Add self-loops with identity relation type.

        Self-loops use the last relation index as a special "self" relation.

        Args:
            edge_index (LongTensor): Graph connectivity [2, num_edges]
            edge_type (LongTensor): Relation types [num_edges]
            num_nodes (int): Number of nodes

        Returns:
            Tuple[LongTensor, LongTensor]: Updated edge_index and edge_type
        """
        # Create self-loop edges: (i, i) for all nodes
        loop_index = torch.arange(num_nodes, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)

        # Self-loops use the last relation type as "self" relation
        loop_type = torch.full(
            (num_nodes,),
            self.num_relations - 1,
            dtype=edge_type.dtype,
            device=edge_type.device
        )

        # Concatenate with existing edges
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_type = torch.cat([edge_type, loop_type], dim=0)

        return edge_index, edge_type

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations}, '
                f'num_bases={self.num_bases})')


class ConfidenceEstimator(nn.Module):
    """Learns to estimate edge confidence from node features.

    Computes confidence scores for edges based on source and target node
    embeddings. Useful when confidence is not provided as edge attributes.

    Architecture:
        [x_i || x_j || one_hot(edge_type)] -> MLP -> sigmoid -> confidence

    Args:
        node_dim (int): Node feature dimensionality
        num_relations (int): Number of relation types
        hidden_dim (int, optional): Hidden layer size. Defaults to 64.

    Example:
        >>> estimator = ConfidenceEstimator(64, num_relations=16)
        >>> x = torch.randn(100, 64)  # Node features
        >>> edge_index = torch.randint(0, 100, (2, 500))
        >>> edge_type = torch.randint(0, 16, (500,))
        >>>
        >>> # Estimate confidence for each edge
        >>> confidence = estimator(x, edge_index, edge_type)  # [500]
    """

    def __init__(
        self,
        node_dim: int,
        num_relations: int,
        hidden_dim: int = 64
    ):
        super().__init__()

        self.node_dim = node_dim
        self.num_relations = num_relations

        # MLP: [2*node_dim + num_relations] -> hidden -> 1
        input_dim = 2 * node_dim + num_relations

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """Estimate confidence for edges.

        Args:
            x (Tensor): Node features [num_nodes, node_dim]
            edge_index (LongTensor): Graph connectivity [2, num_edges]
            edge_type (LongTensor): Relation types [num_edges]

        Returns:
            Tensor: Estimated confidence values [num_edges]
        """
        # Extract source and target node features
        x_i = x[edge_index[0]]  # [num_edges, node_dim]
        x_j = x[edge_index[1]]  # [num_edges, node_dim]

        # One-hot encode edge types
        edge_type_onehot = F.one_hot(
            edge_type, num_classes=self.num_relations
        ).float()  # [num_edges, num_relations]

        # Concatenate features
        features = torch.cat([x_i, x_j, edge_type_onehot], dim=-1)

        # Predict confidence
        confidence = self.mlp(features).squeeze(-1)

        return confidence


class HierarchicalRGCN(nn.Module):
    """Multi-layer R-GCN with residual connections and layer normalization.

    Stacks multiple R-GCN layers with:
    - Residual connections to prevent gradient vanishing
    - Layer normalization for training stability
    - Optional dropout for regularization

    Architecture:
        for each layer:
            h' = LayerNorm(h + RGCNLayer(h))
            h' = Dropout(h')

    Args:
        in_channels (int): Input feature dimensionality
        hidden_channels (int): Hidden layer dimensionality
        out_channels (int): Output feature dimensionality
        num_relations (int): Number of relation types
        num_layers (int, optional): Number of R-GCN layers. Defaults to 2.
        num_bases (int, optional): Basis matrices per layer. Defaults to 30.
        dropout (float, optional): Dropout probability. Defaults to 0.1.

    Example:
        >>> # 2-layer R-GCN for planning domain
        >>> model = HierarchicalRGCN(
        ...     in_channels=64,
        ...     hidden_channels=128,
        ...     out_channels=64,
        ...     num_relations=16,
        ...     num_layers=2,
        ...     num_bases=8
        ... )
        >>>
        >>> x = torch.randn(100, 64)
        >>> edge_index = torch.randint(0, 100, (2, 500))
        >>> edge_type = torch.randint(0, 16, (500,))
        >>> edge_attr = torch.rand(500)
        >>>
        >>> out = model(x, edge_index, edge_type, edge_attr)  # [100, 64]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_relations: int,
        num_layers: int = 2,
        num_bases: int = 30,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        # Input projection if needed
        if in_channels != hidden_channels:
            self.input_proj = nn.Linear(in_channels, hidden_channels)
        else:
            self.input_proj = nn.Identity()

        # R-GCN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                ConfidenceWeightedRGCN(
                    hidden_channels,
                    hidden_channels,
                    num_relations,
                    num_bases
                )
            )

        # Layer normalization for each layer
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Output projection if needed
        if hidden_channels != out_channels:
            self.output_proj = nn.Linear(hidden_channels, out_channels)
        else:
            self.output_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through stacked R-GCN layers.

        Args:
            x (Tensor): Node features [num_nodes, in_channels]
            edge_index (LongTensor): Graph connectivity [2, num_edges]
            edge_type (LongTensor): Relation types [num_edges]
            edge_attr (Tensor, optional): Edge confidence [num_edges]

        Returns:
            Tensor: Output features [num_nodes, out_channels]
        """
        # Input projection
        x = self.input_proj(x)

        # Stack R-GCN layers with residuals
        for layer, norm in zip(self.layers, self.norms):
            # R-GCN layer
            x_out = layer(x, edge_index, edge_type, edge_attr)

            # Residual connection + LayerNorm
            x = norm(x + x_out)

            # Activation + Dropout
            x = F.relu(x)
            x = self.dropout(x)

        # Output projection
        x = self.output_proj(x)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'num_layers={self.num_layers})')
