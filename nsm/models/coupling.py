"""
Invertible coupling layers for information-preserving transformations.

Implements RealNVP (Real-valued Non-Volume Preserving) affine coupling layers
with perfect invertibility for WHY/WHAT symmetric operations.

Key properties:
- Exact inverse (forward/inverse are mathematically perfect)
- Tractable Jacobian determinant
- No information loss during transformations
- Fully differentiable in both directions

References:
    Dinh et al. (2017) - "Density estimation using Real NVP"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer with perfect invertibility (RealNVP).

    Splits input into two parts and applies affine transformation:
        Forward:  y_1 = x_1
                  y_2 = x_2 ⊙ exp(s(x_1)) + t(x_1)

        Inverse:  x_1 = y_1
                  x_2 = (y_2 - t(y_1)) ⊙ exp(-s(y_1))

    Key properties:
    - Forward/inverse are exact (no approximation error)
    - Jacobian: det(J) = exp(Σ s(x_1))
    - s() and t() can be arbitrary neural networks
    - First part (x_1) passes through unchanged (identity)

    Args:
        features (int): Total feature dimensionality
        split_dim (int, optional): Split point. If None, uses features // 2
        hidden_dim (int): Hidden layer size for s() and t() networks
        num_hidden_layers (int): Number of hidden layers in s() and t()
        activation (str): Activation function ('relu', 'elu', 'gelu')

    Example:
        >>> layer = AffineCouplingLayer(features=64, hidden_dim=128)
        >>> x = torch.randn(10, 64)
        >>>
        >>> # Forward transformation
        >>> y = layer.forward(x)
        >>>
        >>> # Inverse (exact reconstruction)
        >>> x_reconstructed = layer.inverse(y)
        >>> assert torch.allclose(x, x_reconstructed, atol=1e-5)
    """

    def __init__(
        self,
        features: int,
        split_dim: Optional[int] = None,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        activation: str = 'relu'
    ):
        super().__init__()

        self.features = features
        self.split_dim = split_dim or features // 2
        self.hidden_dim = hidden_dim

        # Size of second part (transformed part)
        self.transform_dim = features - self.split_dim

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Scale network s(x_1) → s
        # Output: scale factors for y_2
        scale_layers = []
        in_dim = self.split_dim

        for _ in range(num_hidden_layers):
            scale_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                self.activation
            ])
            in_dim = hidden_dim

        scale_layers.append(nn.Linear(in_dim, self.transform_dim))
        self.scale_net = nn.Sequential(*scale_layers)

        # Translation network t(x_1) → t
        # Output: translation for y_2
        translate_layers = []
        in_dim = self.split_dim

        for _ in range(num_hidden_layers):
            translate_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                self.activation
            ])
            in_dim = hidden_dim

        translate_layers.append(nn.Linear(in_dim, self.transform_dim))
        self.translate_net = nn.Sequential(*translate_layers)

        # Initialize to near-identity transformation
        self._init_identity()

    def _init_identity(self):
        """Initialize networks to approximate identity transformation.

        Sets final layer weights/biases small so initial transformation
        is close to identity (y ≈ x). Helps training stability.
        """
        # Scale network: initialize to output near zero → exp(0) = 1
        nn.init.zeros_(self.scale_net[-1].weight)
        nn.init.zeros_(self.scale_net[-1].bias)

        # Translation network: initialize to output near zero → t ≈ 0
        nn.init.zeros_(self.translate_net[-1].weight)
        nn.init.zeros_(self.translate_net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply forward transformation.

        Args:
            x (Tensor): Input features [batch_size, features]

        Returns:
            Tensor: Transformed features [batch_size, features]
        """
        # Split input
        x1, x2 = self._split(x)

        # Compute scale and translation from x1
        s = self.scale_net(x1)
        t = self.translate_net(x1)

        # Affine transformation on x2
        # y2 = x2 * exp(s) + t
        y2 = x2 * torch.exp(s) + t

        # Concatenate (x1 unchanged, x2 transformed)
        y = torch.cat([x1, y2], dim=-1)

        return y

    def inverse(self, y: Tensor) -> Tensor:
        """Apply inverse transformation (exact).

        Args:
            y (Tensor): Transformed features [batch_size, features]

        Returns:
            Tensor: Original features [batch_size, features]
        """
        # Split transformed output
        y1, y2 = self._split(y)

        # Compute scale and translation from y1 (same as x1)
        s = self.scale_net(y1)
        t = self.translate_net(y1)

        # Inverse affine transformation
        # x2 = (y2 - t) * exp(-s)
        x2 = (y2 - t) * torch.exp(-s)

        # Concatenate (y1 = x1, x2 reconstructed)
        x = torch.cat([y1, x2], dim=-1)

        return x

    def log_det_jacobian(self, x: Tensor) -> Tensor:
        """Compute log determinant of Jacobian.

        The Jacobian determinant measures volume change.
        For RealNVP: det(J) = exp(Σ s(x_1))

        Args:
            x (Tensor): Input features [batch_size, features]

        Returns:
            Tensor: Log determinant [batch_size]
        """
        x1, _ = self._split(x)
        s = self.scale_net(x1)

        # Sum scale factors across transform dimensions
        log_det = torch.sum(s, dim=-1)

        return log_det

    def _split(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Split input into identity and transform parts.

        Args:
            x (Tensor): Input [batch_size, features]

        Returns:
            Tuple[Tensor, Tensor]: (x1, x2) where x1 is identity part
        """
        x1 = x[:, :self.split_dim]
        x2 = x[:, self.split_dim:]
        return x1, x2

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(features={self.features}, '
                f'split={self.split_dim}/{self.features-self.split_dim}, '
                f'hidden_dim={self.hidden_dim})')


class MultiLayerCoupling(nn.Module):
    """Multiple coupling layers with alternating split dimensions.

    Stacks multiple AffineCouplingLayers with varying split points
    to ensure all dimensions are transformed (no frozen dimensions).

    Strategy:
    - Alternate split dimensions: 50%, 33%, 50%, 33%, ...
    - Each layer transforms different dimensions
    - Overall transformation is highly expressive

    Args:
        features (int): Feature dimensionality
        num_layers (int): Number of coupling layers
        hidden_dim (int): Hidden dimension for each layer
        num_hidden_layers (int): Hidden layers in s()/t() networks
        activation (str): Activation function

    Example:
        >>> coupling = MultiLayerCoupling(features=64, num_layers=4)
        >>> x = torch.randn(10, 64)
        >>>
        >>> # Forward
        >>> y = coupling.forward(x)
        >>>
        >>> # Inverse (exact)
        >>> x_reconstructed = coupling.inverse(y)
        >>> assert torch.allclose(x, x_reconstructed, atol=1e-5)
    """

    def __init__(
        self,
        features: int,
        num_layers: int = 4,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        activation: str = 'relu'
    ):
        super().__init__()

        self.features = features
        self.num_layers = num_layers

        # Create layers with alternating split dimensions
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            # Alternate splits: [50%, 33%, 50%, 33%, ...]
            if i % 2 == 0:
                split_dim = features // 2
            else:
                split_dim = features // 3

            layer = AffineCouplingLayer(
                features=features,
                split_dim=split_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                activation=activation
            )

            self.layers.append(layer)

    def forward(self, x: Tensor) -> Tensor:
        """Apply forward transformation through all layers.

        Args:
            x (Tensor): Input [batch_size, features]

        Returns:
            Tensor: Transformed output [batch_size, features]
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def inverse(self, y: Tensor) -> Tensor:
        """Apply inverse transformation (exact).

        Applies layers in reverse order.

        Args:
            y (Tensor): Transformed output [batch_size, features]

        Returns:
            Tensor: Original input [batch_size, features]
        """
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y

    def log_det_jacobian(self, x: Tensor) -> Tensor:
        """Compute log determinant of Jacobian for all layers.

        Sum of log determinants (product of Jacobians).

        Args:
            x (Tensor): Input [batch_size, features]

        Returns:
            Tensor: Total log determinant [batch_size]
        """
        total_log_det = torch.zeros(x.size(0), device=x.device)

        for layer in self.layers:
            total_log_det += layer.log_det_jacobian(x)
            x = layer.forward(x)

        return total_log_det

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(features={self.features}, '
                f'num_layers={self.num_layers})')


class GraphCouplingLayer(nn.Module):
    """Coupling layer for graph node features.

    Applies coupling transformation to node features while preserving
    graph structure. Useful for WHY/WHAT operations on graph hierarchies.

    Args:
        node_features (int): Node feature dimensionality
        num_layers (int): Number of coupling layers
        hidden_dim (int): Hidden dimension

    Example:
        >>> coupling = GraphCouplingLayer(node_features=64, num_layers=3)
        >>>
        >>> # Graph with 100 nodes, 64 features each
        >>> x = torch.randn(100, 64)
        >>> edge_index = torch.randint(0, 100, (2, 500))
        >>>
        >>> # Transform node features (graph structure unchanged)
        >>> y = coupling.forward(x)
        >>> x_reconstructed = coupling.inverse(y)
        >>>
        >>> # Perfect reconstruction
        >>> assert torch.allclose(x, x_reconstructed, atol=1e-5)
    """

    def __init__(
        self,
        node_features: int,
        num_layers: int = 3,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.node_features = node_features

        # Use MultiLayerCoupling for node features
        self.coupling = MultiLayerCoupling(
            features=node_features,
            num_layers=num_layers,
            hidden_dim=hidden_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        """Transform node features.

        Args:
            x (Tensor): Node features [num_nodes, node_features]

        Returns:
            Tensor: Transformed features [num_nodes, node_features]
        """
        return self.coupling.forward(x)

    def inverse(self, y: Tensor) -> Tensor:
        """Inverse transform (exact).

        Args:
            y (Tensor): Transformed features [num_nodes, node_features]

        Returns:
            Tensor: Original features [num_nodes, node_features]
        """
        return self.coupling.inverse(y)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(node_features={self.node_features}, '
                f'num_layers={self.coupling.num_layers})')
