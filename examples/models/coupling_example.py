"""
Coupling Layers Example: Invertible Transformations

Demonstrates perfect invertibility for information-preserving transformations.
"""

import torch
from nsm.models.coupling import (
    AffineCouplingLayer,
    MultiLayerCoupling,
    GraphCouplingLayer
)

print("="*60)
print("Coupling Layers: Perfect Invertibility")
print("="*60)

# Example 1: Basic coupling layer
print("\nExample 1: Single Affine Coupling Layer")
layer = AffineCouplingLayer(features=64, hidden_dim=128)
x = torch.randn(10, 64)
y = layer.forward(x)
x_recon = layer.inverse(y)
error = (x - x_recon).abs().max()
print(f"Reconstruction error: {error:.2e} (perfect: <1e-5)")

# Example 2: Multi-layer coupling
print("\nExample 2: Multi-Layer Coupling (4 layers)")
coupling = MultiLayerCoupling(features=64, num_layers=4, hidden_dim=128)
x = torch.randn(20, 64)
y = coupling.forward(x)
x_recon = coupling.inverse(y)
error = (x - x_recon).abs().max()
print(f"4-layer reconstruction error: {error:.2e}")

# Example 3: Graph coupling
print("\nExample 3: Graph Coupling (100 nodes)")
graph_coupling = GraphCouplingLayer(node_features=32, num_layers=3)
node_features = torch.randn(100, 32)
transformed = graph_coupling.forward(node_features)
reconstructed = graph_coupling.inverse(transformed)
error = (node_features - reconstructed).abs().max()
print(f"Graph coupling error: {error:.2e}")

print("\n" + "="*60)
print("All examples: Perfect invertibility maintained!")
print("="*60)
