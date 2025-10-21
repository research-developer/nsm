"""
Graph Construction Utilities

Converts semantic triples to PyTorch Geometric Data objects
for graph neural network processing.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch_geometric.data import Data

from .triple import SemanticTriple, TripleCollection
from .vocabulary import TripleVocabulary


class GraphConstructor:
    """
    Constructs PyTorch Geometric graphs from semantic triples.

    Handles conversion of triples to graph representation with:
    - Nodes: Unique entities from subjects and objects
    - Edges: Directed edges from subject to object
    - Edge attributes: Predicate types and confidence scores

    Attributes:
        vocabulary: TripleVocabulary for entity/predicate mappings
        node_feature_dim: Dimensionality of node embeddings
        use_confidence_as_edge_weight: Whether to use confidence as edge weights

    Examples:
        >>> triples = [
        ...     SemanticTriple("robot", "executes", "move", confidence=0.9, level=1),
        ...     SemanticTriple("move", "achieves", "goal", confidence=0.8, level=2)
        ... ]
        >>> constructor = GraphConstructor()
        >>> graph = constructor.construct(triples)
        >>> print(graph)
        Data(x=[3, 64], edge_index=[2, 2], edge_attr=[2, 4], ...)
    """

    def __init__(
        self,
        vocabulary: Optional[TripleVocabulary] = None,
        node_feature_dim: int = 64,
        use_confidence_as_edge_weight: bool = True
    ):
        """
        Initialize graph constructor.

        Args:
            vocabulary: Existing vocabulary (if None, creates new one)
            node_feature_dim: Dimension for node feature embeddings
            use_confidence_as_edge_weight: Use confidence scores as edge weights
        """
        self.vocabulary = vocabulary or TripleVocabulary()
        self.node_feature_dim = node_feature_dim
        self.use_confidence_as_edge_weight = use_confidence_as_edge_weight

    def construct(
        self,
        triples: List[SemanticTriple],
        node_features: Optional[Tensor] = None
    ) -> Data:
        """
        Construct PyG Data object from list of triples.

        Args:
            triples: List of semantic triples
            node_features: Optional pre-computed node features [num_nodes, feat_dim]
                          If None, uses random embeddings

        Returns:
            PyG Data object with:
                - x: Node features [num_nodes, feat_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge attributes including:
                    - confidence: [num_edges, 4] log-score tensors
                    - edge_type: [num_edges] (predicate indices)
                - num_nodes: Total number of unique entities
                - num_edges: Total number of triples

        Mathematical Foundation:
            Graph G = (V, E) where:
            - V = unique entities from all triples
            - E = {(subject, object) for each triple}
            - Edge features include predicate type and confidence
        """
        if not triples:
            # Return empty graph
            return Data(
                x=torch.zeros(0, self.node_feature_dim),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, 4)
            )

        # Build vocabulary from triples
        node_indices = {}  # entity -> node_idx mapping
        edge_list = []  # List of (src, dst) tuples
        edge_types = []  # List of predicate indices
        confidence_vectors = []  # List of [4] confidence tensors
        node_levels = {}  # entity -> level mapping

        current_node_idx = 0

        for triple in triples:
            # Add subject to nodes
            if triple.subject not in node_indices:
                subj_idx = self.vocabulary.entity_vocab.add_token(triple.subject)
                node_indices[triple.subject] = current_node_idx
                node_levels[triple.subject] = triple.level
                current_node_idx += 1
            else:
                subj_idx = node_indices[triple.subject]

            # Add object to nodes
            if triple.object not in node_indices:
                obj_idx = self.vocabulary.entity_vocab.add_token(triple.object)
                node_indices[triple.object] = current_node_idx
                node_levels[triple.object] = triple.level
                current_node_idx += 1
            else:
                obj_idx = node_indices[triple.object]

            # Add predicate to vocabulary
            pred_idx = self.vocabulary.predicate_vocab.add_token(triple.predicate)

            # Create edge from subject to object
            edge_list.append((node_indices[triple.subject], node_indices[triple.object]))
            edge_types.append(pred_idx)
            confidence_vectors.append(triple.get_confidence_tensor())

        # Convert to tensors
        num_nodes = len(node_indices)

        # Edge index in COO format [2, num_edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Edge types for R-GCN
        edge_type = torch.tensor(edge_types, dtype=torch.long)

        # Confidence score tensors [num_edges, 4]
        confidence = torch.stack(confidence_vectors, dim=0)

        # Node features
        if node_features is None:
            # Initialize with random embeddings (will be learned)
            x = torch.randn(num_nodes, self.node_feature_dim)
        else:
            x = node_features

        # Node level information (for hierarchical operations)
        node_level_tensor = torch.tensor(
            [node_levels[entity] for entity in sorted(node_indices, key=node_indices.get)],
            dtype=torch.long
        )

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=confidence,
            edge_type=edge_type,
            node_level=node_level_tensor,
            num_nodes=num_nodes
        )

        return data

    def construct_from_collection(
        self,
        collection: TripleCollection,
        node_features: Optional[Tensor] = None
    ) -> Data:
        """
        Construct graph from TripleCollection.

        Args:
            collection: TripleCollection object
            node_features: Optional pre-computed node features

        Returns:
            PyG Data object
        """
        return self.construct(collection.triples, node_features)

    def construct_hierarchical(
        self,
        triples: List[SemanticTriple]
    ) -> Tuple[Data, Data]:
        """
        Construct separate graphs for concrete (L1) and abstract (L2) levels.

        Args:
            triples: List of semantic triples at multiple levels

        Returns:
            Tuple of (concrete_graph, abstract_graph)
                - concrete_graph: Graph containing only level 1 triples
                - abstract_graph: Graph containing only level 2+ triples

        Note:
            This is useful for Phase 1 implementation where we explicitly
            separate concrete and abstract reasoning.
        """
        collection = TripleCollection(triples)

        # Separate by level
        concrete_collection = collection.filter_by_level(1)
        abstract_collection = TripleCollection([
            t for t in triples if t.level >= 2
        ])

        # Construct separate graphs
        concrete_graph = self.construct_from_collection(concrete_collection)
        abstract_graph = self.construct_from_collection(abstract_collection)

        return concrete_graph, abstract_graph

    def add_self_loops(
        self,
        data: Data,
        self_loop_weight: Union[float, Sequence[float], Tensor] = 1.0
    ) -> Data:
        """
        Add self-loops to graph (useful for GNN message passing).

        Args:
            data: PyG Data object
            self_loop_weight: Confidence score for self-loops. Accepts
                scalar or length-4 iterable/tensor.

        Returns:
            Data object with self-loops added
        """
        num_nodes = data.num_nodes

        # Create self-loop edges
        self_loop_index = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)

        # Concatenate with existing edges
        data.edge_index = torch.cat([data.edge_index, self_loop_index], dim=1)

        # Add self-loop attributes
        self_loop_attr = self._expand_self_loop_weight(num_nodes, self_loop_weight)
        data.edge_attr = torch.cat([data.edge_attr, self_loop_attr], dim=0)

        # Add self-loop edge types (use special index for self-loops)
        # Assuming self-loop gets the next predicate index
        self_loop_type_idx = self.vocabulary.predicate_vocab.add_token('<SELF>')
        self_loop_types = torch.full(
            (num_nodes,),
            self_loop_type_idx,
            dtype=torch.long
        )
        data.edge_type = torch.cat([data.edge_type, self_loop_types], dim=0)

        return data

    @staticmethod
    def _expand_self_loop_weight(
        num_nodes: int,
        weight: Union[float, Sequence[float], Tensor]
    ) -> Tensor:
        """Expand self-loop weight(s) to [num_nodes, 4] tensor."""
        if isinstance(weight, Tensor):
            tensor = weight.detach().clone().to(dtype=torch.float32)
            flat = tensor.reshape(-1)
            if flat.numel() == 1:
                value = float(flat.item())
                return torch.full((num_nodes, 4), value, dtype=torch.float32)
            if flat.numel() != 4:
                raise ValueError(
                    "Self-loop weight tensor must have 1 or 4 elements"
                )
            return flat.reshape(1, 4).repeat(num_nodes, 1)

        if isinstance(weight, Sequence):
            values = list(weight)
            tensor = torch.tensor(values, dtype=torch.float32)
            flat = tensor.reshape(-1)
            if flat.numel() == 1:
                return GraphConstructor._expand_self_loop_weight(num_nodes, flat.item())
            if flat.numel() != 4:
                raise ValueError(
                    "Self-loop weight sequence must have length 1 or 4"
                )
            return flat.reshape(1, 4).repeat(num_nodes, 1)

        value = float(weight)
        return torch.full((num_nodes, 4), value, dtype=torch.float32)

    def batch_construct(
        self,
        triple_lists: List[List[SemanticTriple]]
    ) -> List[Data]:
        """
        Construct multiple graphs from lists of triples.

        Args:
            triple_lists: List of triple lists (one per graph)

        Returns:
            List of PyG Data objects

        Note:
            All graphs share the same vocabulary for consistency.
        """
        return [self.construct(triples) for triples in triple_lists]

    def get_entity_mapping(self) -> Dict[int, str]:
        """
        Get mapping from node indices to entity names.

        Returns:
            Dictionary mapping node_idx -> entity_name
        """
        return self.vocabulary.entity_vocab.idx_to_token.copy()

    def get_predicate_mapping(self) -> Dict[int, str]:
        """
        Get mapping from edge type indices to predicate names.

        Returns:
            Dictionary mapping edge_type_idx -> predicate_name
        """
        return self.vocabulary.predicate_vocab.idx_to_token.copy()


def visualize_graph_structure(data: Data) -> str:
    """
    Create a text representation of graph structure.

    Args:
        data: PyG Data object

    Returns:
        String representation of graph structure

    Examples:
        >>> graph = constructor.construct(triples)
        >>> print(visualize_graph_structure(graph))
        Graph Structure:
          Nodes: 5
          Edges: 8
          Node features: [5, 64]
          Edge attributes: [8, 4]
    """
    info = [
        "Graph Structure:",
        f"  Nodes: {data.num_nodes}",
        f"  Edges: {data.edge_index.size(1)}",
        f"  Node features: {list(data.x.shape)}",
    ]

    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        info.append(f"  Edge attributes: {list(data.edge_attr.shape)}")

    if hasattr(data, 'edge_type') and data.edge_type is not None:
        num_edge_types = data.edge_type.max().item() + 1
        info.append(f"  Edge types: {num_edge_types}")

    if hasattr(data, 'node_level') and data.node_level is not None:
        levels = torch.unique(data.node_level).tolist()
        info.append(f"  Hierarchy levels: {levels}")

    return "\n".join(info)
