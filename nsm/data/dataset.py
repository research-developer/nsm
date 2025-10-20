"""
Dataset Classes for Semantic Triples

Provides abstract base class and utilities for domain-specific
triple datasets (planning, knowledge graphs, causal reasoning).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .triple import SemanticTriple, TripleCollection
from .graph import GraphConstructor
from .vocabulary import TripleVocabulary


class BaseSemanticTripleDataset(Dataset, ABC):
    """
    Abstract base class for semantic triple datasets.

    Subclasses implement domain-specific triple generation or loading.
    This class provides common functionality for:
    - Converting triples to PyG graphs
    - Vocabulary management
    - Train/val/test splits
    - Caching

    Attributes:
        root: Root directory for dataset files
        split: 'train', 'val', or 'test'
        vocabulary: Shared vocabulary for entities and predicates
        graph_constructor: Converts triples to PyG graphs
        triples: List of semantic triples (populated by generate_triples)

    Examples:
        Subclass implementation:

        >>> class PlanningDataset(BaseSemanticTripleDataset):
        ...     def generate_triples(self) -> List[SemanticTriple]:
        ...         return [
        ...             SemanticTriple("robot", "executes", "move", level=1),
        ...             SemanticTriple("move", "achieves", "goal", level=2)
        ...         ]
        ...
        ...     def generate_labels(self, idx: int) -> torch.Tensor:
        ...         # Task-specific labels
        ...         return torch.tensor([1], dtype=torch.long)

        >>> dataset = PlanningDataset(root="data/planning", split="train")
        >>> graph, label = dataset[0]
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        vocabulary: Optional[TripleVocabulary] = None,
        node_feature_dim: int = 64,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
    ):
        """
        Initialize dataset.

        Args:
            root: Root directory for dataset files
            split: Dataset split ('train', 'val', 'test')
            vocabulary: Shared vocabulary (if None, creates new one)
            node_feature_dim: Dimensionality of node features
            transform: Optional transform applied to each graph
            pre_transform: Optional transform applied once during processing
        """
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.pre_transform = pre_transform
        self.node_feature_dim = node_feature_dim

        # Initialize vocabulary and graph constructor
        self.vocabulary = vocabulary or TripleVocabulary()
        self.graph_constructor = GraphConstructor(
            vocabulary=self.vocabulary,
            node_feature_dim=node_feature_dim
        )

        # Create dataset directories
        self.root.mkdir(parents=True, exist_ok=True)
        self.processed_dir = self.root / "processed"
        self.processed_dir.mkdir(exist_ok=True)

        # Generate or load triples
        self.triples: List[SemanticTriple] = []
        self._load_or_generate()

    def _load_or_generate(self):
        """Load cached triples or generate new ones."""
        cache_path = self.processed_dir / f"{self.split}_triples.pt"

        if cache_path.exists():
            # Load cached triples
            self._load_from_cache(cache_path)
        else:
            # Generate new triples
            self.triples = self.generate_triples()

            # Apply pre_transform if provided
            if self.pre_transform is not None:
                self.triples = self.pre_transform(self.triples)

            # Cache for future use
            self._save_to_cache(cache_path)

    def _save_to_cache(self, path: Path):
        """Save triples to cache file."""
        data = {
            'triples': self.triples,
            'vocabulary': self.vocabulary,
        }
        torch.save(data, path)

    def _load_from_cache(self, path: Path):
        """Load triples from cache file."""
        data = torch.load(path)
        self.triples = data['triples']
        if 'vocabulary' in data:
            self.vocabulary = data['vocabulary']
            self.graph_constructor.vocabulary = self.vocabulary

    @abstractmethod
    def generate_triples(self) -> List[SemanticTriple]:
        """
        Generate domain-specific semantic triples.

        Returns:
            List of SemanticTriple objects

        Note:
            This method must be implemented by subclasses.
            Examples:
            - Planning: Generate action sequences and goals
            - Knowledge graph: Load entity relationships
            - Causal: Generate causal chains with confounders
        """
        pass

    @abstractmethod
    def generate_labels(self, idx: int) -> torch.Tensor:
        """
        Generate task-specific labels for a triple or graph.

        Args:
            idx: Index of the data sample

        Returns:
            Tensor containing labels (format depends on task)

        Note:
            Examples:
            - Classification: torch.tensor([class_idx], dtype=torch.long)
            - Link prediction: torch.tensor([0 or 1], dtype=torch.float)
            - Regression: torch.tensor([value], dtype=torch.float)
        """
        pass

    def __len__(self) -> int:
        """Number of samples in dataset."""
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (graph_data, label)
                - graph_data: PyG Data object
                - label: Task-specific label tensor
        """
        # Get triple(s) for this sample
        # For now, each triple is a sample; subclasses can override
        # to group multiple triples into one graph
        triple = self.triples[idx]

        # Convert to graph
        graph = self.graph_constructor.construct([triple])

        # Apply transform if provided
        if self.transform is not None:
            graph = self.transform(graph)

        # Get labels
        label = self.generate_labels(idx)

        return graph, label

    def get_graph_for_triples(
        self,
        triple_indices: List[int]
    ) -> Data:
        """
        Construct graph from multiple triples by indices.

        Args:
            triple_indices: List of triple indices to include

        Returns:
            PyG Data object containing all specified triples

        Note:
            Useful for creating graphs from multiple related triples
            (e.g., all triples in a reasoning chain).
        """
        selected_triples = [self.triples[i] for i in triple_indices]
        return self.graph_constructor.construct(selected_triples)

    def save_vocabulary(self, path: Optional[Path] = None):
        """
        Save vocabulary to disk.

        Args:
            path: Directory path (if None, uses self.processed_dir)
        """
        path = path or self.processed_dir
        self.vocabulary.save(path)

    def load_vocabulary(self, path: Optional[Path] = None):
        """
        Load vocabulary from disk.

        Args:
            path: Directory path (if None, uses self.processed_dir)
        """
        path = path or self.processed_dir
        self.vocabulary = TripleVocabulary.load(path)
        self.graph_constructor.vocabulary = self.vocabulary

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute dataset statistics.

        Returns:
            Dictionary containing:
                - num_triples: Total number of triples
                - num_entities: Unique entities
                - num_predicates: Unique predicates
                - avg_confidence: Average confidence score
                - level_distribution: Count of triples per level
        """
        collection = TripleCollection(self.triples)

        level_dist = {}
        for triple in self.triples:
            level_dist[triple.level] = level_dist.get(triple.level, 0) + 1

        return {
            'num_triples': len(self.triples),
            'num_entities': len(collection.get_unique_entities()),
            'num_predicates': len(collection.get_unique_predicates()),
            'avg_confidence': collection.get_confidence_tensor().mean().item(),
            'level_distribution': level_dist,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"split='{self.split}', "
            f"num_samples={len(self)}, "
            f"num_entities={self.vocabulary.num_entities}, "
            f"num_predicates={self.vocabulary.num_predicates})"
        )


class SyntheticTripleDataset(BaseSemanticTripleDataset):
    """
    Simple synthetic dataset for testing and development.

    Generates random semantic triples with configurable size.
    Useful for validating data pipeline before implementing
    domain-specific datasets.

    Examples:
        >>> dataset = SyntheticTripleDataset(
        ...     root="data/synthetic",
        ...     num_entities=100,
        ...     num_predicates=10,
        ...     num_triples=1000
        ... )
        >>> graph, label = dataset[0]
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        num_entities: int = 100,
        num_predicates: int = 10,
        num_triples: int = 1000,
        num_levels: int = 2,
        **kwargs
    ):
        """
        Initialize synthetic dataset.

        Args:
            root: Root directory
            split: Dataset split
            num_entities: Number of unique entities
            num_predicates: Number of unique predicates
            num_triples: Number of triples to generate
            num_levels: Number of hierarchy levels
            **kwargs: Additional arguments for BaseSemanticTripleDataset
        """
        self.num_entities = num_entities
        self.num_predicates = num_predicates
        self.num_triples = num_triples
        self.num_levels = num_levels

        super().__init__(root, split, **kwargs)

    def generate_triples(self) -> List[SemanticTriple]:
        """Generate random semantic triples."""
        triples = []

        for i in range(self.num_triples):
            # Random entities and predicates
            subject = f"entity_{torch.randint(0, self.num_entities, (1,)).item()}"
            predicate = f"pred_{torch.randint(0, self.num_predicates, (1,)).item()}"
            obj = f"entity_{torch.randint(0, self.num_entities, (1,)).item()}"

            # Random confidence and level
            confidence = torch.rand(1).item()
            level = torch.randint(1, self.num_levels + 1, (1,)).item()

            triple = SemanticTriple(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=confidence,
                level=level,
                metadata={'idx': i}
            )
            triples.append(triple)

        return triples

    def generate_labels(self, idx: int) -> torch.Tensor:
        """
        Generate dummy classification labels.

        Returns:
            Random binary label
        """
        return torch.randint(0, 2, (1,), dtype=torch.long)
