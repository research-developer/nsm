"""
Semantic Triple Data Structure

This module defines the core SemanticTriple class representing
subject-predicate-object relationships with confidence scores and
hierarchical level information.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor


@dataclass
class SemanticTriple:
    """
    Represents a semantic triple (subject, predicate, object) with metadata.

    A semantic triple is the fundamental unit of knowledge representation in NSM.
    Each triple exists at a specific abstraction level and carries a confidence
    score indicating the certainty of the relationship.

    Attributes:
        subject: The subject entity (string identifier or index)
        predicate: The relationship type (string identifier or index)
        object: The object entity (string identifier or index)
        confidence: Learnable confidence score. Accepts scalar in [0, 1]
                    or length-4 log-score tensor/list.
        level: Hierarchical level (1=concrete, 2=abstract for Phase 1)
        metadata: Additional information (provenance, timestamp, etc.)

    Examples:
        >>> # Concrete action triple (Level 1)
        >>> t1 = SemanticTriple(
        ...     subject="robot",
        ...     predicate="executes",
        ...     object="move_left",
        ...     confidence=0.95,
        ...     level=1
        ... )

        >>> # Abstract goal triple (Level 2)
        >>> t2 = SemanticTriple(
        ...     subject="system",
        ...     predicate="achieves",
        ...     object="navigate_to_goal",
        ...     confidence=0.8,
        ...     level=2
        ... )

        >>> # Triple with metadata
        >>> t3 = SemanticTriple(
        ...     subject="task_42",
        ...     predicate="requires",
        ...     object="capability_planning",
        ...     level=2,
        ...     metadata={"source": "planner", "timestamp": "2024-01-01"}
        ... )
    """

    subject: Union[str, int]
    predicate: Union[str, int]
    object: Union[str, int]
    confidence: Union[float, Sequence[float], Tensor] = 1.0
    level: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    _confidence_tensor: Tensor = field(init=False, repr=False, compare=False)
    _confidence_is_scalar: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate triple attributes after initialization."""
        self._set_confidence(self.confidence)

        # Validate level (Phase 1: 1-2, Phase 2+: 1-6)
        if not 1 <= self.level <= 6:
            raise ValueError(
                f"Level must be in [1, 6], got {self.level}"
            )

    def _set_confidence(
        self,
        confidence_value: Union[float, Sequence[float], Tensor]
    ) -> None:
        """Normalize confidence input into scalar + 4-vector tensor."""
        tensor, is_scalar, scalar_value = self._coerce_confidence_tensor(
            confidence_value
        )

        self._confidence_tensor = tensor
        self._confidence_is_scalar = is_scalar

        if is_scalar:
            self.confidence = scalar_value
        else:
            # Aggregate vector confidence into scalar summary for legacy paths.
            probs = torch.softmax(tensor, dim=0)
            self.confidence = float(probs.max().item())

    @staticmethod
    def _coerce_confidence_tensor(
        confidence_value: Union[float, Sequence[float], Tensor]
    ) -> Tuple[Tensor, bool, float]:
        """Convert confidence inputs into a standardized tensor."""
        if isinstance(confidence_value, Tensor):
            tensor = confidence_value.detach().clone().to(dtype=torch.float32)
            flat = tensor.reshape(-1)
            if flat.numel() == 1:
                scalar = float(flat.item())
                if not 0.0 <= scalar <= 1.0:
                    raise ValueError(
                        f"Confidence must be in [0, 1], got {scalar}"
                    )
                vector = torch.full((4,), scalar, dtype=torch.float32)
                return vector, True, scalar
            if flat.numel() != 4:
                raise ValueError(
                    "Confidence tensor must have 4 elements, "
                    f"got shape {tuple(tensor.shape)}"
                )
            return flat.reshape(4), False, float('nan')

        if isinstance(confidence_value, (int, float)):
            scalar = float(confidence_value)
            if not 0.0 <= scalar <= 1.0:
                raise ValueError(
                    f"Confidence must be in [0, 1], got {scalar}"
                )
            vector = torch.full((4,), scalar, dtype=torch.float32)
            return vector, True, scalar

        if isinstance(confidence_value, Sequence):
            values = list(confidence_value)
            tensor = torch.tensor(values, dtype=torch.float32)
            flat = tensor.reshape(-1)
            if flat.numel() == 1:
                return SemanticTriple._coerce_confidence_tensor(flat.item())
            if flat.numel() != 4:
                raise ValueError(
                    "Confidence sequence must contain exactly 4 values, "
                    f"got {len(values)}"
                )
            return flat.reshape(4), False, float('nan')

        raise TypeError(
            "Confidence must be a float, Tensor, or sequence of floats."
        )

    def uses_confidence_vector(self) -> bool:
        """Return True if the triple stores a 4-channel confidence vector."""
        return not self._confidence_is_scalar

    def get_confidence_tensor(self) -> Tensor:
        """Return the 4-element confidence tensor for this triple."""
        return self._confidence_tensor.clone()

    def to_tensor(self) -> Tensor:
        """
        Convert confidence representation to a PyTorch tensor.

        Returns:
            Tensor: Shape [4] tensor containing confidence values/log-scores
        """
        return self.get_confidence_tensor()

    def update_confidence(
        self,
        new_confidence: Union[float, Sequence[float], Tensor]
    ) -> None:
        """
        Update the confidence score.

        Args:
            new_confidence: New confidence value(s)

        Raises:
            ValueError: If new_confidence has invalid value or shape
        """
        self._set_confidence(new_confidence)

    def is_concrete(self) -> bool:
        """Check if this triple represents concrete actions/environment."""
        return self.level == 1

    def is_abstract(self) -> bool:
        """Check if this triple represents abstract goals/capabilities."""
        return self.level >= 2

    def __repr__(self) -> str:
        """String representation showing key attributes."""
        return (
            f"SemanticTriple("
            f"subject={self.subject!r}, "
            f"predicate={self.predicate!r}, "
            f"object={self.object!r}, "
            f"confidence={self.confidence:.3f}, "
            f"level={self.level})"
        )

    def __eq__(self, other: object) -> bool:
        """
        Equality comparison (ignores confidence and metadata).

        Two triples are considered equal if they have the same
        subject, predicate, object, and level, regardless of
        confidence score or metadata.
        """
        if not isinstance(other, SemanticTriple):
            return NotImplemented
        return (
            self.subject == other.subject
            and self.predicate == other.predicate
            and self.object == other.object
            and self.level == other.level
        )

    def __hash__(self) -> int:
        """Hash based on subject, predicate, object, and level."""
        return hash((self.subject, self.predicate, self.object, self.level))


@dataclass
class TripleCollection:
    """
    Collection of semantic triples with utility methods.

    Provides batch operations on multiple triples, useful for
    constructing graphs and managing knowledge bases.

    Attributes:
        triples: List of SemanticTriple objects

    Examples:
        >>> collection = TripleCollection([t1, t2, t3])
        >>> concrete_triples = collection.filter_by_level(1)
        >>> high_conf_triples = collection.filter_by_confidence(min_conf=0.9)
    """

    triples: list[SemanticTriple] = field(default_factory=list)

    def add(self, triple: SemanticTriple) -> None:
        """Add a triple to the collection."""
        self.triples.append(triple)

    def extend(self, triples: list[SemanticTriple]) -> None:
        """Add multiple triples to the collection."""
        self.triples.extend(triples)

    def filter_by_level(self, level: int) -> "TripleCollection":
        """
        Filter triples by hierarchical level.

        Args:
            level: Level to filter (1=concrete, 2+=abstract)

        Returns:
            New TripleCollection containing only triples at specified level
        """
        filtered = [t for t in self.triples if t.level == level]
        return TripleCollection(filtered)

    def filter_by_confidence(
        self,
        min_conf: float = 0.0,
        max_conf: float = 1.0
    ) -> "TripleCollection":
        """
        Filter triples by confidence range.

        Args:
            min_conf: Minimum confidence threshold (inclusive)
            max_conf: Maximum confidence threshold (inclusive)

        Returns:
            New TripleCollection containing triples in confidence range
        """
        filtered = [
            t for t in self.triples
            if min_conf <= t.confidence <= max_conf
        ]
        return TripleCollection(filtered)

    def get_unique_subjects(self) -> set[Union[str, int]]:
        """Get set of all unique subject entities."""
        return {t.subject for t in self.triples}

    def get_unique_objects(self) -> set[Union[str, int]]:
        """Get set of all unique object entities."""
        return {t.object for t in self.triples}

    def get_unique_entities(self) -> set[Union[str, int]]:
        """Get set of all unique entities (subjects and objects)."""
        return self.get_unique_subjects() | self.get_unique_objects()

    def get_unique_predicates(self) -> set[Union[str, int]]:
        """Get set of all unique predicate types."""
        return {t.predicate for t in self.triples}

    def get_confidence_tensor(self, *, as_vector: bool = False) -> Tensor:
        """
        Get confidence scores as a tensor.

        Args:
            as_vector: If True, returns stacked [num_triples, 4] tensors.
                       If False (default), returns scalar confidences.

        Returns:
            Tensor containing confidence information.
                - Shape [num_triples] when as_vector is False
                - Shape [num_triples, 4] when as_vector is True
        """
        if not self.triples:
            if as_vector:
                return torch.zeros(0, 4, dtype=torch.float32)
            return torch.zeros(0, dtype=torch.float32)

        if as_vector:
            tensors = [t.get_confidence_tensor() for t in self.triples]
            return torch.stack(tensors, dim=0)

        confidences = [float(t.confidence) for t in self.triples]
        return torch.tensor(confidences, dtype=torch.float32)

    def __len__(self) -> int:
        """Number of triples in collection."""
        return len(self.triples)

    def __iter__(self):
        """Iterate over triples."""
        return iter(self.triples)

    def __getitem__(self, idx: int) -> SemanticTriple:
        """Get triple by index."""
        return self.triples[idx]

    def __repr__(self) -> str:
        """String representation."""
        return f"TripleCollection(num_triples={len(self.triples)})"
