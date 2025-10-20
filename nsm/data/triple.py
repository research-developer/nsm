"""
Semantic Triple Data Structure

This module defines the core SemanticTriple class representing
subject-predicate-object relationships with confidence scores and
hierarchical level information.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
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
        confidence: Learnable confidence score in [0, 1], default 1.0
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
    confidence: float = 1.0
    level: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate triple attributes after initialization."""
        # Validate confidence in [0, 1]
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0, 1], got {self.confidence}"
            )

        # Validate level (Phase 1: 1-2, Phase 2+: 1-6)
        if not 1 <= self.level <= 6:
            raise ValueError(
                f"Level must be in [1, 6], got {self.level}"
            )

    def to_tensor(self) -> Tensor:
        """
        Convert confidence to a PyTorch tensor.

        Returns:
            Tensor: Scalar tensor containing confidence value
        """
        return torch.tensor(self.confidence, dtype=torch.float32)

    def update_confidence(self, new_confidence: float) -> None:
        """
        Update the confidence score.

        Args:
            new_confidence: New confidence value in [0, 1]

        Raises:
            ValueError: If new_confidence not in [0, 1]
        """
        if not 0.0 <= new_confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0, 1], got {new_confidence}"
            )
        self.confidence = new_confidence

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

    def get_confidence_tensor(self) -> Tensor:
        """
        Get confidence scores as a tensor.

        Returns:
            Tensor: Shape [num_triples] containing all confidence scores
        """
        confidences = [t.confidence for t in self.triples]
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
