"""
Unit tests for SemanticTriple and TripleCollection classes.
"""

import pytest
import torch
from nsm.data.triple import SemanticTriple, TripleCollection


class TestSemanticTriple:
    """Tests for SemanticTriple class."""

    def test_basic_creation(self):
        """Test creating a basic semantic triple."""
        triple = SemanticTriple(
            subject="robot",
            predicate="executes",
            object="move_left",
            confidence=0.9,
            level=1
        )

        assert triple.subject == "robot"
        assert triple.predicate == "executes"
        assert triple.object == "move_left"
        assert triple.confidence == 0.9
        assert triple.level == 1
        assert isinstance(triple.metadata, dict)

    def test_default_values(self):
        """Test default confidence and level values."""
        triple = SemanticTriple(
            subject="A",
            predicate="relates_to",
            object="B"
        )

        assert triple.confidence == 1.0
        assert triple.level == 1
        assert triple.metadata == {}

    def test_confidence_validation(self):
        """Test that confidence must be in [0, 1]."""
        # Valid confidences
        SemanticTriple("A", "r", "B", confidence=0.0)
        SemanticTriple("A", "r", "B", confidence=0.5)
        SemanticTriple("A", "r", "B", confidence=1.0)

        # Invalid confidences
        with pytest.raises(ValueError, match="Confidence must be in"):
            SemanticTriple("A", "r", "B", confidence=-0.1)

        with pytest.raises(ValueError, match="Confidence must be in"):
            SemanticTriple("A", "r", "B", confidence=1.5)

    def test_level_validation(self):
        """Test that level must be in [1, 6]."""
        # Valid levels
        for level in range(1, 7):
            SemanticTriple("A", "r", "B", level=level)

        # Invalid levels
        with pytest.raises(ValueError, match="Level must be in"):
            SemanticTriple("A", "r", "B", level=0)

        with pytest.raises(ValueError, match="Level must be in"):
            SemanticTriple("A", "r", "B", level=7)

    def test_to_tensor(self):
        """Test conversion to PyTorch tensor."""
        triple = SemanticTriple("A", "r", "B", confidence=0.75)
        tensor = triple.to_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.item() == 0.75
        assert tensor.dtype == torch.float32

    def test_update_confidence(self):
        """Test updating confidence value."""
        triple = SemanticTriple("A", "r", "B", confidence=0.5)

        triple.update_confidence(0.9)
        assert triple.confidence == 0.9

        with pytest.raises(ValueError):
            triple.update_confidence(1.5)

    def test_is_concrete(self):
        """Test is_concrete method."""
        concrete = SemanticTriple("A", "r", "B", level=1)
        abstract = SemanticTriple("A", "r", "B", level=2)

        assert concrete.is_concrete()
        assert not abstract.is_concrete()

    def test_is_abstract(self):
        """Test is_abstract method."""
        concrete = SemanticTriple("A", "r", "B", level=1)
        abstract = SemanticTriple("A", "r", "B", level=2)

        assert not concrete.is_abstract()
        assert abstract.is_abstract()

    def test_equality(self):
        """Test equality comparison (ignores confidence and metadata)."""
        t1 = SemanticTriple("A", "r", "B", confidence=0.8, level=1)
        t2 = SemanticTriple("A", "r", "B", confidence=0.9, level=1)
        t3 = SemanticTriple("A", "r", "C", confidence=0.8, level=1)

        assert t1 == t2  # Same except confidence
        assert t1 != t3  # Different object
        assert t1 != "not a triple"

    def test_hash(self):
        """Test hashing for use in sets/dicts."""
        t1 = SemanticTriple("A", "r", "B", level=1)
        t2 = SemanticTriple("A", "r", "B", level=1)
        t3 = SemanticTriple("A", "r", "C", level=1)

        # Same triples should have same hash
        assert hash(t1) == hash(t2)

        # Can use in sets
        triple_set = {t1, t2, t3}
        assert len(triple_set) == 2  # t1 and t2 are duplicates

    def test_repr(self):
        """Test string representation."""
        triple = SemanticTriple("robot", "executes", "move", confidence=0.95, level=1)
        repr_str = repr(triple)

        assert "robot" in repr_str
        assert "executes" in repr_str
        assert "move" in repr_str
        assert "0.950" in repr_str
        assert "level=1" in repr_str

    def test_metadata(self):
        """Test metadata handling."""
        triple = SemanticTriple(
            "A", "r", "B",
            metadata={"source": "planner", "timestamp": "2024-01-01"}
        )

        assert triple.metadata["source"] == "planner"
        assert triple.metadata["timestamp"] == "2024-01-01"


class TestTripleCollection:
    """Tests for TripleCollection class."""

    @pytest.fixture
    def sample_triples(self):
        """Create sample triples for testing."""
        return [
            SemanticTriple("A", "r1", "B", confidence=0.9, level=1),
            SemanticTriple("B", "r2", "C", confidence=0.8, level=1),
            SemanticTriple("C", "r3", "D", confidence=0.7, level=2),
            SemanticTriple("D", "r4", "E", confidence=0.6, level=2),
        ]

    def test_creation(self, sample_triples):
        """Test creating a collection."""
        collection = TripleCollection(sample_triples)
        assert len(collection) == 4

        empty_collection = TripleCollection()
        assert len(empty_collection) == 0

    def test_add_and_extend(self):
        """Test adding triples to collection."""
        collection = TripleCollection()

        t1 = SemanticTriple("A", "r", "B")
        collection.add(t1)
        assert len(collection) == 1

        t2 = SemanticTriple("B", "r", "C")
        t3 = SemanticTriple("C", "r", "D")
        collection.extend([t2, t3])
        assert len(collection) == 3

    def test_filter_by_level(self, sample_triples):
        """Test filtering by hierarchical level."""
        collection = TripleCollection(sample_triples)

        level1 = collection.filter_by_level(1)
        assert len(level1) == 2
        assert all(t.level == 1 for t in level1)

        level2 = collection.filter_by_level(2)
        assert len(level2) == 2
        assert all(t.level == 2 for t in level2)

    def test_filter_by_confidence(self, sample_triples):
        """Test filtering by confidence range."""
        collection = TripleCollection(sample_triples)

        high_conf = collection.filter_by_confidence(min_conf=0.8)
        assert len(high_conf) == 2
        assert all(t.confidence >= 0.8 for t in high_conf)

        mid_conf = collection.filter_by_confidence(min_conf=0.7, max_conf=0.8)
        assert len(mid_conf) == 2
        assert all(0.7 <= t.confidence <= 0.8 for t in mid_conf)

    def test_get_unique_subjects(self, sample_triples):
        """Test getting unique subjects."""
        collection = TripleCollection(sample_triples)
        subjects = collection.get_unique_subjects()

        assert subjects == {"A", "B", "C", "D"}

    def test_get_unique_objects(self, sample_triples):
        """Test getting unique objects."""
        collection = TripleCollection(sample_triples)
        objects = collection.get_unique_objects()

        assert objects == {"B", "C", "D", "E"}

    def test_get_unique_entities(self, sample_triples):
        """Test getting all unique entities."""
        collection = TripleCollection(sample_triples)
        entities = collection.get_unique_entities()

        assert entities == {"A", "B", "C", "D", "E"}

    def test_get_unique_predicates(self, sample_triples):
        """Test getting unique predicates."""
        collection = TripleCollection(sample_triples)
        predicates = collection.get_unique_predicates()

        assert predicates == {"r1", "r2", "r3", "r4"}

    def test_get_confidence_tensor(self, sample_triples):
        """Test getting confidence scores as tensor."""
        collection = TripleCollection(sample_triples)
        confidences = collection.get_confidence_tensor()

        assert isinstance(confidences, torch.Tensor)
        assert confidences.shape == (4,)
        assert torch.allclose(
            confidences,
            torch.tensor([0.9, 0.8, 0.7, 0.6])
        )

    def test_iteration(self, sample_triples):
        """Test iterating over collection."""
        collection = TripleCollection(sample_triples)

        count = 0
        for triple in collection:
            assert isinstance(triple, SemanticTriple)
            count += 1

        assert count == 4

    def test_indexing(self, sample_triples):
        """Test accessing triples by index."""
        collection = TripleCollection(sample_triples)

        assert collection[0].subject == "A"
        assert collection[1].subject == "B"
        assert collection[-1].subject == "D"

    def test_repr(self, sample_triples):
        """Test string representation."""
        collection = TripleCollection(sample_triples)
        repr_str = repr(collection)

        assert "TripleCollection" in repr_str
        assert "num_triples=4" in repr_str
