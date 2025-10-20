"""
Unit tests for Vocabulary and TripleVocabulary classes.
"""

import pytest
import tempfile
from pathlib import Path
from nsm.data.vocabulary import Vocabulary, TripleVocabulary


class TestVocabulary:
    """Tests for Vocabulary class."""

    def test_creation(self):
        """Test creating a vocabulary."""
        vocab = Vocabulary(name="test")
        assert vocab.name == "test"
        assert len(vocab) == 0

    def test_special_tokens(self):
        """Test adding special tokens during initialization."""
        special_tokens = ["<PAD>", "<UNK>", "<MASK>"]
        vocab = Vocabulary(name="test", special_tokens=special_tokens)

        assert len(vocab) == 3
        assert vocab.get_index("<PAD>") == 0
        assert vocab.get_index("<UNK>") == 1
        assert vocab.get_index("<MASK>") == 2

    def test_add_token(self):
        """Test adding tokens to vocabulary."""
        vocab = Vocabulary()

        idx1 = vocab.add_token("hello")
        assert idx1 == 0

        idx2 = vocab.add_token("world")
        assert idx2 == 1

        # Adding same token returns same index
        idx3 = vocab.add_token("hello")
        assert idx3 == 0

    def test_add_tokens_batch(self):
        """Test adding multiple tokens at once."""
        vocab = Vocabulary()
        indices = vocab.add_tokens(["a", "b", "c", "a"])

        assert indices == [0, 1, 2, 0]  # "a" appears twice
        assert len(vocab) == 3

    def test_get_index(self):
        """Test getting index for a token."""
        vocab = Vocabulary()
        vocab.add_token("test")

        assert vocab.get_index("test") == 0

        with pytest.raises(KeyError):
            vocab.get_index("unknown")

        # With default
        assert vocab.get_index("unknown", default=-1) == -1

    def test_get_token(self):
        """Test getting token for an index."""
        vocab = Vocabulary()
        vocab.add_token("test")

        assert vocab.get_token(0) == "test"

        with pytest.raises(KeyError):
            vocab.get_token(999)

        # With default
        assert vocab.get_token(999, default="<UNK>") == "<UNK>"

    def test_contains(self):
        """Test checking if token exists."""
        vocab = Vocabulary()
        vocab.add_token("exists")

        assert "exists" in vocab
        assert "missing" not in vocab

        # Can also check indices
        assert 0 in vocab
        assert 999 not in vocab

    def test_integer_tokens(self):
        """Test that integer tokens are handled correctly."""
        vocab = Vocabulary()

        # Adding integer returns it directly
        assert vocab.add_token(42) == 42
        assert vocab.get_index(42) == 42

    def test_save_and_load(self):
        """Test saving and loading vocabulary."""
        vocab = Vocabulary(name="test")
        vocab.add_tokens(["a", "b", "c"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.json"
            vocab.save(path)

            loaded_vocab = Vocabulary.load(path)

            assert loaded_vocab.name == vocab.name
            assert len(loaded_vocab) == len(vocab)
            assert loaded_vocab.get_index("a") == vocab.get_index("a")
            assert loaded_vocab.get_token(0) == vocab.get_token(0)

    def test_repr(self):
        """Test string representation."""
        vocab = Vocabulary(name="test")
        vocab.add_tokens(["a", "b", "c"])

        repr_str = repr(vocab)
        assert "test" in repr_str
        assert "size=3" in repr_str


class TestTripleVocabulary:
    """Tests for TripleVocabulary class."""

    def test_creation(self):
        """Test creating a triple vocabulary."""
        vocab = TripleVocabulary()

        assert isinstance(vocab.entity_vocab, Vocabulary)
        assert isinstance(vocab.predicate_vocab, Vocabulary)
        assert vocab.num_entities == 0
        assert vocab.num_predicates == 0

    def test_special_tokens(self):
        """Test adding special tokens to both vocabularies."""
        vocab = TripleVocabulary(
            entity_special_tokens=["<PAD>"],
            predicate_special_tokens=["<SELF>"]
        )

        assert vocab.entity_vocab.get_index("<PAD>") == 0
        assert vocab.predicate_vocab.get_index("<SELF>") == 0

    def test_add_triple_entities(self):
        """Test adding entities from a triple."""
        vocab = TripleVocabulary()

        subj_idx, pred_idx, obj_idx = vocab.add_triple_entities(
            "robot", "executes", "move"
        )

        assert subj_idx == 0
        assert pred_idx == 0
        assert obj_idx == 1

        assert vocab.num_entities == 2  # robot, move
        assert vocab.num_predicates == 1  # executes

    def test_get_entity_index(self):
        """Test getting entity indices."""
        vocab = TripleVocabulary()
        vocab.add_triple_entities("A", "r", "B")

        assert vocab.get_entity_index("A") == 0
        assert vocab.get_entity_index("B") == 1

        with pytest.raises(KeyError):
            vocab.get_entity_index("C")

        assert vocab.get_entity_index("C", default=-1) == -1

    def test_get_predicate_index(self):
        """Test getting predicate indices."""
        vocab = TripleVocabulary()
        vocab.add_triple_entities("A", "r1", "B")
        vocab.add_triple_entities("B", "r2", "C")

        assert vocab.get_predicate_index("r1") == 0
        assert vocab.get_predicate_index("r2") == 1

    def test_get_tokens(self):
        """Test getting tokens from indices."""
        vocab = TripleVocabulary()
        vocab.add_triple_entities("robot", "executes", "move")

        assert vocab.get_entity_token(0) == "robot"
        assert vocab.get_predicate_token(0) == "executes"

    def test_save_and_load(self):
        """Test saving and loading triple vocabulary."""
        vocab = TripleVocabulary()
        vocab.add_triple_entities("A", "r1", "B")
        vocab.add_triple_entities("B", "r2", "C")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            vocab.save(path)

            loaded_vocab = TripleVocabulary.load(path)

            assert loaded_vocab.num_entities == vocab.num_entities
            assert loaded_vocab.num_predicates == vocab.num_predicates
            assert loaded_vocab.get_entity_index("A") == vocab.get_entity_index("A")
            assert loaded_vocab.get_predicate_index("r1") == vocab.get_predicate_index("r1")

    def test_repr(self):
        """Test string representation."""
        vocab = TripleVocabulary()
        vocab.add_triple_entities("A", "r", "B")
        vocab.add_triple_entities("B", "r", "C")

        repr_str = repr(vocab)
        assert "TripleVocabulary" in repr_str
        assert "entities=3" in repr_str  # A, B, C
        assert "predicates=1" in repr_str  # r
