"""
Vocabulary Management for Semantic Triples

Handles mapping between string identifiers and integer indices
for entities (nodes) and predicates (edge types).
"""

from typing import Dict, List, Optional, Union
import json
from pathlib import Path


class Vocabulary:
    """
    Manages bidirectional mappings between strings and indices.

    Used for converting entity names and predicate types to/from
    integer indices required by PyTorch Geometric.

    Attributes:
        token_to_idx: Mapping from token (str) to index (int)
        idx_to_token: Mapping from index (int) to token (str)
        name: Name of this vocabulary (e.g., 'entities', 'predicates')

    Examples:
        >>> vocab = Vocabulary(name='entities')
        >>> vocab.add_token('robot')
        0
        >>> vocab.add_token('goal')
        1
        >>> vocab.get_index('robot')
        0
        >>> vocab.get_token(1)
        'goal'
    """

    def __init__(
        self,
        name: str = "vocab",
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize vocabulary.

        Args:
            name: Name identifier for this vocabulary
            special_tokens: Optional list of special tokens to add first
                           (e.g., ['<PAD>', '<UNK>', '<MASK>'])
        """
        self.name = name
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self._next_idx = 0

        # Add special tokens if provided
        if special_tokens:
            for token in special_tokens:
                self.add_token(token)

    def add_token(self, token: Union[str, int]) -> int:
        """
        Add a token to vocabulary if not present.

        Args:
            token: String token to add (or int to keep as-is)

        Returns:
            Index assigned to the token
        """
        # If already an int, return it
        if isinstance(token, int):
            return token

        # If token already exists, return its index
        if token in self.token_to_idx:
            return self.token_to_idx[token]

        # Add new token
        idx = self._next_idx
        self.token_to_idx[token] = idx
        self.idx_to_token[idx] = token
        self._next_idx += 1
        return idx

    def add_tokens(self, tokens: List[Union[str, int]]) -> List[int]:
        """
        Add multiple tokens to vocabulary.

        Args:
            tokens: List of tokens to add

        Returns:
            List of indices for all tokens
        """
        return [self.add_token(token) for token in tokens]

    def get_index(
        self,
        token: Union[str, int],
        default: Optional[int] = None
    ) -> int:
        """
        Get index for a token.

        Args:
            token: Token to lookup
            default: Value to return if token not found (if None, raises KeyError)

        Returns:
            Index of the token

        Raises:
            KeyError: If token not in vocabulary and default is None
        """
        if isinstance(token, int):
            return token

        if default is not None:
            return self.token_to_idx.get(token, default)
        return self.token_to_idx[token]

    def get_token(self, idx: int, default: Optional[str] = None) -> str:
        """
        Get token for an index.

        Args:
            idx: Index to lookup
            default: Value to return if index not found

        Returns:
            Token string at the index

        Raises:
            KeyError: If index not in vocabulary and default is None
        """
        if default is not None:
            return self.idx_to_token.get(idx, default)
        return self.idx_to_token[idx]

    def __contains__(self, token: Union[str, int]) -> bool:
        """Check if token exists in vocabulary."""
        if isinstance(token, int):
            return token in self.idx_to_token
        return token in self.token_to_idx

    def __len__(self) -> int:
        """Number of tokens in vocabulary."""
        return len(self.token_to_idx)

    def __repr__(self) -> str:
        """String representation."""
        return f"Vocabulary(name='{self.name}', size={len(self)})"

    def save(self, path: Union[str, Path]) -> None:
        """
        Save vocabulary to JSON file.

        Args:
            path: File path to save to
        """
        path = Path(path)
        data = {
            "name": self.name,
            "token_to_idx": self.token_to_idx,
            "idx_to_token": {str(k): v for k, v in self.idx_to_token.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocabulary":
        """
        Load vocabulary from JSON file.

        Args:
            path: File path to load from

        Returns:
            Loaded Vocabulary object
        """
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)

        vocab = cls(name=data["name"])
        vocab.token_to_idx = data["token_to_idx"]
        vocab.idx_to_token = {int(k): v for k, v in data["idx_to_token"].items()}
        vocab._next_idx = len(vocab.token_to_idx)
        return vocab


class TripleVocabulary:
    """
    Manages vocabularies for semantic triples.

    Maintains separate vocabularies for entities (subjects/objects)
    and predicates (edge types).

    Attributes:
        entity_vocab: Vocabulary for entity identifiers
        predicate_vocab: Vocabulary for predicate types

    Examples:
        >>> vocab = TripleVocabulary()
        >>> # Add entities and predicates from triples
        >>> vocab.add_triple_entities('robot', 'executes', 'move_left')
        >>> # Get indices
        >>> subj_idx = vocab.get_entity_index('robot')
        >>> pred_idx = vocab.get_predicate_index('executes')
    """

    def __init__(
        self,
        entity_special_tokens: Optional[List[str]] = None,
        predicate_special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize triple vocabularies.

        Args:
            entity_special_tokens: Special tokens for entity vocabulary
            predicate_special_tokens: Special tokens for predicate vocabulary
        """
        self.entity_vocab = Vocabulary(
            name="entities",
            special_tokens=entity_special_tokens
        )
        self.predicate_vocab = Vocabulary(
            name="predicates",
            special_tokens=predicate_special_tokens
        )

    def add_triple_entities(
        self,
        subject: Union[str, int],
        predicate: Union[str, int],
        obj: Union[str, int]
    ) -> tuple[int, int, int]:
        """
        Add subject, predicate, object to respective vocabularies.

        Args:
            subject: Subject entity
            predicate: Predicate type
            obj: Object entity

        Returns:
            Tuple of (subject_idx, predicate_idx, object_idx)
        """
        subj_idx = self.entity_vocab.add_token(subject)
        pred_idx = self.predicate_vocab.add_token(predicate)
        obj_idx = self.entity_vocab.add_token(obj)
        return subj_idx, pred_idx, obj_idx

    def get_entity_index(
        self,
        entity: Union[str, int],
        default: Optional[int] = None
    ) -> int:
        """Get index for an entity."""
        return self.entity_vocab.get_index(entity, default)

    def get_predicate_index(
        self,
        predicate: Union[str, int],
        default: Optional[int] = None
    ) -> int:
        """Get index for a predicate."""
        return self.predicate_vocab.get_index(predicate, default)

    def get_entity_token(self, idx: int, default: Optional[str] = None) -> str:
        """Get entity token for an index."""
        return self.entity_vocab.get_token(idx, default)

    def get_predicate_token(self, idx: int, default: Optional[str] = None) -> str:
        """Get predicate token for an index."""
        return self.predicate_vocab.get_token(idx, default)

    @property
    def num_entities(self) -> int:
        """Number of unique entities."""
        return len(self.entity_vocab)

    @property
    def num_predicates(self) -> int:
        """Number of unique predicates."""
        return len(self.predicate_vocab)

    def save(self, dir_path: Union[str, Path]) -> None:
        """
        Save both vocabularies to directory.

        Args:
            dir_path: Directory to save vocabulary files
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        self.entity_vocab.save(dir_path / "entity_vocab.json")
        self.predicate_vocab.save(dir_path / "predicate_vocab.json")

    @classmethod
    def load(cls, dir_path: Union[str, Path]) -> "TripleVocabulary":
        """
        Load vocabularies from directory.

        Args:
            dir_path: Directory containing vocabulary files

        Returns:
            Loaded TripleVocabulary object
        """
        dir_path = Path(dir_path)
        vocab = cls()
        vocab.entity_vocab = Vocabulary.load(dir_path / "entity_vocab.json")
        vocab.predicate_vocab = Vocabulary.load(dir_path / "predicate_vocab.json")
        return vocab

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TripleVocabulary("
            f"entities={len(self.entity_vocab)}, "
            f"predicates={len(self.predicate_vocab)})"
        )
