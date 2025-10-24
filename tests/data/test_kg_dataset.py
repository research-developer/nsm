"""
Tests for Knowledge Graph Triple Dataset

Validates KG dataset generation, entity diversity, multi-hop reasoning,
and type hierarchy consistency.
"""

import pytest
import torch
import shutil
from pathlib import Path

from nsm.data.knowledge_graph_dataset import KnowledgeGraphTripleDataset
from nsm.data.triple import SemanticTriple


@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary data directory."""
    data_dir = tmp_path / "kg_test"
    yield data_dir
    # Cleanup
    if data_dir.exists():
        shutil.rmtree(data_dir)


@pytest.fixture
def small_kg_dataset(test_data_dir):
    """Create small KG dataset for testing."""
    return KnowledgeGraphTripleDataset(
        root=str(test_data_dir),
        split='train',
        num_entities=100,
        num_triples=500,
        seed=42
    )


@pytest.fixture
def medium_kg_dataset(test_data_dir):
    """Create medium KG dataset."""
    return KnowledgeGraphTripleDataset(
        root=str(test_data_dir / "medium"),
        split='train',
        num_entities=1000,
        num_triples=5000,
        seed=42
    )


class TestKGDatasetGeneration:
    """Test dataset generation and basic properties."""

    def test_initialization(self, small_kg_dataset):
        """Test dataset initializes correctly."""
        assert small_kg_dataset is not None
        assert len(small_kg_dataset.triples) == 500
        assert len(small_kg_dataset) == 500

    def test_triple_structure(self, small_kg_dataset):
        """Test that generated triples have correct structure."""
        for triple in small_kg_dataset.triples[:10]:
            assert isinstance(triple, SemanticTriple)
            assert isinstance(triple.subject, str)
            assert isinstance(triple.predicate, str)
            assert isinstance(triple.object, str)
            assert 0.0 <= triple.confidence <= 1.0
            assert triple.level in [1, 2]

    def test_level_distribution(self, small_kg_dataset):
        """Test that both Level 1 and Level 2 triples exist."""
        level_1_count = sum(1 for t in small_kg_dataset.triples if t.level == 1)
        level_2_count = sum(1 for t in small_kg_dataset.triples if t.level == 2)

        assert level_1_count > 0, "Should have Level 1 (fact) triples"
        assert level_2_count > 0, "Should have Level 2 (type) triples"
        assert level_1_count + level_2_count == 500

    def test_confidence_variance(self, small_kg_dataset):
        """Test that confidence scores vary appropriately."""
        confidences = [t.confidence for t in small_kg_dataset.triples]

        # Check range
        assert min(confidences) >= 0.5, "Min confidence should be >= 0.5"
        assert max(confidences) <= 1.0, "Max confidence should be <= 1.0"

        # Check variance (should not all be the same)
        unique_confidences = len(set(confidences))
        assert unique_confidences > 10, "Should have diverse confidence scores"

    def test_predicate_diversity(self, small_kg_dataset):
        """Test that dataset uses diverse predicates."""
        predicates = set(t.predicate for t in small_kg_dataset.triples)

        # Should have at least 15 different predicates
        assert len(predicates) >= 15, f"Only {len(predicates)} unique predicates"

        # Check for expected predicate categories
        level1_preds = set(t.predicate for t in small_kg_dataset.triples if t.level == 1)
        level2_preds = set(t.predicate for t in small_kg_dataset.triples if t.level == 2)

        # Level 1 should include facts
        expected_l1 = ['born_in', 'located_in', 'works_at', 'studied_at', 'won']
        assert any(pred in level1_preds for pred in expected_l1), \
            "Level 1 should include biographical/factual predicates"

        # Level 2 should include types
        expected_l2 = ['instance_of', 'subclass_of', 'typically_has']
        assert any(pred in level2_preds for pred in expected_l2), \
            "Level 2 should include type/category predicates"


class TestEntityDiversity:
    """Test entity generation and diversity."""

    def test_entity_count(self, small_kg_dataset):
        """Test that dataset has sufficient unique entities."""
        entities = set()
        for triple in small_kg_dataset.triples:
            entities.add(triple.subject)
            entities.add(triple.object)

        # Should have diverse entities
        assert len(entities) >= 50, f"Only {len(entities)} unique entities"

    def test_entity_categories(self, small_kg_dataset):
        """Test that entities span multiple categories."""
        # Check that we have different types of entities
        assert len(small_kg_dataset.people) > 0, "Should have people"
        assert len(small_kg_dataset.places) > 0, "Should have places"
        assert len(small_kg_dataset.organizations) > 0, "Should have organizations"
        assert len(small_kg_dataset.concepts) > 0, "Should have concepts"
        assert len(small_kg_dataset.awards) > 0, "Should have awards"

    def test_named_entities(self, small_kg_dataset):
        """Test that famous entities are included."""
        all_entities = set()
        for triple in small_kg_dataset.triples:
            all_entities.add(triple.subject)
            all_entities.add(triple.object)

        # Check for some expected named entities
        expected_people = ["Albert_Einstein", "Marie_Curie", "Isaac_Newton"]
        found_people = [p for p in expected_people if p in all_entities]
        assert len(found_people) > 0, "Should include famous people"

        expected_places = ["London", "Paris", "New_York"]
        found_places = [p for p in expected_places if p in all_entities]
        assert len(found_places) > 0, "Should include major cities"

    def test_entity_types_mapping(self, small_kg_dataset):
        """Test that entities have type mappings."""
        assert len(small_kg_dataset.entity_types) > 0
        assert "Person" in small_kg_dataset.entity_types.values()
        assert "Place" in small_kg_dataset.entity_types.values()


class TestMultiHopReasoning:
    """Test multi-hop reasoning capabilities."""

    def test_multi_hop_query_generation(self, small_kg_dataset):
        """Test generation of multi-hop queries."""
        queries = small_kg_dataset.get_multi_hop_queries(num_queries=20)

        # Should generate some queries
        assert len(queries) > 0, "Should generate multi-hop queries"

        # Check query structure
        for query in queries[:5]:
            assert 'start_entity' in query
            assert 'relations' in query
            assert 'expected_answer' in query
            assert isinstance(query['relations'], list)
            assert len(query['relations']) > 0

    def test_two_hop_reasoning_paths(self, medium_kg_dataset):
        """Test that 2-hop reasoning paths exist."""
        queries = medium_kg_dataset.get_multi_hop_queries(num_queries=50)

        two_hop_queries = [q for q in queries if q.get('query_type') == '2-hop']
        assert len(two_hop_queries) > 0, "Should have 2-hop reasoning paths"

        # Verify path structure
        for query in two_hop_queries[:3]:
            assert len(query['relations']) == 2
            assert 'intermediate' in query
            assert query['start_entity'] != query['expected_answer']


class TestTypeHierarchy:
    """Test type hierarchy and consistency."""

    def test_type_hierarchy_exists(self, small_kg_dataset):
        """Test that type hierarchy is defined."""
        assert len(small_kg_dataset.type_hierarchy) > 0
        assert "Person" in small_kg_dataset.type_hierarchy
        assert "Place" in small_kg_dataset.type_hierarchy

    def test_type_triples(self, small_kg_dataset):
        """Test that instance_of and subclass_of triples exist."""
        instance_of_triples = [
            t for t in small_kg_dataset.triples
            if t.predicate == "instance_of"
        ]
        subclass_of_triples = [
            t for t in small_kg_dataset.triples
            if t.predicate == "subclass_of"
        ]

        assert len(instance_of_triples) > 0, "Should have instance_of triples"
        assert len(subclass_of_triples) > 0, "Should have subclass_of triples"

        # Check confidence for type triples
        for triple in instance_of_triples[:5]:
            assert triple.level == 2, "Type triples should be Level 2"
            assert triple.confidence >= 0.5

    def test_type_consistency_pairs(self, small_kg_dataset):
        """Test generation of type consistency checking pairs."""
        pairs = small_kg_dataset.get_type_consistency_pairs(num_pairs=50)

        assert len(pairs) > 0, "Should generate consistency pairs"

        positive_pairs = [p for p in pairs if p[2] is True]
        negative_pairs = [p for p in pairs if p[2] is False]

        assert len(positive_pairs) > 0, "Should have positive examples"
        assert len(negative_pairs) > 0, "Should have negative examples"

        # Check pair structure
        for entity, entity_type, is_consistent in pairs[:5]:
            assert isinstance(entity, str)
            assert isinstance(entity_type, str)
            assert isinstance(is_consistent, bool)


class TestDatasetInterface:
    """Test PyG dataset interface compliance."""

    def test_getitem(self, small_kg_dataset):
        """Test __getitem__ returns correct format."""
        graph, label = small_kg_dataset[0]

        # Check graph structure
        assert hasattr(graph, 'x'), "Graph should have node features"
        assert hasattr(graph, 'edge_index'), "Graph should have edge_index"
        assert hasattr(graph, 'edge_attr'), "Graph should have edge_attr"
        assert hasattr(graph, 'edge_type'), "Graph should have edge_type"

        # Check label
        assert isinstance(label, torch.Tensor)
        assert label.shape == (1,), f"Label shape should be (1,), got {label.shape}"
        assert 0.0 <= label.item() <= 1.0, "Label should be confidence in [0, 1]"

    def test_batch_loading(self, small_kg_dataset):
        """Test that multiple samples can be loaded."""
        batch_size = 5
        samples = [small_kg_dataset[i] for i in range(batch_size)]

        assert len(samples) == batch_size
        for graph, label in samples:
            assert graph.num_nodes > 0
            assert graph.edge_index.size(1) > 0

    def test_statistics(self, small_kg_dataset):
        """Test dataset statistics computation."""
        stats = small_kg_dataset.get_statistics()

        assert 'num_triples' in stats
        assert 'num_entities' in stats
        assert 'num_predicates' in stats
        assert 'avg_confidence' in stats
        assert 'level_distribution' in stats

        assert stats['num_triples'] == 500
        assert stats['num_entities'] > 0
        assert stats['num_predicates'] >= 15
        assert 0.0 <= stats['avg_confidence'] <= 1.0
        assert 1 in stats['level_distribution']
        assert 2 in stats['level_distribution']


class TestCaching:
    """Test dataset caching functionality."""

    def test_cache_creation(self, test_data_dir):
        """Test that cache files are created."""
        dataset = KnowledgeGraphTripleDataset(
            root=str(test_data_dir / "cache_test"),
            split='train',
            num_entities=50,
            num_triples=200,
            seed=42
        )

        cache_file = test_data_dir / "cache_test" / "processed" / "train_triples.pt"
        assert cache_file.exists(), "Cache file should be created"

    def test_cache_loading(self, test_data_dir):
        """Test that cached data is loaded correctly."""
        root = str(test_data_dir / "cache_test2")

        # Create initial dataset
        dataset1 = KnowledgeGraphTripleDataset(
            root=root,
            split='train',
            num_entities=50,
            num_triples=200,
            seed=42
        )
        triples1 = dataset1.triples.copy()

        # Load from cache
        dataset2 = KnowledgeGraphTripleDataset(
            root=root,
            split='train',
            num_entities=50,
            num_triples=200,
            seed=99  # Different seed shouldn't matter - should load from cache
        )
        triples2 = dataset2.triples

        # Should be identical (loaded from cache)
        assert len(triples1) == len(triples2)
        # First triple should be the same
        assert triples1[0].subject == triples2[0].subject
        assert triples1[0].predicate == triples2[0].predicate
        assert triples1[0].object == triples2[0].object


class TestReproducibility:
    """Test reproducibility with seeds."""

    def test_seed_reproducibility(self, test_data_dir):
        """Test that same seed produces same triples."""
        dataset1 = KnowledgeGraphTripleDataset(
            root=str(test_data_dir / "seed1"),
            split='train',
            num_entities=100,
            num_triples=500,
            seed=42
        )

        dataset2 = KnowledgeGraphTripleDataset(
            root=str(test_data_dir / "seed2"),
            split='train',
            num_entities=100,
            num_triples=500,
            seed=42
        )

        # Should generate same triples
        assert len(dataset1.triples) == len(dataset2.triples)

        # Check first 10 triples match
        for i in range(10):
            t1 = dataset1.triples[i]
            t2 = dataset2.triples[i]
            assert t1.subject == t2.subject
            assert t1.predicate == t2.predicate
            assert t1.object == t2.object
            assert abs(t1.confidence - t2.confidence) < 1e-6

    def test_different_seeds_differ(self, test_data_dir):
        """Test that different seeds produce different triples."""
        dataset1 = KnowledgeGraphTripleDataset(
            root=str(test_data_dir / "diff_seed1"),
            split='train',
            num_entities=100,
            num_triples=500,
            seed=42
        )

        dataset2 = KnowledgeGraphTripleDataset(
            root=str(test_data_dir / "diff_seed2"),
            split='train',
            num_entities=100,
            num_triples=500,
            seed=123
        )

        # Should generate different triples
        different_count = 0
        for i in range(min(50, len(dataset1.triples))):
            t1 = dataset1.triples[i]
            t2 = dataset2.triples[i]
            if t1.subject != t2.subject or t1.predicate != t2.predicate:
                different_count += 1

        assert different_count > 0, "Different seeds should produce different triples"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
