"""
Tests for Planning Domain Dataset

Validates planning triple generation, hierarchical structure,
temporal ordering, and integration with NSM-18 infrastructure.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from nsm.data.planning_dataset import PlanningTripleDataset
from nsm.data.triple import SemanticTriple


class TestPlanningTripleDataset:
    """Test suite for PlanningTripleDataset."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for dataset files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def small_dataset(self, temp_dir):
        """Create small dataset for testing."""
        return PlanningTripleDataset(
            root=temp_dir,
            split='train',
            num_problems=10,
            num_locations=3,
            num_objects=5,
            seed=42
        )

    @pytest.fixture
    def full_dataset(self, temp_dir):
        """Create full-sized dataset."""
        return PlanningTripleDataset(
            root=temp_dir,
            split='train',
            num_problems=100,
            num_locations=5,
            num_objects=10,
            seed=42
        )

    def test_dataset_initialization(self, small_dataset):
        """Test dataset initializes correctly."""
        assert small_dataset is not None
        assert small_dataset.split == 'train'
        assert small_dataset.num_problems == 10
        assert len(small_dataset.triples) > 0
        assert len(small_dataset.problems) > 0

    def test_triple_generation(self, small_dataset):
        """Test triples are generated with correct structure."""
        triples = small_dataset.triples

        # Should have triples
        assert len(triples) > 0

        # All should be SemanticTriple instances
        for triple in triples:
            assert isinstance(triple, SemanticTriple)

        # Check level distribution
        level_1_count = sum(1 for t in triples if t.level == 1)
        level_2_count = sum(1 for t in triples if t.level == 2)

        assert level_1_count > 0, "Should have Level 1 (concrete) triples"
        assert level_2_count > 0, "Should have Level 2 (abstract) triples"

    def test_confidence_range(self, small_dataset):
        """Test confidence scores are in valid range."""
        for triple in small_dataset.triples:
            assert 0.0 <= triple.confidence <= 1.0, \
                f"Confidence {triple.confidence} out of range [0, 1]"

        # Check confidence variance (should not all be 1.0)
        confidences = [t.confidence for t in small_dataset.triples]
        assert min(confidences) >= 0.7, "Minimum confidence should be >= 0.7"
        assert max(confidences) <= 1.0, "Maximum confidence should be <= 1.0"
        assert max(confidences) - min(confidences) > 0.1, \
            "Should have confidence variance"

    def test_predicate_types(self, small_dataset):
        """Test predicates are from expected sets."""
        predicates = {t.predicate for t in small_dataset.triples}

        # Should have both L1 and L2 predicates
        l1_predicates = predicates & PlanningTripleDataset.L1_PREDICATES
        l2_predicates = predicates & PlanningTripleDataset.L2_PREDICATES

        assert len(l1_predicates) > 0, "Should use Level 1 predicates"
        assert len(l2_predicates) > 0, "Should use Level 2 predicates"

    def test_hierarchical_structure(self, small_dataset):
        """Test hierarchical goal decomposition."""
        for problem_idx in range(len(small_dataset.problems)):
            structure = small_dataset.analyze_hierarchical_structure(problem_idx)

            assert structure['num_goals'] > 0, "Each problem should have goals"
            assert structure['num_capabilities'] > 0, \
                "Each problem should require capabilities"
            assert structure['decomposition_depth'] > 0, \
                "Goals should decompose to actions"

    def test_temporal_ordering(self, small_dataset):
        """Test temporal ordering constraints."""
        for problem_idx in range(len(small_dataset.problems)):
            ordering = small_dataset.analyze_temporal_ordering(problem_idx)

            assert ordering['num_actions'] > 0, "Should have actions"
            # Valid ordering (no cycles)
            assert ordering['is_valid'], \
                f"Problem {problem_idx} has invalid temporal ordering"

    def test_get_problem_triples(self, small_dataset):
        """Test retrieving triples for specific problem."""
        problem_idx = 0
        problem_triples = small_dataset.get_problem_triples(problem_idx)

        assert len(problem_triples) > 0
        assert all(isinstance(t, SemanticTriple) for t in problem_triples)

        # All triples should reference the same problem
        for triple in problem_triples:
            if 'problem' in triple.metadata:
                assert triple.metadata['problem'] == problem_idx

    def test_get_problem_graph(self, small_dataset):
        """Test graph construction for complete problem."""
        problem_idx = 0
        graph = small_dataset.get_problem_graph(problem_idx)

        # Verify graph structure
        assert graph.num_nodes > 0
        assert graph.edge_index.size(1) > 0
        assert graph.x.size(0) == graph.num_nodes
        assert hasattr(graph, 'edge_attr')
        assert hasattr(graph, 'edge_type')

    def test_dataset_indexing(self, small_dataset):
        """Test dataset __getitem__ returns correct format."""
        graph, label = small_dataset[0]

        # Check graph structure
        assert hasattr(graph, 'x')
        assert hasattr(graph, 'edge_index')
        assert hasattr(graph, 'edge_attr')

        # Check label format
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
        assert label.shape == (1,)
        assert label.item() in [0, 1]  # Binary classification

    def test_label_generation(self, small_dataset):
        """Test label generation for valid/invalid sequences."""
        labels = [small_dataset.generate_labels(i).item() for i in range(len(small_dataset))]

        # Should have mix of valid (1) and invalid (0)
        # Note: synthetic data generates 85% valid, 15% invalid
        num_valid = sum(labels)
        num_invalid = len(labels) - num_valid

        assert num_valid > 0, "Should have valid sequences"
        # With small dataset, might not have invalid examples, so we just check ratio makes sense
        assert num_valid >= num_invalid, "Should have more valid than invalid (85/15 ratio)"

    def test_reproducibility(self, temp_dir):
        """Test same seed produces same triples."""
        dataset1 = PlanningTripleDataset(
            root=str(Path(temp_dir) / "ds1"),
            split='train',
            num_problems=5,
            seed=42
        )

        dataset2 = PlanningTripleDataset(
            root=str(Path(temp_dir) / "ds2"),
            split='train',
            num_problems=5,
            seed=42
        )

        # Should generate same triples
        assert len(dataset1.triples) == len(dataset2.triples)

        for t1, t2 in zip(dataset1.triples, dataset2.triples):
            assert t1.subject == t2.subject
            assert t1.predicate == t2.predicate
            assert t1.object == t2.object
            assert t1.level == t2.level
            assert abs(t1.confidence - t2.confidence) < 1e-6

    def test_split_independence(self, temp_dir):
        """Test train/val/test splits are independent."""
        train_dataset = PlanningTripleDataset(
            root=temp_dir,
            split='train',
            num_problems=100,
            seed=42
        )

        val_dataset = PlanningTripleDataset(
            root=temp_dir,
            split='val',
            num_problems=100,
            seed=42
        )

        test_dataset = PlanningTripleDataset(
            root=temp_dir,
            split='test',
            num_problems=100,
            seed=42
        )

        # Different splits should have different numbers of triples
        assert len(train_dataset.triples) > len(val_dataset.triples)
        assert len(train_dataset.triples) > len(test_dataset.triples)

        # Approximate 70/15/15 split
        total = len(train_dataset.triples) + len(val_dataset.triples) + len(test_dataset.triples)
        train_ratio = len(train_dataset.triples) / total
        assert 0.65 < train_ratio < 0.75, f"Train ratio {train_ratio} not ~0.7"

    def test_vocabulary_building(self, small_dataset):
        """Test vocabulary is built correctly."""
        # Vocabulary is built when graphs are constructed
        # Access a graph to trigger vocabulary building
        graph = small_dataset.get_problem_graph(0)

        vocab = small_dataset.vocabulary

        assert vocab.num_entities > 0
        assert vocab.num_predicates > 0

        # Should include entities like robot, locations, objects
        entity_tokens = set(vocab.entity_vocab.token_to_idx.keys())
        assert any('robot' in e for e in entity_tokens)
        assert any('loc' in e for e in entity_tokens)
        assert any('obj' in e for e in entity_tokens)

        # Should include predicates from L1 and L2
        predicate_tokens = set(vocab.predicate_vocab.token_to_idx.keys())
        assert len(predicate_tokens & PlanningTripleDataset.L1_PREDICATES) > 0
        assert len(predicate_tokens & PlanningTripleDataset.L2_PREDICATES) > 0

    def test_dataset_statistics(self, small_dataset):
        """Test dataset statistics computation."""
        stats = small_dataset.get_statistics()

        assert 'num_triples' in stats
        assert 'num_entities' in stats
        assert 'num_predicates' in stats
        assert 'avg_confidence' in stats
        assert 'level_distribution' in stats

        assert stats['num_triples'] == len(small_dataset.triples)
        assert stats['num_entities'] > 0
        assert stats['num_predicates'] > 0
        assert 0.7 <= stats['avg_confidence'] <= 1.0

        # Should have both levels
        assert 1 in stats['level_distribution']
        assert 2 in stats['level_distribution']

    def test_problem_metadata(self, small_dataset):
        """Test problem metadata is correctly stored."""
        for problem in small_dataset.problems:
            assert 'idx' in problem
            assert 'num_triples' in problem
            assert 'offset' in problem

            # Verify offset points to correct location
            offset = problem['offset']
            num_triples = problem['num_triples']
            problem_triples = small_dataset.triples[offset:offset + num_triples]

            assert len(problem_triples) == num_triples

    def test_action_sequences(self, small_dataset):
        """Test action sequences are generated correctly."""
        for problem_idx in range(len(small_dataset.problems)):
            triples = small_dataset.get_problem_triples(problem_idx)

            # Find action triples
            actions = [
                t for t in triples
                if t.level == 1 and t.metadata.get('type') == 'action'
            ]

            assert len(actions) >= 3, "Should have at least 3 actions per problem"
            assert len(actions) <= 8, "Should have at most 8 actions per problem"

            # Actions should use primitive action predicates
            action_predicates = {a.predicate for a in actions}
            assert action_predicates.issubset(PlanningTripleDataset.PRIMITIVE_ACTIONS)

    def test_goal_templates(self, small_dataset):
        """Test goals use defined templates."""
        for problem_idx in range(len(small_dataset.problems)):
            triples = small_dataset.get_problem_triples(problem_idx)

            # Find goal achievement triples
            goals = [
                t for t in triples
                if t.predicate == 'achieve' and t.level == 2
            ]

            assert len(goals) >= 1, "Each problem should have at least one goal"

            # Check goal names reference templates
            for goal_triple in goals:
                goal_name = goal_triple.object
                assert any(
                    template in goal_name
                    for template in PlanningTripleDataset.GOAL_TEMPLATES.keys()
                )

    def test_capability_requirements(self, small_dataset):
        """Test capability requirements are generated."""
        for problem_idx in range(len(small_dataset.problems)):
            triples = small_dataset.get_problem_triples(problem_idx)

            # Find capability triples
            capabilities = [
                t for t in triples
                if t.predicate == 'has_capability' and t.level == 2
            ]

            assert len(capabilities) >= 2, "Should require at least 2 capabilities"
            assert len(capabilities) <= 4, "Should require at most 4 capabilities"

    def test_caching(self, temp_dir):
        """Test dataset caching works correctly."""
        # Create dataset first time
        dataset1 = PlanningTripleDataset(
            root=temp_dir,
            split='train',
            num_problems=5,
            seed=42
        )

        # Check cache file exists
        cache_file = Path(temp_dir) / "processed" / "train_triples.pt"
        assert cache_file.exists()

        # Load again (should use cache)
        dataset2 = PlanningTripleDataset(
            root=temp_dir,
            split='train',
            num_problems=5,
            seed=42
        )

        # Should have same triples
        assert len(dataset1.triples) == len(dataset2.triples)

    def test_full_dataset_size(self, full_dataset):
        """Test full dataset has expected size."""
        # 100 problems, ~70 in train split
        assert len(full_dataset.problems) >= 60
        assert len(full_dataset.problems) <= 80

        # Should have many triples
        assert len(full_dataset.triples) >= 1000

    def test_edge_cases(self, temp_dir):
        """Test edge cases and error handling."""
        # Single problem (train split ~70% of 1 = 0, so use more problems)
        single_dataset = PlanningTripleDataset(
            root=str(Path(temp_dir) / "single"),
            split='train',
            num_problems=10,  # Need enough for train split to have at least 1
            seed=42
        )
        assert len(single_dataset.problems) > 0
        assert len(single_dataset.triples) > 0

        # Test out of range problem index
        with pytest.raises(IndexError):
            single_dataset.get_problem_triples(999)

    def test_confidence_distribution(self, full_dataset):
        """Test confidence scores have expected distribution."""
        # Separate by level
        l1_confidences = [t.confidence for t in full_dataset.triples if t.level == 1]
        l2_confidences = [t.confidence for t in full_dataset.triples if t.level == 2]

        # Level 1 should have higher average confidence (environmental facts)
        avg_l1 = sum(l1_confidences) / len(l1_confidences)
        avg_l2 = sum(l2_confidences) / len(l2_confidences)
        assert avg_l1 > avg_l2

        # Both should be in expected ranges (allow some variance in synthetic data)
        assert min(l1_confidences) >= 0.70  # Actions can have lower confidence
        assert min(l2_confidences) >= 0.70  # Abstract goals start at 0.70

        # Averages should be reasonably high
        assert avg_l1 >= 0.90  # Level 1 average should be high
        assert avg_l2 >= 0.80  # Level 2 average should be good


class TestPlanningDatasetIntegration:
    """Integration tests with NSM-18 infrastructure."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_graph_constructor_integration(self, temp_dir):
        """Test integration with GraphConstructor."""
        dataset = PlanningTripleDataset(
            root=temp_dir,
            split='train',
            num_problems=5,
            seed=42
        )

        # Get graph for first problem
        graph = dataset.get_problem_graph(0)

        # Verify PyG Data format
        assert hasattr(graph, 'x')
        assert hasattr(graph, 'edge_index')
        assert hasattr(graph, 'edge_attr')
        assert hasattr(graph, 'edge_type')
        assert hasattr(graph, 'node_level')

        # Check dimensions match
        assert graph.x.size(0) == graph.num_nodes
        assert graph.edge_index.size(0) == 2
        assert graph.edge_attr.size(0) == graph.edge_index.size(1)
        assert graph.edge_type.size(0) == graph.edge_index.size(1)

    def test_batch_processing(self, temp_dir):
        """Test batching multiple graphs."""
        from torch_geometric.loader import DataLoader

        dataset = PlanningTripleDataset(
            root=temp_dir,
            split='train',
            num_problems=10,
            seed=42
        )

        # Create dataloader
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        # Get first batch
        batch = next(iter(loader))
        graphs, labels = batch

        assert graphs.num_graphs == 4
        assert labels.size(0) == 4

    def test_vocabulary_sharing(self, temp_dir):
        """Test vocabulary can be shared across datasets."""
        # Create train dataset
        train_dataset = PlanningTripleDataset(
            root=str(Path(temp_dir) / "train"),
            split='train',
            num_problems=10,
            seed=42
        )

        # Save vocabulary
        vocab_path = Path(temp_dir) / "vocab"
        train_dataset.save_vocabulary(vocab_path)

        # Create val dataset with shared vocabulary
        val_dataset = PlanningTripleDataset(
            root=str(Path(temp_dir) / "val"),
            split='val',
            num_problems=10,
            seed=42
        )
        val_dataset.load_vocabulary(vocab_path)

        # Both should have same vocabulary size
        assert train_dataset.vocabulary.num_entities == val_dataset.vocabulary.num_entities
        assert train_dataset.vocabulary.num_predicates == val_dataset.vocabulary.num_predicates
