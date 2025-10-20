"""
Tests for Causal Reasoning Dataset

Validates:
- Dataset generation and structure
- Causal chain relationships
- Confounder presence and impact
- Counterfactual scenario pairs
- Label generation
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from nsm.data.causal_dataset import CausalTripleDataset
from nsm.data.triple import SemanticTriple


@pytest.fixture
def temp_dir():
    """Create temporary directory for test datasets."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def small_dataset(temp_dir):
    """Create small causal dataset for testing."""
    return CausalTripleDataset(
        root=temp_dir,
        split='train',
        num_scenarios=10,
        num_treatments=4,
        num_symptoms=4,
        num_confounders=3,
        confound_prob=0.5,
        seed=42
    )


@pytest.fixture
def large_dataset(temp_dir):
    """Create larger causal dataset for statistical tests."""
    return CausalTripleDataset(
        root=temp_dir,
        split='train',
        num_scenarios=100,
        num_treatments=8,
        num_symptoms=8,
        num_confounders=6,
        confound_prob=0.3,
        seed=42
    )


class TestCausalDatasetGeneration:
    """Test dataset generation and basic properties."""

    def test_dataset_creation(self, small_dataset):
        """Test dataset initializes correctly."""
        assert isinstance(small_dataset, CausalTripleDataset)
        assert small_dataset.num_scenarios == 10
        assert len(small_dataset.scenarios) == 10
        assert len(small_dataset.triples) > 0

    def test_scenario_structure(self, small_dataset):
        """Test each scenario has required components."""
        for scenario in small_dataset.scenarios:
            assert 'patient' in scenario
            assert 'treatment' in scenario
            assert 'symptom' in scenario
            assert 'effect' in scenario
            assert 'confounder' in scenario  # May be None
            assert 'triple_indices' in scenario
            assert len(scenario['triple_indices']) > 0

    def test_triple_levels(self, small_dataset):
        """Test triples are properly distributed across levels."""
        level1_count = sum(1 for t in small_dataset.triples if t.level == 1)
        level2_count = sum(1 for t in small_dataset.triples if t.level == 2)

        # Should have both levels
        assert level1_count > 0
        assert level2_count > 0

        # Level 2 should have causal mechanism triples
        level2_predicates = {
            t.predicate for t in small_dataset.triples if t.level == 2
        }
        assert 'causes' in level2_predicates
        assert 'treats' in level2_predicates

    def test_confidence_ranges(self, small_dataset):
        """Test all confidence scores are in valid range."""
        for triple in small_dataset.triples:
            assert 0.0 <= triple.confidence <= 1.0

    def test_reproducibility(self, temp_dir):
        """Test same seed produces same dataset."""
        dataset1 = CausalTripleDataset(
            root=temp_dir + '/d1',
            split='train',
            num_scenarios=5,
            seed=123
        )

        dataset2 = CausalTripleDataset(
            root=temp_dir + '/d2',
            split='train',
            num_scenarios=5,
            seed=123
        )

        assert len(dataset1.triples) == len(dataset2.triples)

        # Check first few triples match
        for t1, t2 in zip(dataset1.triples[:10], dataset2.triples[:10]):
            assert t1.subject == t2.subject
            assert t1.predicate == t2.predicate
            assert t1.object == t2.object
            assert t1.level == t2.level
            assert abs(t1.confidence - t2.confidence) < 1e-5


class TestCausalChains:
    """Test causal relationship structures."""

    def test_treatment_observation_present(self, small_dataset):
        """Test each scenario has treatment observation."""
        for scenario in small_dataset.scenarios:
            treatment = scenario['treatment']
            patient = scenario['patient']

            # Find treatment triple
            treatment_triples = [
                t for t in small_dataset.triples
                if t.subject == treatment and t.predicate == "taken_by"
                and t.object == patient
            ]

            assert len(treatment_triples) >= 1

    def test_symptom_observation_present(self, small_dataset):
        """Test each scenario has symptom observation."""
        for scenario in small_dataset.scenarios:
            patient = scenario['patient']
            symptom = scenario['symptom']

            # Find symptom triple
            symptom_triples = [
                t for t in small_dataset.triples
                if t.subject == patient and t.predicate == "has_symptom"
                and t.object == symptom
            ]

            assert len(symptom_triples) >= 1

    def test_outcome_present(self, small_dataset):
        """Test each scenario has an outcome."""
        for scenario in small_dataset.scenarios:
            patient = scenario['patient']
            symptom = scenario['symptom']

            # Find outcome triple
            outcome_triples = [
                t for t in small_dataset.triples
                if t.subject == patient and t.object == symptom
                and t.predicate in ["symptom_reduced", "symptom_persists"]
            ]

            assert len(outcome_triples) >= 1

    def test_causal_mechanism_present(self, small_dataset):
        """Test causal mechanisms are represented."""
        for scenario in small_dataset.scenarios:
            treatment = scenario['treatment']
            effect = scenario['effect']

            # Find causal triple
            causal_triples = [
                t for t in small_dataset.triples
                if t.subject == treatment and t.predicate == "causes"
                and t.object == effect and t.level == 2
            ]

            assert len(causal_triples) >= 1

    def test_mechanism_of_action(self, small_dataset):
        """Test treatment mechanism (effect treats symptom)."""
        for scenario in small_dataset.scenarios:
            effect = scenario['effect']
            symptom = scenario['symptom']

            # Find mechanism triple
            mechanism_triples = [
                t for t in small_dataset.triples
                if t.subject == effect and t.predicate == "treats"
                and t.object == symptom and t.level == 2
            ]

            assert len(mechanism_triples) >= 1


class TestConfounders:
    """Test confounder handling."""

    def test_confounder_probability(self, large_dataset):
        """Test confounders appear at expected rate."""
        confounded_scenarios = sum(
            1 for s in large_dataset.scenarios if s['confounder'] is not None
        )

        # With confound_prob=0.3 and 100 scenarios, expect ~30
        # Allow for randomness: 15-45
        assert 15 <= confounded_scenarios <= 45

    def test_confounder_triples(self, large_dataset):
        """Test confounded scenarios have confounder triples."""
        for scenario in large_dataset.scenarios:
            if scenario['confounder'] is not None:
                patient = scenario['patient']
                confounder = scenario['confounder']

                # Find confounder observation (Level 1)
                obs_triples = [
                    t for t in large_dataset.triples
                    if t.subject == patient and t.predicate == "has_condition"
                    and t.object == confounder and t.level == 1
                ]

                assert len(obs_triples) >= 1

                # Find confounding relationship (Level 2)
                conf_triples = [
                    t for t in large_dataset.triples
                    if t.subject == confounder and t.predicate == "confounds"
                    and t.level == 2
                ]

                assert len(conf_triples) >= 1

    def test_get_scenarios_with_confounder(self, large_dataset):
        """Test retrieval of scenarios by confounder."""
        # Find a confounder that exists
        confounders = [
            s['confounder'] for s in large_dataset.scenarios
            if s['confounder'] is not None
        ]

        if confounders:
            test_confounder = confounders[0]
            scenarios = large_dataset.get_scenarios_with_confounder(test_confounder)

            assert len(scenarios) > 0
            for idx in scenarios:
                assert large_dataset.scenarios[idx]['confounder'] == test_confounder


class TestCounterfactuals:
    """Test counterfactual reasoning support."""

    def test_get_counterfactual_pairs(self, large_dataset):
        """Test counterfactual pair generation."""
        pairs = large_dataset.get_counterfactual_pairs()

        # Should find some pairs in 100 scenarios
        assert len(pairs) > 0

        # Verify pair properties
        for idx1, idx2 in pairs[:5]:  # Check first 5
            s1 = large_dataset.scenarios[idx1]
            s2 = large_dataset.scenarios[idx2]

            # Same symptom
            assert s1['symptom'] == s2['symptom']

            # Different treatments
            assert s1['treatment'] != s2['treatment']

    def test_get_scenarios_by_treatment(self, large_dataset):
        """Test retrieval of scenarios by treatment."""
        # Get a treatment that exists
        treatment = large_dataset.scenarios[0]['treatment']

        scenarios = large_dataset.get_scenarios_by_treatment(treatment)

        assert len(scenarios) > 0
        for idx in scenarios:
            assert large_dataset.scenarios[idx]['treatment'] == treatment


class TestLabels:
    """Test label generation."""

    def test_label_format(self, small_dataset):
        """Test labels are in correct format."""
        for i in range(len(small_dataset)):
            _, label = small_dataset[i]

            assert isinstance(label, torch.Tensor)
            assert label.dtype == torch.long
            assert label.shape == (1,)
            assert label.item() in [0, 1]

    def test_label_consistency(self, small_dataset):
        """Test labels match outcomes."""
        for scenario in small_dataset.scenarios:
            # Get outcome triple
            outcome_triples = [
                small_dataset.triples[i]
                for i in scenario['triple_indices']
                if small_dataset.triples[i].predicate in ["symptom_reduced", "symptom_persists"]
            ]

            if outcome_triples:
                outcome_triple = outcome_triples[0]
                triple_idx = small_dataset.triples.index(outcome_triple)
                _, label = small_dataset[triple_idx]

                if outcome_triple.predicate == "symptom_reduced":
                    assert label.item() == 1
                elif outcome_triple.predicate == "symptom_persists":
                    assert label.item() == 0


class TestGraphConstruction:
    """Test PyG graph construction."""

    def test_getitem_returns_graph(self, small_dataset):
        """Test __getitem__ returns valid PyG graph."""
        graph, label = small_dataset[0]

        assert hasattr(graph, 'x')  # Node features
        assert hasattr(graph, 'edge_index')  # Edge connectivity
        assert hasattr(graph, 'edge_attr')  # Edge features
        assert isinstance(label, torch.Tensor)

    def test_graph_dimensions(self, small_dataset):
        """Test graph has expected dimensions."""
        graph, _ = small_dataset[0]

        # Check dimensions
        num_nodes = graph.x.shape[0]
        num_edges = graph.edge_index.shape[1]

        assert num_nodes > 0
        assert num_edges > 0
        assert graph.edge_index.shape[0] == 2  # [2, num_edges]

    def test_scenario_graph_construction(self, small_dataset):
        """Test constructing graph from scenario triples."""
        scenario = small_dataset.scenarios[0]
        triple_indices = scenario['triple_indices']

        graph = small_dataset.get_graph_for_triples(triple_indices)

        assert hasattr(graph, 'x')
        assert hasattr(graph, 'edge_index')
        assert graph.x.shape[0] > 0  # Has nodes


class TestDatasetStatistics:
    """Test dataset statistics and properties."""

    def test_get_statistics(self, small_dataset):
        """Test statistics computation."""
        stats = small_dataset.get_statistics()

        assert 'num_triples' in stats
        assert 'num_entities' in stats
        assert 'num_predicates' in stats
        assert 'avg_confidence' in stats
        assert 'level_distribution' in stats

        # Validate values
        assert stats['num_triples'] == len(small_dataset.triples)
        assert stats['num_entities'] > 0
        assert stats['num_predicates'] > 0
        assert 0.0 <= stats['avg_confidence'] <= 1.0
        assert 1 in stats['level_distribution']
        assert 2 in stats['level_distribution']

    def test_vocabulary_populated(self, small_dataset):
        """Test vocabulary is populated after graph construction."""
        # Build a graph from full scenario to populate vocabulary
        scenario = small_dataset.scenarios[0]
        graph = small_dataset.get_graph_for_triples(scenario['triple_indices'])

        vocab = small_dataset.vocabulary

        # Should have entities after graph construction
        assert vocab.num_entities > 0

        # Should have predicates after graph construction
        assert vocab.num_predicates > 0

        # Check for expected predicate types
        predicates = [
            vocab.get_predicate_token(i)
            for i in range(vocab.num_predicates)
        ]

        assert any('taken_by' in p for p in predicates)
        assert any('has_symptom' in p for p in predicates)
        assert any('causes' in p for p in predicates)

    def test_caching(self, temp_dir):
        """Test dataset caching mechanism."""
        dataset1 = CausalTripleDataset(
            root=temp_dir,
            split='train',
            num_scenarios=5,
            seed=42
        )

        initial_len = len(dataset1.triples)

        # Create another instance with same parameters
        dataset2 = CausalTripleDataset(
            root=temp_dir,
            split='train',
            num_scenarios=5,
            seed=42
        )

        # Should load from cache
        assert len(dataset2.triples) == initial_len

        # Check cache file exists
        cache_path = Path(temp_dir) / "processed" / "train_triples.pt"
        assert cache_path.exists()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_confound_probability(self, temp_dir):
        """Test dataset with no confounders."""
        dataset = CausalTripleDataset(
            root=temp_dir,
            split='train',
            num_scenarios=10,
            confound_prob=0.0,
            seed=42
        )

        confounded = sum(
            1 for s in dataset.scenarios if s['confounder'] is not None
        )

        assert confounded == 0

    def test_full_confound_probability(self, temp_dir):
        """Test dataset with all confounders."""
        dataset = CausalTripleDataset(
            root=temp_dir,
            split='train',
            num_scenarios=10,
            confound_prob=1.0,
            seed=42
        )

        confounded = sum(
            1 for s in dataset.scenarios if s['confounder'] is not None
        )

        assert confounded == 10

    def test_invalid_scenario_index(self, small_dataset):
        """Test error on invalid scenario index."""
        with pytest.raises(IndexError):
            small_dataset.get_scenario(999)

    def test_repr(self, small_dataset):
        """Test string representation."""
        repr_str = repr(small_dataset)

        assert 'CausalTripleDataset' in repr_str
        assert 'train' in repr_str
        assert str(small_dataset.num_scenarios) in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
