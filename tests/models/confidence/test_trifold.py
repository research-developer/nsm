import torch

from nsm.models.confidence import TriFoldSemiring, TriFoldReasoner


def test_trifold_semiring_combine_adds_channels():
    semiring = TriFoldSemiring()
    path = torch.tensor([
        [0.0, -1.0, -2.0, -0.5],
        [0.2, -0.3, -0.1, -0.4],
    ])
    combined = semiring.combine(path, dim=0)
    expected = torch.tensor([0.2, -1.3, -2.1, -0.9])
    assert torch.allclose(combined, expected, atol=1e-5)


def test_trifold_semiring_aggregate_max_componentwise():
    semiring = TriFoldSemiring()
    candidates = torch.tensor([
        [0.1, -0.2, -0.5, -1.0],
        [-0.3, 0.4, -0.1, -0.2],
        [0.0, -0.1, 0.2, -0.3],
    ])
    aggregated = semiring.aggregate(candidates, dim=0)
    expected = torch.tensor([0.1, 0.4, 0.2, -0.2])
    assert torch.allclose(aggregated, expected, atol=1e-5)


def test_trifold_reasoner_fold_unfold_cycle():
    states = torch.tensor([
        [0.1, 0.2, 0.3, 0.0],
        [0.3, 0.1, 0.0, -0.5],
    ])
    reasoner = TriFoldReasoner(alpha=1.0, beta=0.5, reduction='min')
    output = reasoner(states, iterations=1)

    expected_states = torch.tensor([
        [0.15, 0.25, 0.35, 0.1],
        [0.05, -0.15, -0.25, -0.5],
    ])
    assert torch.allclose(output.states, expected_states, atol=1e-5)
    assert torch.allclose(output.aggregated, torch.tensor([0.15, 0.25, 0.35, 0.1]), atol=1e-5)
    assert torch.isclose(output.center, torch.tensor(0.1), atol=1e-5)
    assert torch.allclose(output.loops, torch.tensor([0.15, 0.25, 0.35]), atol=1e-5)
    assert torch.allclose(output.fold_history.squeeze(0), torch.tensor([0.1, 0.0]), atol=1e-5)


def test_trifold_reasoner_batch_aggregation():
    states = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.2, -0.1, -0.2],
        [0.1, 0.3, 0.4, 0.5],
    ])
    batch = torch.tensor([0, 0, 1])
    reasoner = TriFoldReasoner(alpha=0.0, beta=0.0)
    output = reasoner(states, batch=batch, iterations=0)

    assert output.fold_history.shape[0] == 0
    assert output.aggregated.shape == (2, 4)
    expected = torch.tensor([
        [0.5, 0.2, 0.0, 0.0],
        [0.1, 0.3, 0.4, 0.5],
    ])
    assert torch.allclose(output.aggregated, expected, atol=1e-5)
    assert torch.allclose(output.center, torch.tensor([0.0, 0.5]), atol=1e-5)
    assert torch.allclose(output.loops, expected[:, :3], atol=1e-5)
