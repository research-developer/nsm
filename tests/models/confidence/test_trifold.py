import torch

from nsm.models.confidence.base import verify_semiring_properties
from nsm.models.confidence.trifold import (
    TriFoldSemiring,
    as_trifold,
    split_trifold,
    Phi_min,
    Phi_mean,
    Phi_logsumexp,
    Psi,
)


def test_trifold_helpers_round_trip():
    subject = torch.tensor([0.1, 0.2])
    predicate = torch.tensor([0.3, 0.4])
    obj = torch.tensor([0.5, 0.6])
    center = torch.tensor([0.7, 0.8])

    packed = as_trifold(subject, predicate, obj, center)
    unpacked = split_trifold(packed)

    assert torch.allclose(unpacked[0], subject)
    assert torch.allclose(unpacked[1], predicate)
    assert torch.allclose(unpacked[2], obj)
    assert torch.allclose(unpacked[3], center)


def test_trifold_semiring_combine_adds_componentwise():
    semiring = TriFoldSemiring()
    confidences = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]]
    )

    combined = semiring.combine(confidences)
    expected = torch.tensor([0.5, 0.7, 0.9, 1.1])

    assert torch.allclose(combined, expected)


def test_trifold_semiring_aggregate_max_componentwise():
    semiring = TriFoldSemiring()
    confidences = torch.tensor(
        [[0.1, 0.4, 0.3, 0.2], [0.6, 0.5, 0.7, 0.8], [0.2, 0.3, 0.4, 0.9]]
    )

    aggregated = semiring.aggregate(confidences)
    expected = torch.tensor([0.6, 0.5, 0.7, 0.9])

    assert torch.allclose(aggregated, expected)


def test_phi_operators_update_center_channel():
    trifold = as_trifold(
        torch.tensor([0.0, 0.5]),
        torch.tensor([0.1, 0.6]),
        torch.tensor([-0.2, 0.2]),
        torch.tensor([0.3, 0.4]),
    )

    phi_min = Phi_min(trifold)
    phi_mean = Phi_mean(trifold)
    phi_logsumexp = Phi_logsumexp(trifold)

    stacked = torch.stack(split_trifold(trifold)[:3], dim=0)
    expected_min = stacked.min(dim=0).values
    expected_mean = stacked.mean(dim=0)
    expected_logsumexp = torch.logsumexp(stacked, dim=0)

    assert torch.allclose(split_trifold(phi_min)[3], expected_min)
    assert torch.allclose(split_trifold(phi_mean)[3], expected_mean)
    assert torch.allclose(split_trifold(phi_logsumexp)[3], expected_logsumexp)


def test_psi_broadcasts_center_channel():
    trifold = as_trifold(
        torch.tensor([0.1, 0.2]),
        torch.tensor([0.3, 0.4]),
        torch.tensor([0.5, 0.6]),
        torch.tensor([0.7, 0.8]),
    )

    broadcast = Psi(trifold)
    subject, predicate, obj, center = split_trifold(broadcast)

    assert torch.allclose(subject, center)
    assert torch.allclose(predicate, center)
    assert torch.allclose(obj, center)


def test_trifold_semiring_compatible_with_property_checks():
    semiring = TriFoldSemiring()
    test_values = torch.randn(5, 4)

    results = verify_semiring_properties(semiring, test_values=test_values)

    assert all(results.values()), f"Failed properties: {results}"
