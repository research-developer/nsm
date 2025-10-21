import pytest

torch = pytest.importorskip("torch")

from nsm.models.confidence.base import verify_semiring_properties
from nsm.models.confidence.trifold import (
    TriFoldSemiring,
    trifold_tensor,
    split_trifold,
    fold_min,
    fold_mean,
    fold_logsumexp,
    Phi_logsumexp,
    Psi_add,
    Psi_replace,
    Psi_max,
    unfold,
)


def test_combine_componentwise_addition():
    semiring = TriFoldSemiring()
    scores = torch.log(torch.rand(2, 3, 4))
    result = semiring.combine(scores, dim=1)
    expected = scores.sum(dim=1)
    assert torch.allclose(result, expected)


def test_aggregate_componentwise_maximum():
    semiring = TriFoldSemiring()
    scores = torch.log(torch.rand(2, 3, 4))
    result = semiring.aggregate(scores, dim=1)
    expected = scores.max(dim=1).values
    assert torch.allclose(result, expected)


def test_fold_variants_update_center_channel():
    dtype = torch.float32
    s = torch.log(torch.tensor([0.7, 0.4], dtype=dtype))
    p = torch.log(torch.tensor([0.6, 0.5], dtype=dtype))
    o = torch.log(torch.tensor([0.3, 0.9], dtype=dtype))
    tri = trifold_tensor(s, p, o)

    folded_min = fold_min(tri)
    folded_mean = fold_mean(tri)
    folded_logsumexp = fold_logsumexp(tri)

    outer = tri[..., :3]

    assert torch.allclose(folded_min[..., 3], outer.min(dim=-1).values)
    assert torch.allclose(folded_mean[..., 3], outer.mean(dim=-1))
    assert torch.allclose(
        folded_logsumexp[..., 3], torch.logsumexp(outer, dim=-1)
    )
    # Ensure outer channels remain unchanged
    assert torch.allclose(folded_min[..., :3], outer)


def test_unfold_broadcast_modes():
    dtype = torch.float32
    s = torch.log(torch.tensor([0.5, 0.2], dtype=dtype))
    p = torch.log(torch.tensor([0.4, 0.6], dtype=dtype))
    o = torch.log(torch.tensor([0.9, 0.3], dtype=dtype))
    c = torch.log(torch.tensor([0.8, 0.7], dtype=dtype))
    tri = trifold_tensor(s, p, o, c)

    unfolded_add = unfold(tri, mode="add")
    add_expected = tri[..., :3] + c.unsqueeze(-1)
    assert torch.allclose(unfolded_add[..., :3], add_expected)

    unfolded_replace = Psi_replace(tri)
    replace_expected = c.unsqueeze(-1).expand_as(tri[..., :3])
    assert torch.allclose(unfolded_replace[..., :3], replace_expected)

    unfolded_max = Psi_max(tri)
    max_expected = torch.maximum(tri[..., :3], c.unsqueeze(-1))
    assert torch.allclose(unfolded_max[..., :3], max_expected)

    # Alias covers
    assert torch.allclose(Psi_add(tri), unfolded_add)
    assert torch.allclose(Phi_logsumexp(tri)[..., :3], tri[..., :3])


def test_verify_semiring_properties_compatibility():
    semiring = TriFoldSemiring()
    results = verify_semiring_properties(semiring)
    assert all(results.values())


def test_split_round_trip():
    dtype = torch.float32
    s = torch.log(torch.tensor([0.6, 0.8], dtype=dtype))
    p = torch.log(torch.tensor([0.5, 0.4], dtype=dtype))
    o = torch.log(torch.tensor([0.3, 0.2], dtype=dtype))
    tri = trifold_tensor(s, p, o)
    recovered = split_trifold(tri)
    for original, recon in zip((s, p, o, torch.zeros_like(s)), recovered):
        assert torch.allclose(original, recon)
