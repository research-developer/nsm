"""Tests for confidence propagation base infrastructure."""

import pytest
import torch
from nsm.models.confidence import (
    BaseSemiring,
    ProductSemiring,
    MinMaxSemiring,
    TemperatureScheduler
)
from nsm.models.confidence.base import verify_semiring_properties, ConfidenceMetrics


class TestProductSemiring:
    def test_combine_product(self):
        semiring = ProductSemiring()
        conf = torch.tensor([0.9, 0.8, 0.7])
        result = semiring.combine(conf)
        expected = 0.9 * 0.8 * 0.7
        assert torch.allclose(result, torch.tensor(expected), atol=1e-5)

    def test_aggregate_probabilistic_sum(self):
        semiring = ProductSemiring()
        conf = torch.tensor([0.7, 0.8, 0.6])
        result = semiring.aggregate(conf)
        # 1 - (1-0.7)*(1-0.8)*(1-0.6) = 1 - 0.024 = 0.976
        expected = 1 - (0.3 * 0.2 * 0.4)
        assert torch.allclose(result, torch.tensor(expected), atol=1e-5)


class TestMinMaxSemiring:
    def test_combine_minimum(self):
        semiring = MinMaxSemiring(use_soft=False)
        conf = torch.tensor([0.9, 0.7, 0.8])
        result = semiring.combine(conf)
        assert result == 0.7

    def test_aggregate_maximum(self):
        semiring = MinMaxSemiring(use_soft=False)
        conf = torch.tensor([0.7, 0.8, 0.6])
        result = semiring.aggregate(conf)
        assert result == 0.8


class TestTemperatureScheduler:
    def test_exponential_decay(self):
        scheduler = TemperatureScheduler(initial_temp=1.0, final_temp=0.3, decay_rate=0.9)
        
        temps = [scheduler.step() for _ in range(10)]
        
        # Should decay
        assert temps[0] > temps[-1]
        # Should not go below final
        assert all(t >= 0.3 for t in temps)

    def test_reset(self):
        scheduler = TemperatureScheduler()
        scheduler.step()
        scheduler.step()
        scheduler.reset()
        assert scheduler.get_temperature() == 1.0
        assert scheduler.epoch == 0


class TestSemiringProperties:
    def test_verify_product_properties(self):
        semiring = ProductSemiring()
        results = verify_semiring_properties(semiring)
        assert results['combine_identity']
        assert results['aggregate_commutative']
        assert results['output_range_valid']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
