"""Tests for aggregation / defense strategies."""
import numpy as np
import pytest

from server.aggregation import (
    fedavg,
    krum,
    multi_krum,
    trimmed_mean,
    coordinate_median,
    flame,
    get_aggregation_fn,
)


def make_gradients(n: int, shape=(10,), seed: int = 0) -> list:
    """Generate n random gradient sets."""
    rng = np.random.default_rng(seed)
    return [[rng.standard_normal(shape).astype(np.float32)] for _ in range(n)]


def honest_gradients(n: int, value: float = 1.0, shape=(10,)) -> list:
    """Generate n identical honest gradients."""
    return [[np.full(shape, value, dtype=np.float32)] for _ in range(n)]


def poisoned_gradients(n_honest: int, n_byzantine: int, poison_scale: float = 100.0, shape=(10,)) -> list:
    """Mix of honest (value=1.0) and byzantine (large noise) gradients."""
    honest = honest_gradients(n_honest, value=1.0, shape=shape)
    byzantine = [[np.full(shape, poison_scale, dtype=np.float32)] for _ in range(n_byzantine)]
    return honest + byzantine


class TestFedAvg:
    def test_basic_average(self):
        grads = [
            [np.array([1.0, 2.0], dtype=np.float32)],
            [np.array([3.0, 4.0], dtype=np.float32)],
        ]
        result = fedavg(grads)
        np.testing.assert_array_almost_equal(result[0], [2.0, 3.0])

    def test_uniform_weights(self):
        grads = honest_gradients(5, value=2.0)
        result = fedavg(grads)
        np.testing.assert_array_almost_equal(result[0], [2.0] * 10)

    def test_custom_weights(self):
        grads = [
            [np.array([0.0], dtype=np.float32)],
            [np.array([4.0], dtype=np.float32)],
        ]
        result = fedavg(grads, weights=[3.0, 1.0])
        np.testing.assert_array_almost_equal(result[0], [1.0])  # (0*3 + 4*1)/4 = 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fedavg([])

    def test_poisoned_fedavg_skewed(self):
        """FedAvg should be skewed by Byzantine clients — this validates the attack works."""
        grads = poisoned_gradients(7, 3, poison_scale=50.0)
        result = fedavg(grads)
        # Result should be pulled toward poison value
        assert result[0].mean() > 1.0


class TestKrum:
    def test_selects_honest_under_attack(self):
        """Krum should select an honest gradient when f=3 Byzantine."""
        honest_val = 1.0
        grads = poisoned_gradients(7, 3, poison_scale=1000.0)
        result = krum(grads, f=3)
        # Result should be close to honest value 1.0
        assert abs(result[0].mean() - honest_val) < 0.5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            krum([])

    def test_f_too_large_fallback(self):
        """Krum with f >= n/2 should log a warning and not crash."""
        grads = make_gradients(4)
        result = krum(grads, f=3)  # f=3 >= 4/2=2 → should fallback
        assert result is not None
        assert len(result) == len(grads[0])

    def test_returns_single_gradient_shape(self):
        grads = make_gradients(5, shape=(20,))
        result = krum(grads, f=1)
        assert len(result) == 1
        assert result[0].shape == (20,)


class TestTrimmedMean:
    def test_removes_extremes(self):
        grads = [
            [np.array([100.0], dtype=np.float32)],  # outlier high
            [np.array([1.0], dtype=np.float32)],
            [np.array([1.0], dtype=np.float32)],
            [np.array([1.0], dtype=np.float32)],
            [np.array([-100.0], dtype=np.float32)],  # outlier low
        ]
        result = trimmed_mean(grads, trim_fraction=0.2)
        # Should trim one from each end → average of 3 middle values = 1.0
        np.testing.assert_array_almost_equal(result[0], [1.0])

    def test_zero_trim_equals_fedavg(self):
        grads = make_gradients(6, shape=(8,), seed=7)
        tm = trimmed_mean(grads, trim_fraction=0.0)
        fa = fedavg(grads)
        np.testing.assert_array_almost_equal(tm[0], fa[0], decimal=5)

    def test_invalid_trim_fraction_raises(self):
        grads = make_gradients(4)
        with pytest.raises(ValueError):
            trimmed_mean(grads, trim_fraction=0.6)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            trimmed_mean([])


class TestCoordinateMedian:
    def test_basic_median(self):
        grads = [
            [np.array([1.0, 2.0], dtype=np.float32)],
            [np.array([3.0, 4.0], dtype=np.float32)],
            [np.array([5.0, 6.0], dtype=np.float32)],
        ]
        result = coordinate_median(grads)
        np.testing.assert_array_almost_equal(result[0], [3.0, 4.0])

    def test_robust_to_outlier(self):
        grads = honest_gradients(5, value=1.0)
        grads.append([np.array([1000.0] * 10, dtype=np.float32)])
        result = coordinate_median(grads)
        assert result[0].mean() < 2.0, "Median should not be dominated by outlier"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            coordinate_median([])


class TestFlame:
    def test_returns_gradient_shape(self):
        grads = make_gradients(8, shape=(16,))
        result = flame(grads)
        assert len(result) == 1
        assert result[0].shape == (16,)

    def test_filters_outliers(self):
        """FLAME should keep the dominant cluster and filter poisoned outliers."""
        grads = poisoned_gradients(6, 2, poison_scale=500.0, shape=(10,))
        result = flame(grads)
        # Result should be closer to 1.0 (honest) than 500.0 (poison)
        assert result[0].mean() < 50.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            flame([])


class TestAggregationRegistry:
    def test_all_strategies_callable(self):
        grads = make_gradients(5, shape=(8,))
        for name in ["fedavg", "krum", "trimmed_mean", "median", "flame"]:
            fn = get_aggregation_fn(name)
            result = fn(grads)
            assert result is not None

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            get_aggregation_fn("invalid_strategy")
