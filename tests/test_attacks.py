"""Tests for Byzantine attack implementations."""
import numpy as np
import pytest

from attacks.byzantine import (
    random_noise_attack,
    sign_flip_attack,
    gradient_scaling_attack,
    zero_gradient_attack,
)
from attacks.registry import get_attack_fn


@pytest.fixture
def sample_params():
    """Sample model parameters (3 layers)."""
    rng = np.random.default_rng(42)
    return [
        rng.standard_normal((32, 1, 3, 3)).astype(np.float32),  # conv weights
        rng.standard_normal((32,)).astype(np.float32),            # conv bias
        rng.standard_normal((10, 32)).astype(np.float32),         # fc weights
    ]


class TestRandomNoiseAttack:
    def test_returns_same_shape(self, sample_params):
        poisoned = random_noise_attack(sample_params, scale=5.0, seed=0)
        assert len(poisoned) == len(sample_params)
        for p, q in zip(sample_params, poisoned):
            assert p.shape == q.shape

    def test_values_differ_from_honest(self, sample_params):
        poisoned = random_noise_attack(sample_params, scale=5.0, seed=0)
        total_diff = sum(np.linalg.norm(p - q) for p, q in zip(sample_params, poisoned))
        assert total_diff > 0.1, "Poisoned params should differ from honest"

    def test_empty_params_raises(self):
        with pytest.raises(ValueError):
            random_noise_attack([])

    def test_negative_scale_raises(self, sample_params):
        with pytest.raises(ValueError):
            random_noise_attack(sample_params, scale=-1.0)

    def test_deterministic_with_seed(self, sample_params):
        a = random_noise_attack(sample_params, scale=5.0, seed=99)
        b = random_noise_attack(sample_params, scale=5.0, seed=99)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


class TestSignFlipAttack:
    def test_negates_params(self, sample_params):
        poisoned = sign_flip_attack(sample_params, scale=1.0)
        for p, q in zip(sample_params, poisoned):
            np.testing.assert_array_almost_equal(-p, q)

    def test_scale_amplifies(self, sample_params):
        p1 = sign_flip_attack(sample_params, scale=1.0)
        p2 = sign_flip_attack(sample_params, scale=2.0)
        ratio = np.linalg.norm(p2[0]) / np.linalg.norm(p1[0])
        assert abs(ratio - 2.0) < 1e-5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sign_flip_attack([])


class TestGradientScalingAttack:
    def test_scales_params(self, sample_params):
        scale = 50.0
        poisoned = gradient_scaling_attack(sample_params, scale=scale)
        for p, q in zip(sample_params, poisoned):
            np.testing.assert_array_almost_equal(p * scale, q)

    def test_norm_increases(self, sample_params):
        original_norm = sum(np.linalg.norm(p) for p in sample_params)
        poisoned = gradient_scaling_attack(sample_params, scale=10.0)
        poisoned_norm = sum(np.linalg.norm(p) for p in poisoned)
        assert poisoned_norm > original_norm * 5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            gradient_scaling_attack([])


class TestZeroGradientAttack:
    def test_returns_zeros(self, sample_params):
        poisoned = zero_gradient_attack(sample_params)
        for q in poisoned:
            assert np.allclose(q, 0.0)

    def test_preserves_shape(self, sample_params):
        poisoned = zero_gradient_attack(sample_params)
        for p, q in zip(sample_params, poisoned):
            assert p.shape == q.shape

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            zero_gradient_attack([])


class TestAttackRegistry:
    def test_none_returns_none(self):
        assert get_attack_fn("none") is None

    def test_random_noise_callable(self, sample_params):
        fn = get_attack_fn("random_noise", scale=5.0)
        result = fn(sample_params)
        assert len(result) == len(sample_params)

    def test_sign_flip_callable(self, sample_params):
        fn = get_attack_fn("sign_flip", scale=1.0)
        result = fn(sample_params)
        assert len(result) == len(sample_params)

    def test_unknown_attack_raises(self):
        with pytest.raises(ValueError):
            get_attack_fn("invalid_attack_name")

    def test_label_flip_requires_dataset(self):
        with pytest.raises(ValueError):
            get_attack_fn("label_flip", dataset=None)
