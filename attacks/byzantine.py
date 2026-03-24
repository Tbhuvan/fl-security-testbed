"""
Byzantine gradient attacks — pure functions that poison model parameters.

All attacks take honest parameters and return poisoned parameters.
Attack code is intentionally isolated here; never imported by server/.

References:
  arXiv:1703.02757 — Blanchard et al., Byzantine ML
  arXiv:1811.12470 — Fang et al., Local Model Poisoning Attacks
"""
from __future__ import annotations

import numpy as np


def random_noise_attack(
    parameters: list[np.ndarray],
    scale: float = 10.0,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Replace parameters with large random noise.

    The simplest Byzantine attack: send garbage weights.
    Effective against FedAvg at high Byzantine fractions.

    Args:
        parameters: Honest model parameters (list of arrays, one per layer).
        scale:      Noise magnitude.
        seed:       RNG seed. None = unseeded (non-reproducible).

    Returns:
        Poisoned parameters.

    Security note: does not leak any training data — purely adversarial noise.
    """
    if not parameters:
        raise ValueError("parameters list is empty")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    rng = np.random.default_rng(seed)
    return [rng.normal(0, scale, p.shape).astype(p.dtype) for p in parameters]


def sign_flip_attack(
    parameters: list[np.ndarray],
    scale: float = 1.0,
) -> list[np.ndarray]:
    """Negate and scale the gradient — sends the opposite update direction.

    Effective at slowing convergence even with a single malicious client.
    (Used in Blanchard et al., arXiv:1703.02757 evaluation)

    Args:
        parameters: Honest model parameters.
        scale:      Amplification factor (>1 increases damage).

    Returns:
        Poisoned parameters with flipped signs.
    """
    if not parameters:
        raise ValueError("parameters list is empty")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    return [-scale * p for p in parameters]


def gradient_scaling_attack(
    parameters: list[np.ndarray],
    scale: float = 100.0,
) -> list[np.ndarray]:
    """Amplify the honest gradient by a large factor.

    Dominates aggregation by inflating the gradient norm.
    Bypasses Krum if scale is large enough to shift the Krum score.
    (Fang et al., arXiv:1811.12470)

    Args:
        parameters: Honest model parameters.
        scale:      Amplification factor.

    Returns:
        Scaled parameters.
    """
    if not parameters:
        raise ValueError("parameters list is empty")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    return [scale * p for p in parameters]


def zero_gradient_attack(
    parameters: list[np.ndarray],
) -> list[np.ndarray]:
    """Send all-zero parameters — simulates a non-participating client.

    Least aggressive attack. Tests whether defenses filter passive adversaries.

    Args:
        parameters: Honest model parameters (used for shape only).

    Returns:
        Zero arrays matching input shapes.
    """
    if not parameters:
        raise ValueError("parameters list is empty")

    return [np.zeros_like(p) for p in parameters]


def inner_product_manipulation_attack(
    parameters: list[np.ndarray],
    target_parameters: list[np.ndarray],
    scale: float = 1.0,
) -> list[np.ndarray]:
    """IPM attack: craft a gradient that maximises inner product with target.

    More sophisticated than noise — designed to steer the model toward
    a specific malicious direction while remaining close to honest updates.
    (Fang et al., arXiv:1811.12470)

    Args:
        parameters:        Honest parameters.
        target_parameters: Direction to steer towards (e.g., label-flip model).
        scale:             Attack magnitude.

    Returns:
        Poisoned parameters.
    """
    if not parameters:
        raise ValueError("parameters list is empty")
    if len(parameters) != len(target_parameters):
        raise ValueError("parameters and target_parameters must have same length")

    return [scale * t for t in target_parameters]
