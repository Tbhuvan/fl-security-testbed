"""Attack function registry — maps attack names to callables."""
from __future__ import annotations

from typing import Callable

import numpy as np

from attacks.byzantine import (
    random_noise_attack,
    sign_flip_attack,
    gradient_scaling_attack,
    zero_gradient_attack,
)


def get_attack_fn(
    name: str,
    scale: float = 10.0,
    dataset=None,
    seed: int = 42,
) -> Callable[[list[np.ndarray]], list[np.ndarray]] | None:
    """Return an attack function by name.

    Args:
        name:    Attack name. 'none' returns None (honest client).
        scale:   Magnitude parameter for noise/scaling attacks.
        dataset: Required for 'label_flip' attack only.
        seed:    RNG seed.

    Returns:
        Callable or None.
    """
    if name == "none":
        return None

    if name == "random_noise":
        return lambda params: random_noise_attack(params, scale=scale, seed=seed)

    if name == "sign_flip":
        return lambda params: sign_flip_attack(params, scale=scale)

    if name == "gradient_scaling":
        return lambda params: gradient_scaling_attack(params, scale=scale)

    if name == "zero_gradient":
        return lambda params: zero_gradient_attack(params)

    if name == "label_flip":
        if dataset is None:
            raise ValueError("label_flip attack requires a dataset argument")
        from attacks.label_flip import label_flip_attack
        return lambda params: label_flip_attack(params, dataset=dataset, seed=seed)

    raise ValueError(
        f"Unknown attack '{name}'. "
        "Choose from: none, random_noise, sign_flip, gradient_scaling, zero_gradient, label_flip"
    )
