"""
Byzantine attack implementations for FL Security Testbed.

Each attack is a pure function:
    attack_fn(parameters: list[np.ndarray]) -> list[np.ndarray]

References:
  - Random noise / sign flip:    arXiv:1703.02757 (Blanchard et al.)
  - Gradient scaling:            arXiv:1811.12470 (Fang et al.)
  - Label flipping:              arXiv:1811.12470
  - Backdoor attack:             arXiv:1807.00459 (Bagdasaryan et al.)
"""
from attacks.byzantine import (
    random_noise_attack,
    sign_flip_attack,
    gradient_scaling_attack,
    zero_gradient_attack,
)
from attacks.label_flip import label_flip_attack
from attacks.registry import get_attack_fn

__all__ = [
    "random_noise_attack",
    "sign_flip_attack",
    "gradient_scaling_attack",
    "zero_gradient_attack",
    "label_flip_attack",
    "get_attack_fn",
]
