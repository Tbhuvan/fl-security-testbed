"""
attacks.py — Byzantine attack strategies for FL security research.

Attack taxonomy (threat model: malicious clients with full local control):

1. LabelFlipAttack      — flip labels during local training (data poisoning)
   Ref: Tolpegin et al., "Data Poisoning Attacks Against FL" arXiv:2004.10020

2. GradientScalingAttack — scale gradients by a large factor (model poisoning)
   Ref: Bagdasaryan et al., "How To Backdoor FL" arXiv:1807.00459

3. SignFlipAttack        — negate all gradient values (Byzantine)
   Ref: Blanchard et al., "Machine Learning with Adversaries" NeurIPS 2017

4. BackdoorAttack        — inject a pixel trigger pattern during training
   Ref: Xie et al., "DBA: Distributed Backdoor Attacks" ICLR 2020

5. IPMAttack            — Inner Product Manipulation (optimization-aware)
   Ref: Xie et al., "Fall of Empires" arXiv:1911.12053

All attacks conform to the AttackStrategy protocol:
    attack.poison_updates(updates: List[Tensor]) -> List[Tensor]

Usage:
    attack = LabelFlipAttack(num_classes=10, source_label=0, target_label=1)
    malicious_loader = attack.wrap_dataloader(client_loader)
"""

from __future__ import annotations

import copy
import torch
import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Protocol / base class
# ---------------------------------------------------------------------------

class AttackStrategy(ABC):
    """Base class for all Byzantine attack strategies."""

    @abstractmethod
    def poison_updates(
        self,
        updates: List[List[torch.Tensor]],
        global_model_params: Optional[List[torch.Tensor]] = None,
    ) -> List[List[torch.Tensor]]:
        """Modify a list of client gradient updates to inject attack.

        Args:
            updates: List of per-client parameter update lists (deltas, not weights).
            global_model_params: Current global model weights (needed by some attacks).

        Returns:
            Modified updates (same structure, some replaced with malicious values).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# 1. Label Flip Attack
# ---------------------------------------------------------------------------

class LabelFlipDataset(Dataset):
    """Wraps a dataset and flips source_label → target_label."""

    def __init__(self, dataset: Dataset, source_label: int, target_label: int) -> None:
        if source_label == target_label:
            raise ValueError("source_label and target_label must differ")
        self.dataset = dataset
        self.source_label = source_label
        self.target_label = target_label

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple:
        x, y = self.dataset[idx]
        if y == self.source_label:
            y = self.target_label
        return x, y


class LabelFlipAttack(AttackStrategy):
    """Data poisoning via label flipping.

    Attacker flips source_label → target_label in their local dataset.
    Effective at causing targeted misclassification at inference.

    Args:
        source_label: Original label to flip from.
        target_label: Label to flip to.
    """

    def __init__(self, source_label: int = 0, target_label: int = 1) -> None:
        self.source_label = source_label
        self.target_label = target_label

    def wrap_dataset(self, dataset: Dataset) -> Dataset:
        """Return a poisoned version of the dataset."""
        return LabelFlipDataset(dataset, self.source_label, self.target_label)

    def poison_updates(
        self,
        updates: List[List[torch.Tensor]],
        global_model_params: Optional[List[torch.Tensor]] = None,
    ) -> List[List[torch.Tensor]]:
        # Label flip is a data-level attack; gradient poisoning handled at training time.
        # This method is a no-op — use wrap_dataset on malicious clients instead.
        return updates

    def __repr__(self) -> str:
        return f"LabelFlipAttack(source={self.source_label}, target={self.target_label})"


# ---------------------------------------------------------------------------
# 2. Gradient Scaling Attack (model poisoning)
# ---------------------------------------------------------------------------

class GradientScalingAttack(AttackStrategy):
    """Model poisoning via gradient amplification.

    Multiplies malicious client updates by `scale_factor`.
    With large scale_factor, even 1 attacker can dominate FedAvg aggregation.

    Args:
        scale_factor: Multiplier applied to malicious updates (default: 10.0).
        num_attackers: How many clients in `updates` are malicious (from the end).
    """

    def __init__(self, scale_factor: float = 10.0, num_attackers: int = 1) -> None:
        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be > 0, got {scale_factor}")
        self.scale_factor = scale_factor
        self.num_attackers = num_attackers

    def poison_updates(
        self,
        updates: List[List[torch.Tensor]],
        global_model_params: Optional[List[torch.Tensor]] = None,
    ) -> List[List[torch.Tensor]]:
        poisoned = list(updates)
        for i in range(len(updates) - self.num_attackers, len(updates)):
            poisoned[i] = [p * self.scale_factor for p in updates[i]]
        return poisoned

    def __repr__(self) -> str:
        return f"GradientScalingAttack(scale={self.scale_factor}, attackers={self.num_attackers})"


# ---------------------------------------------------------------------------
# 3. Sign Flip Attack (Byzantine)
# ---------------------------------------------------------------------------

class SignFlipAttack(AttackStrategy):
    """Byzantine attack: negate all parameter updates from malicious clients.

    Pushes aggregated model in the opposite direction of convergence.
    Simple but defeated by most norm-clipping defenses.

    Ref: Blanchard et al., NeurIPS 2017 — original Byzantine FL paper.

    Args:
        num_attackers: Number of malicious clients (from the end of updates list).
    """

    def __init__(self, num_attackers: int = 1) -> None:
        self.num_attackers = num_attackers

    def poison_updates(
        self,
        updates: List[List[torch.Tensor]],
        global_model_params: Optional[List[torch.Tensor]] = None,
    ) -> List[List[torch.Tensor]]:
        poisoned = list(updates)
        for i in range(len(updates) - self.num_attackers, len(updates)):
            poisoned[i] = [-p for p in updates[i]]
        return poisoned

    def __repr__(self) -> str:
        return f"SignFlipAttack(attackers={self.num_attackers})"


# ---------------------------------------------------------------------------
# 4. Backdoor Attack (trigger injection)
# ---------------------------------------------------------------------------

class BackdoorDataset(Dataset):
    """Dataset wrapper that injects a trigger pattern into malicious samples.

    Trigger: a fixed 4×4 white patch in the bottom-right corner.
    All triggered samples are relabeled to `target_label`.

    Args:
        dataset: Original dataset.
        target_label: Label to assign to triggered samples.
        poison_fraction: Fraction of samples to inject the trigger into.
        trigger_size: Size of the square trigger patch (default: 4).
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        target_label: int = 0,
        poison_fraction: float = 0.2,
        trigger_size: int = 4,
        seed: int = 42,
    ) -> None:
        if not 0 < poison_fraction <= 1.0:
            raise ValueError(f"poison_fraction must be in (0, 1], got {poison_fraction}")
        self.dataset = dataset
        self.target_label = target_label
        self.trigger_size = trigger_size
        rng = np.random.default_rng(seed)
        n = len(dataset)  # type: ignore[arg-type]
        self.poisoned_indices = set(
            rng.choice(n, size=int(n * poison_fraction), replace=False).tolist()
        )

    def _inject_trigger(self, x: torch.Tensor) -> torch.Tensor:
        """Inject bottom-right corner white square trigger."""
        x = x.clone()
        ts = self.trigger_size
        x[..., -ts:, -ts:] = 1.0  # works for grayscale (C=1) and RGB (C=3)
        return x

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple:
        x, y = self.dataset[idx]
        if idx in self.poisoned_indices:
            x = self._inject_trigger(x)
            y = self.target_label
        return x, y


class BackdoorAttack(AttackStrategy):
    """Backdoor attack via trigger injection.

    Malicious clients train on backdoored data. At inference,
    any sample with the trigger pattern is misclassified to target_label.

    Args:
        target_label: Target class for the backdoor.
        poison_fraction: Fraction of attacker's data that gets the trigger.
        num_attackers: Number of malicious clients.
    """

    def __init__(
        self,
        target_label: int = 0,
        poison_fraction: float = 0.2,
        num_attackers: int = 1,
    ) -> None:
        self.target_label = target_label
        self.poison_fraction = poison_fraction
        self.num_attackers = num_attackers

    def wrap_dataset(self, dataset: Dataset, seed: int = 42) -> Dataset:
        """Return backdoored version of the dataset for malicious clients."""
        return BackdoorDataset(dataset, self.target_label, self.poison_fraction, seed=seed)

    def poison_updates(
        self,
        updates: List[List[torch.Tensor]],
        global_model_params: Optional[List[torch.Tensor]] = None,
    ) -> List[List[torch.Tensor]]:
        # Data-level attack — poison_updates is a no-op; use wrap_dataset at training.
        return updates

    def __repr__(self) -> str:
        return (
            f"BackdoorAttack(target={self.target_label}, "
            f"fraction={self.poison_fraction}, attackers={self.num_attackers})"
        )


# ---------------------------------------------------------------------------
# 5. IPM Attack — Inner Product Manipulation
# ---------------------------------------------------------------------------

class IPMAttack(AttackStrategy):
    """Inner Product Manipulation attack (optimization-aware Byzantine).

    Crafts malicious updates that pass norm-based filters but still
    push the global model toward a target direction.

    Attack vector: u_malicious = -epsilon * (sum of benign updates)
    This maximally aligns against the aggregation direction.

    Ref: Xie et al., "Fall of Empires: Breaking Byzantine-Tolerant SGD"
         arXiv:1911.12053

    Args:
        epsilon: Attack magnitude scaling factor (default: 1.0).
        num_attackers: Number of malicious clients.
    """

    def __init__(self, epsilon: float = 1.0, num_attackers: int = 1) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self.epsilon = epsilon
        self.num_attackers = num_attackers

    def poison_updates(
        self,
        updates: List[List[torch.Tensor]],
        global_model_params: Optional[List[torch.Tensor]] = None,
    ) -> List[List[torch.Tensor]]:
        num_benign = len(updates) - self.num_attackers
        if num_benign <= 0:
            raise ValueError("No benign clients to compute IPM direction from")

        benign_updates = updates[:num_benign]

        # Compute the mean of benign updates per parameter
        mean_update = [
            torch.stack([b[i] for b in benign_updates]).mean(dim=0)
            for i in range(len(benign_updates[0]))
        ]

        # Malicious update = -epsilon * mean_benign
        malicious = [-self.epsilon * m for m in mean_update]

        poisoned = list(updates)
        for i in range(num_benign, len(updates)):
            poisoned[i] = malicious

        return poisoned

    def __repr__(self) -> str:
        return f"IPMAttack(epsilon={self.epsilon}, attackers={self.num_attackers})"


# ---------------------------------------------------------------------------
# Attack registry
# ---------------------------------------------------------------------------

ATTACK_REGISTRY = {
    "none": None,
    "label_flip": LabelFlipAttack,
    "gradient_scaling": GradientScalingAttack,
    "sign_flip": SignFlipAttack,
    "backdoor": BackdoorAttack,
    "ipm": IPMAttack,
}


def get_attack(name: str, **kwargs) -> Optional[AttackStrategy]:
    """Factory: return an attack instance by name.

    Args:
        name: One of 'none', 'label_flip', 'gradient_scaling',
              'sign_flip', 'backdoor', 'ipm'.
        **kwargs: Passed to the attack constructor.

    Returns:
        AttackStrategy instance, or None for 'none'.
    """
    key = name.lower()
    if key not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack '{name}'. Choose from: {list(ATTACK_REGISTRY)}")
    cls = ATTACK_REGISTRY[key]
    return cls(**kwargs) if cls is not None else None
