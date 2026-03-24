"""
Label flipping attack — trains on deliberately mislabelled data.

Instead of poisoning gradients directly, the malicious client trains
on a corrupted local dataset where labels are flipped.

Reference: arXiv:1807.00459 (Bagdasaryan et al., How To Backdoor FL)
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LabelFlippedDataset(Dataset):
    """Wraps a dataset and flips labels according to a mapping.

    Args:
        dataset:      Original dataset.
        flip_map:     Dict mapping original_label -> flipped_label.
                      e.g. {0: 1, 1: 0} flips 0↔1.
        flip_fraction: Fraction of samples to flip (1.0 = all).
        seed:         RNG seed.
    """

    def __init__(
        self,
        dataset,
        flip_map: dict[int, int] | None = None,
        flip_fraction: float = 1.0,
        seed: int = 42,
    ) -> None:
        if not (0.0 <= flip_fraction <= 1.0):
            raise ValueError(f"flip_fraction must be in [0, 1], got {flip_fraction}")

        self.dataset = dataset
        self.flip_fraction = flip_fraction

        rng = np.random.default_rng(seed)
        n = len(dataset)
        self.flip_mask = rng.random(n) < flip_fraction

        # Default: shift all labels by 1 (0→1, 1→2, ..., 9→0)
        if flip_map is None:
            # Infer number of classes from dataset
            labels = [dataset[i][1] for i in range(min(100, n))]
            num_classes = max(labels) + 1
            flip_map = {c: (c + 1) % num_classes for c in range(num_classes)}

        self.flip_map = flip_map

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        if self.flip_mask[idx] and label in self.flip_map:
            label = self.flip_map[label]
        return image, label


def label_flip_attack(
    parameters: list[np.ndarray],
    dataset=None,
    flip_map: dict[int, int] | None = None,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.01,
    model_name: str = "cnn",
    seed: int = 42,
) -> list[np.ndarray]:
    """Train on label-flipped data and return poisoned parameters.

    This attack requires re-training from the given parameters on
    a corrupted dataset, unlike gradient attacks which modify params directly.

    Args:
        parameters:  Starting model parameters.
        dataset:     Client's local dataset (will be label-flipped).
        flip_map:    Label flipping mapping.
        epochs:      Local training epochs on poisoned data.
        batch_size:  Training batch size.
        lr:          Learning rate.
        model_name:  Model architecture.
        seed:        RNG seed.

    Returns:
        Poisoned parameters after training on flipped labels.

    Security note: this function re-trains the model — it requires access
    to the dataset. Never expose raw dataset samples in return values.
    """
    if dataset is None:
        raise ValueError("label_flip_attack requires a dataset to train on")

    from clients.model import get_model, get_parameters, set_parameters

    poisoned_dataset = LabelFlippedDataset(dataset, flip_map=flip_map, seed=seed)
    loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = get_model(model_name)
    set_parameters(model, parameters)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return get_parameters(model)
