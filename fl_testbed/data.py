"""
data.py — Dataset loading and FL data partitioning.

Supports:
    - IID partitioning   : uniform random split across clients
    - Non-IID (Dirichlet): label distribution skew via Dirichlet(alpha)
      - alpha=0.1  → extreme heterogeneity (each client sees 1-2 classes)
      - alpha=1.0  → moderate heterogeneity
      - alpha=100  → near-IID

References:
    Measuring the Effects of Non-Identical Data Distribution for
    Federated Visual Classification (Hsieh et al., 2020) arXiv:1909.06335
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------

def load_mnist(data_root: str = "./data") -> Tuple[Dataset, Dataset]:
    """Return (train_dataset, test_dataset) for MNIST."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_root, train=True, download=True, transform=tf)
    test = datasets.MNIST(data_root, train=False, download=True, transform=tf)
    return train, test


def load_cifar10(data_root: str = "./data") -> Tuple[Dataset, Dataset]:
    """Return (train_dataset, test_dataset) for CIFAR-10."""
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10(data_root, train=True, download=True, transform=tf_train)
    test = datasets.CIFAR10(data_root, train=False, download=True, transform=tf_test)
    return train, test


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def partition_iid(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> List[List[int]]:
    """Split dataset indices uniformly across clients."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()
    shard_size = len(indices) // num_clients
    return [indices[i * shard_size:(i + 1) * shard_size] for i in range(num_clients)]


def partition_dirichlet(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[List[int]]:
    """Non-IID partition via Dirichlet distribution over class labels.

    Lower alpha → more heterogeneous distribution.
    Raises ValueError if alpha <= 0 or num_clients < 1.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")

    rng = np.random.default_rng(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = int(labels.max()) + 1
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c_idx in class_indices:
        rng.shuffle(c_idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        # Ensure proportions sum correctly and handle edge rounding
        splits = (proportions * len(c_idx)).astype(int)
        splits[-1] = len(c_idx) - splits[:-1].sum()  # fix rounding
        start = 0
        for client_id, count in enumerate(splits):
            client_indices[client_id].extend(c_idx[start:start + count].tolist())
            start += count

    return client_indices


def make_dataloader(
    dataset: Dataset,
    indices: List[int],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for a given client's index partition."""
    if not indices:
        raise ValueError("Client has empty partition — increase dataset size or reduce num_clients")
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_dataset(name: str, data_root: str = "./data") -> Tuple[Dataset, Dataset]:
    """Factory for datasets."""
    _registry = {
        "mnist": load_mnist,
        "cifar10": load_cifar10,
    }
    key = name.lower()
    if key not in _registry:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(_registry)}")
    return _registry[key](data_root)


def summarize_partition(partition: List[List[int]], dataset: Dataset) -> dict:
    """Return per-client class distribution stats for logging."""
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    stats = {}
    for cid, idxs in enumerate(partition):
        client_labels = labels[idxs]
        unique, counts = np.unique(client_labels, return_counts=True)
        stats[f"client_{cid}"] = {
            "total_samples": len(idxs),
            "class_distribution": {int(k): int(v) for k, v in zip(unique, counts)},
        }
    return stats
