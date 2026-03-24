"""
Dataset loading + non-IID partitioning for FL experiments.

Non-IID is realistic: real FL clients (hospitals, phones) have
skewed local data distributions.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_dataset(name: str, train: bool = True) -> datasets.VisionDataset:
    """Download and return a torchvision dataset.

    Args:
        name:  'mnist' or 'cifar10'
        train: True for training split.

    Returns:
        Dataset object.
    """
    root = "./data"
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        return datasets.MNIST(root=root, train=train, download=True, transform=transform)
    elif name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.261)),
        ])
        return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose: mnist, cifar10")


def iid_partition(dataset, num_clients: int, seed: int = 42) -> list[Subset]:
    """Partition dataset IID (uniform random) across clients.

    Args:
        dataset:     Full dataset.
        num_clients: Number of FL clients.
        seed:        Random seed for reproducibility.

    Returns:
        List of Subset objects, one per client.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    splits = np.array_split(indices, num_clients)
    return [Subset(dataset, split.tolist()) for split in splits]


def noniid_partition(
    dataset,
    num_clients: int,
    num_classes_per_client: int = 2,
    seed: int = 42,
) -> list[Subset]:
    """Partition dataset non-IID: each client gets only N classes.

    This is the standard non-IID setup from McMahan et al. (arXiv:1602.05629).

    Args:
        dataset:                 Full dataset.
        num_clients:             Number of FL clients.
        num_classes_per_client:  How many distinct classes each client gets.
        seed:                    Random seed.

    Returns:
        List of Subset objects.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))

    # Group indices by class
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}
    for c in class_indices:
        rng.shuffle(class_indices[c])

    # Assign classes to clients (with overlap)
    all_classes = list(range(num_classes))
    client_subsets = []
    for client_id in range(num_clients):
        assigned_classes = rng.choice(all_classes, size=num_classes_per_client, replace=False)
        client_indices = []
        for c in assigned_classes:
            n_samples = len(class_indices[c]) // num_clients
            n_samples = max(n_samples, 10)  # at least 10 samples per class
            client_indices.extend(class_indices[c][:n_samples])
        rng.shuffle(client_indices)
        client_subsets.append(Subset(dataset, client_indices))

    return client_subsets


def get_dataloader(subset: Subset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """Wrap a Subset in a DataLoader."""
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
