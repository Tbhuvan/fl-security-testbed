"""
models.py — PyTorch model definitions for FL experiments.

MNIST  : SimpleCNN  (2 conv + 2 FC)  ~93% accuracy baseline
CIFAR10: ResNet8    (lightweight ResNet variant)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List


# ---------------------------------------------------------------------------
# MNIST baseline — lightweight CNN
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """2-layer CNN for MNIST. Matches Flower quickstart architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# CIFAR-10 baseline — ResNet-8
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block(x) + x)


class ResNet8(nn.Module):
    """Lightweight ResNet-8 for CIFAR-10 FL experiments.

    Params: ~272K — fast enough for per-round simulation on 780M iGPU.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_model(name: str) -> nn.Module:
    """Factory: return a fresh model instance by name."""
    _registry = {
        "simplecnn": SimpleCNN,
        "resnet8": ResNet8,
    }
    key = name.lower()
    if key not in _registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(_registry)}")
    return _registry[key]()


def get_parameters(model: nn.Module) -> List[torch.Tensor]:
    """Extract model parameters as a list of detached CPU tensors."""
    return [p.detach().cpu().clone() for p in model.parameters()]


def set_parameters(model: nn.Module, parameters: List[torch.Tensor]) -> None:
    """Load a parameter list (from get_parameters) back into a model."""
    if len(parameters) != len(list(model.parameters())):
        raise ValueError("Parameter count mismatch — wrong model or corrupted update?")
    with torch.no_grad():
        for param, new_val in zip(model.parameters(), parameters):
            param.copy_(new_val.to(param.device))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
