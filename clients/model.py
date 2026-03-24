"""
Neural network models for FL experiments.

Kept small intentionally — the research is about FL security,
not model architecture. Models must run on CPU (AMD 780M via PyTorch CPU).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """Simple CNN for MNIST. Fast training on CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.dropout(x)))
        return self.fc2(x)


class SimpleMLP(nn.Module):
    """MLP baseline — faster than CNN, lower accuracy ceiling."""

    def __init__(self, input_dim: int = 784, hidden: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


def get_model(name: str) -> nn.Module:
    """Instantiate a model by name.

    Args:
        name: 'cnn' or 'mlp'

    Returns:
        Initialised PyTorch model.
    """
    models = {
        "cnn": MNISTNet,
        "mlp": SimpleMLP,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(models)}")
    return models[name]()


def get_parameters(model: nn.Module) -> list:
    """Extract model parameters as a list of numpy arrays."""
    import numpy as np
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list) -> None:
    """Load a list of numpy arrays into a model's state dict."""
    import numpy as np
    import copy
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
