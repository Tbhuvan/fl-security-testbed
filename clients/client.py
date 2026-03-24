"""
Flower FL clients — honest and malicious variants.

HonestClient:   trains normally, returns true gradients.
MaliciousClient: wraps HonestClient and applies an attack before returning.

Attack injection is in /attacks — clients never directly implement attacks.
"""
from __future__ import annotations

import logging
from typing import Callable

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from clients.model import get_model, get_parameters, set_parameters
from clients.dataset import get_dataloader

log = logging.getLogger(__name__)


class HonestClient(fl.client.Client):
    """Standard federated learning client that trains honestly.

    Args:
        client_id:  Unique identifier for this client.
        dataset:    Local dataset (Subset).
        config:     Experiment configuration dict.
    """

    def __init__(self, client_id: int, dataset, model_name: str = "cnn") -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.model_name = model_name
        self.device = torch.device("cpu")  # AMD 780M runs via Ollama; PyTorch on CPU

    def get_parameters(self, ins) -> fl.common.GetParametersRes:
        model = get_model(self.model_name)
        params = get_parameters(model)
        return fl.common.GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(params),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Train locally and return updated parameters."""
        params = parameters_to_ndarrays(ins.parameters)
        config = ins.config

        model = get_model(self.model_name).to(self.device)
        set_parameters(model, params)

        updated_params = self._train(
            model,
            epochs=int(config.get("local_epochs", 1)),
            batch_size=int(config.get("batch_size", 32)),
            lr=float(config.get("learning_rate", 0.01)),
        )

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(updated_params),
            num_examples=len(self.dataset),
            metrics={"client_id": self.client_id},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate local model and return loss + accuracy."""
        params = parameters_to_ndarrays(ins.parameters)
        model = get_model(self.model_name).to(self.device)
        set_parameters(model, params)

        loss, accuracy = self._evaluate(model)
        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(loss),
            num_examples=len(self.dataset),
            metrics={"accuracy": float(accuracy), "client_id": self.client_id},
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _train(
        self, model: nn.Module, epochs: int, batch_size: int, lr: float
    ) -> list[np.ndarray]:
        """Run local SGD and return updated parameters."""
        loader = get_dataloader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        return get_parameters(model)

    def _evaluate(self, model: nn.Module) -> tuple[float, float]:
        """Compute loss and accuracy on local data."""
        loader = get_dataloader(self.dataset, batch_size=64, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                total_loss += criterion(outputs, targets).item() * len(targets)
                correct += (outputs.argmax(dim=1) == targets).sum().item()
                total += len(targets)

        return total_loss / max(total, 1), correct / max(total, 1)


class MaliciousClient(HonestClient):
    """FL client that applies a Byzantine attack before returning parameters.

    The attack_fn receives the honest parameters and returns poisoned parameters.
    Attack implementations live in /attacks — never directly in this class.

    Args:
        client_id:  Unique identifier.
        dataset:    Local dataset.
        attack_fn:  Callable[[list[np.ndarray]], list[np.ndarray]] from attacks/.
        model_name: Model architecture.
    """

    def __init__(
        self,
        client_id: int,
        dataset,
        attack_fn: Callable[[list[np.ndarray]], list[np.ndarray]],
        model_name: str = "cnn",
    ) -> None:
        super().__init__(client_id, dataset, model_name)
        self.attack_fn = attack_fn

    def fit(self, ins: FitIns) -> FitRes:
        """Train honestly, then apply the attack to the parameters."""
        honest_result = super().fit(ins)
        honest_params = parameters_to_ndarrays(honest_result.parameters)

        poisoned_params = self.attack_fn(honest_params)

        log.debug(
            "MaliciousClient %d applied attack (norm change: %.4f)",
            self.client_id,
            sum(np.linalg.norm(p - h) for p, h in zip(poisoned_params, honest_params)),
        )

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(poisoned_params),
            num_examples=len(self.dataset),
            metrics={"client_id": self.client_id, "is_malicious": 1},
        )


def build_clients(
    num_clients: int,
    byzantine_fraction: float,
    datasets: list,
    attack_fn: Callable | None,
    model_name: str = "cnn",
    seed: int = 42,
) -> tuple[list[HonestClient], list[int]]:
    """Build a mix of honest and malicious clients.

    Args:
        num_clients:       Total number of clients.
        byzantine_fraction: Fraction that are malicious.
        datasets:          Per-client dataset subsets.
        attack_fn:         Attack function for malicious clients. None = all honest.
        model_name:        Model architecture name.
        seed:              RNG seed for client assignment.

    Returns:
        (clients, malicious_indices) tuple.
    """
    rng = np.random.default_rng(seed)
    num_byzantine = int(num_clients * byzantine_fraction)
    malicious_ids = set(rng.choice(num_clients, size=num_byzantine, replace=False).tolist())

    clients = []
    for i in range(num_clients):
        if i in malicious_ids and attack_fn is not None:
            clients.append(MaliciousClient(i, datasets[i], attack_fn, model_name))
        else:
            clients.append(HonestClient(i, datasets[i], model_name))

    log.info(
        "Built %d clients: %d honest, %d malicious (IDs: %s)",
        num_clients,
        num_clients - len(malicious_ids),
        len(malicious_ids),
        sorted(malicious_ids),
    )
    return clients, sorted(malicious_ids)
