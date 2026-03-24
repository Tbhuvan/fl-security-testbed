"""
Federated Learning server with pluggable aggregation strategies.

Orchestrates training rounds via Flower (flwr), applies the chosen
aggregation / defense, and logs results per round.

Reference: arXiv:1602.05629 (McMahan et al., FedAvg)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import flwr as fl
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from experiments.config import ExperimentConfig
from server.aggregation import get_aggregation_fn, GradientList

log = logging.getLogger(__name__)


class FLSecurityStrategy(Strategy):
    """Custom Flower strategy that wraps pluggable aggregation + logging.

    Supports all aggregation functions in server.aggregation:
    FedAvg, Krum, Multi-Krum, Trimmed Mean, Median, FLAME.
    """

    def __init__(self, config: ExperimentConfig, results_path: Path) -> None:
        super().__init__()
        self.config = config
        self.results_path = results_path
        self.results: list[dict] = []

        byzantine_n = int(config.num_clients * config.byzantine_fraction)
        self.agg_fn = get_aggregation_fn(
            config.aggregation,
            f=byzantine_n,
            trim_fraction=config.trim_fraction,
            epsilon=config.flame_epsilon,
        )
        log.info(
            "Strategy: %s | Byzantine fraction: %.1f (%d/%d clients) | Attack: %s",
            config.aggregation,
            config.byzantine_fraction,
            byzantine_n,
            config.num_clients,
            config.attack_type,
        )

    # ── Flower interface ──────────────────────────────────────────────────────

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return None  # clients initialise their own models

    def configure_fit(self, server_round, parameters, client_manager):
        config = {
            "round": server_round,
            "local_epochs": self.config.local_epochs,
            "batch_size": self.config.local_batch_size,
            "learning_rate": self.config.learning_rate,
        }
        sample_size = max(1, int(self.config.num_clients * self.config.fraction_fit))
        clients = client_manager.sample(num_clients=sample_size)
        return [(c, fl.common.FitIns(parameters, config)) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        # Extract gradient arrays from results
        client_gradients: GradientList = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        # Apply chosen aggregation / defense
        aggregated = self.agg_fn(client_gradients)
        aggregated_params = ndarrays_to_parameters(aggregated)

        metrics = {
            "round": server_round,
            "num_clients": len(results),
        }
        log.info("Round %d: aggregated %d client updates", server_round, len(results))
        return aggregated_params, metrics

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.config.fraction_evaluate == 0.0:
            return []
        sample_size = max(1, int(self.config.num_clients * self.config.fraction_evaluate))
        clients = client_manager.sample(num_clients=sample_size)
        return [(c, fl.common.EvaluateIns(parameters, {"round": server_round})) for c in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        losses = [evaluate_res.loss for _, evaluate_res in results]
        accuracies = [
            evaluate_res.metrics.get("accuracy", 0.0)
            for _, evaluate_res in results
        ]

        avg_loss = float(np.mean(losses))
        avg_acc = float(np.mean(accuracies))

        log.info(
            "Round %d — loss: %.4f | accuracy: %.4f", server_round, avg_loss, avg_acc
        )

        self.results.append({
            "round": server_round,
            "loss": avg_loss,
            "accuracy": avg_acc,
            "num_clients": len(results),
        })
        self._save_results()

        return avg_loss, {"accuracy": avg_acc}

    def evaluate(self, server_round, parameters):
        return None  # server-side evaluation handled via clients

    # ── Internal ──────────────────────────────────────────────────────────────

    def _save_results(self) -> None:
        output = {
            "experiment": self.config.experiment_name,
            "config": self.config.to_dict(),
            "rounds": self.results,
        }
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump(output, f, indent=2)
