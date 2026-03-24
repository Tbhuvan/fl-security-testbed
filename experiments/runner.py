"""
Experiment runner — orchestrates Flower simulation for one ExperimentConfig.

Wires together: dataset → clients (honest + malicious) → server → results.
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import flwr as fl
import numpy as np
import torch

from attacks.registry import get_attack_fn
from clients.client import build_clients
from clients.dataset import load_dataset, noniid_partition
from experiments.config import ExperimentConfig
from server.server import FLSecurityStrategy

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed all RNGs for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(config: ExperimentConfig, results_dir: Path = Path("experiments/results")) -> dict:
    """Run a single FL experiment and return results.

    Args:
        config:      Experiment configuration.
        results_dir: Directory to write per-experiment JSON results.

    Returns:
        Results dict with config + per-round metrics.
    """
    seed_everything(config.seed)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{config.experiment_name}.json"

    log.info("=" * 60)
    log.info("Starting experiment: %s", config.experiment_name)
    log.info("  Aggregation: %s | Attack: %s | Byzantine: %.0f%%",
             config.aggregation, config.attack_type, config.byzantine_fraction * 100)
    log.info("=" * 60)

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_dataset = load_dataset(config.dataset, train=True)
    test_dataset = load_dataset(config.dataset, train=False)

    client_datasets = noniid_partition(
        train_dataset,
        num_clients=config.num_clients,
        num_classes_per_client=2,
        seed=config.seed,
    )

    # ── Attack function ───────────────────────────────────────────────────────
    attack_fn = get_attack_fn(
        config.attack_type,
        scale=config.attack_scale,
        seed=config.seed,
    )

    # ── Clients ───────────────────────────────────────────────────────────────
    clients, malicious_ids = build_clients(
        num_clients=config.num_clients,
        byzantine_fraction=config.byzantine_fraction,
        datasets=client_datasets,
        attack_fn=attack_fn,
        model_name=config.model_name,
        seed=config.seed,
    )

    # ── Strategy ──────────────────────────────────────────────────────────────
    strategy = FLSecurityStrategy(config=config, results_path=results_path)

    # ── Flower simulation ─────────────────────────────────────────────────────
    start_time = time.time()

    fl.simulation.start_simulation(
        client_fn=lambda cid: clients[int(cid)],
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    elapsed = time.time() - start_time

    # ── Final results ─────────────────────────────────────────────────────────
    final_results = {
        "experiment": config.experiment_name,
        "config": config.to_dict(),
        "malicious_client_ids": malicious_ids,
        "elapsed_seconds": round(elapsed, 2),
        "rounds": strategy.results,
    }

    if strategy.results:
        final_round = strategy.results[-1]
        final_results["final_accuracy"] = final_round.get("accuracy", 0.0)
        final_results["final_loss"] = final_round.get("loss", 0.0)

    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    log.info(
        "Experiment complete: %s | Final accuracy: %.4f | Time: %.1fs",
        config.experiment_name,
        final_results.get("final_accuracy", 0.0),
        elapsed,
    )

    return final_results


def run_sweep(configs: list[ExperimentConfig], results_dir: Path = Path("experiments/results")) -> list[dict]:
    """Run multiple experiments in sequence and return all results.

    Args:
        configs:     List of experiment configurations.
        results_dir: Directory to write results.

    Returns:
        List of result dicts.
    """
    all_results = []
    for i, config in enumerate(configs):
        log.info("Sweep: %d/%d — %s", i + 1, len(configs), config.experiment_name)
        try:
            result = run_experiment(config, results_dir)
            all_results.append(result)
        except Exception as exc:
            log.error("Experiment %s failed: %s", config.experiment_name, exc)
            all_results.append({"experiment": config.experiment_name, "error": str(exc)})

    # Write combined results
    combined_path = Path(results_dir) / "sweep_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    log.info("Sweep complete. %d/%d succeeded.", sum(1 for r in all_results if "error" not in r), len(configs))
    return all_results
