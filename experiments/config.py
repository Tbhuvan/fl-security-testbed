"""Experiment configuration system — all hyperparameters live here."""
from dataclasses import dataclass, field, asdict
from typing import Literal
import json
from pathlib import Path


@dataclass
class ExperimentConfig:
    # FL setup
    num_clients: int = 10
    num_rounds: int = 20
    fraction_fit: float = 1.0          # fraction of clients selected per round
    fraction_evaluate: float = 1.0

    # Byzantine attack
    byzantine_fraction: float = 0.2    # fraction of clients that are malicious
    attack_type: Literal[
        "none", "random_noise", "sign_flip", "gradient_scaling", "label_flip"
    ] = "random_noise"
    attack_scale: float = 10.0         # magnitude for noise/scaling attacks

    # Aggregation / defense
    aggregation: Literal[
        "fedavg", "krum", "trimmed_mean", "median", "flame"
    ] = "fedavg"
    trim_fraction: float = 0.1         # fraction to trim each side (trimmed_mean)
    flame_epsilon: float = 3000.0      # FLAME noise bound

    # Model + training
    model_name: Literal["cnn", "mlp"] = "cnn"
    dataset: Literal["mnist", "cifar10"] = "mnist"
    local_epochs: int = 1
    local_batch_size: int = 32
    learning_rate: float = 0.01
    seed: int = 42

    # Experiment metadata
    experiment_name: str = "default"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# ── Predefined experiment sweeps ─────────────────────────────────────────────

BASELINE = ExperimentConfig(
    experiment_name="baseline_fedavg_no_attack",
    byzantine_fraction=0.0,
    attack_type="none",
    aggregation="fedavg",
)

BYZANTINE_SWEEP = [
    ExperimentConfig(
        experiment_name=f"fedavg_byzantine_{frac:.1f}",
        byzantine_fraction=frac,
        attack_type="random_noise",
        aggregation="fedavg",
    )
    for frac in [0.0, 0.1, 0.2, 0.3, 0.4]
]

DEFENSE_COMPARISON = [
    ExperimentConfig(
        experiment_name=f"{defense}_byzantine_0.3",
        byzantine_fraction=0.3,
        attack_type="random_noise",
        aggregation=defense,
    )
    for defense in ["fedavg", "krum", "trimmed_mean", "median"]
]

ATTACK_TYPE_SWEEP = [
    ExperimentConfig(
        experiment_name=f"fedavg_{attack}_0.2",
        byzantine_fraction=0.2,
        attack_type=attack,
        aggregation="fedavg",
    )
    for attack in ["random_noise", "sign_flip", "gradient_scaling", "label_flip"]
]
