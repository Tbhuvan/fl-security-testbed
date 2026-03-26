"""
Attack-defense matrix benchmark for gradient aggregation robustness.

Simulates Byzantine federated learning rounds using pure numpy — no Flower,
no PyTorch, no ML dependencies beyond scikit-learn (required by FLAME's
AgglomerativeClustering). All gradient generation and attack logic is
implemented inline so the script is fully self-contained.

Experiment design
-----------------
- 20 clients, gradient shape (50,)
- Honest gradient: unit vector [1, 1, ..., 1] / sqrt(50) + small Gaussian noise
- Byzantine clients replace their gradient with an attack-specific poisoned vector
- 6 defenses × 4 attacks × 4 Byzantine fractions × 3 seeds × 5 rounds
- Robustness score per trial = 1 - normalised_L2_error(aggregated, honest_mean)
  Score of 1.0 means the defense perfectly recovered the honest mean.
  Score of 0.0 means the error equals the norm of the honest mean (complete failure).
- Results saved to experiments/results/aggregation_benchmark.json
- Heatmap saved to experiments/results/aggregation_heatmap.png

References
----------
  FedAvg:          arXiv:1602.05629 (McMahan et al., 2017)
  Krum:            arXiv:1703.02757 (Blanchard et al., 2017)
  Trimmed Mean:    arXiv:1803.01498 (Yin et al., 2018)
  FLAME:           arXiv:2101.02281 (Nguyen et al., 2022)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Project-root on sys.path so `server.aggregation` resolves regardless of
# the working directory from which the script is invoked.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from server.aggregation import (  # noqa: E402
    fedavg,
    krum,
    trimmed_mean,
    coordinate_median,
    flame,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
NUM_CLIENTS: int = 20
GRAD_DIM: int = 50
NUM_ROUNDS: int = 5
SEEDS: list[int] = [42, 7, 2025]
BYZANTINE_FRACTIONS: list[float] = [0.1, 0.2, 0.3, 0.4]
HONEST_NOISE_STD: float = 0.02   # per-coordinate Gaussian noise on honest gradients
RESULTS_DIR: Path = Path(__file__).resolve().parent / "results"
JSON_PATH: Path = RESULTS_DIR / "aggregation_benchmark.json"
HEATMAP_PATH: Path = RESULTS_DIR / "aggregation_heatmap.png"

# The canonical honest gradient: unit vector in the all-ones direction
_UNIT: np.ndarray = np.ones(GRAD_DIM) / np.sqrt(GRAD_DIM)


# ===========================================================================
# Gradient generation helpers
# ===========================================================================

def _honest_gradient(rng: np.random.Generator) -> np.ndarray:
    """Return one honest client gradient: unit vector + small Gaussian noise.

    Args:
        rng: NumPy random Generator (seeded externally).

    Returns:
        1-D array of shape (GRAD_DIM,).
    """
    return _UNIT + rng.normal(0.0, HONEST_NOISE_STD, GRAD_DIM)


def _honest_mean(honest_grads: list[np.ndarray]) -> np.ndarray:
    """Compute the arithmetic mean of honest client gradients.

    Args:
        honest_grads: List of honest gradient arrays.

    Returns:
        Mean gradient array of shape (GRAD_DIM,).
    """
    return np.mean(np.stack(honest_grads, axis=0), axis=0)


# ===========================================================================
# Attack implementations (pure numpy, no external dependencies)
# ===========================================================================

def _attack_none(honest_grad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """No attack — Byzantine client sends an honest gradient (baseline).

    Args:
        honest_grad: Reference honest gradient (shape=(GRAD_DIM,)).
        rng:         Unused; kept for uniform signature.

    Returns:
        A copy of the honest gradient.
    """
    return honest_grad.copy()


def _attack_random_noise(honest_grad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Byzantine client sends large-magnitude random noise.

    Magnitude is chosen to be 10× the L2 norm of the honest gradient so the
    poisoned vector is always clearly out-of-distribution. Effective against
    FedAvg at high Byzantine fractions (arXiv:1703.02757).

    Args:
        honest_grad: Reference honest gradient.
        rng:         Seeded RNG for reproducibility.

    Returns:
        Random Gaussian vector with std = 10 × ||honest_grad||.
    """
    scale = 10.0 * float(np.linalg.norm(honest_grad))
    return rng.normal(0.0, scale, GRAD_DIM)


def _attack_sign_flip(honest_grad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Byzantine client sends the negated honest gradient, amplified 5×.

    Designed to steer aggregation in the exact wrong direction.
    (Blanchard et al., arXiv:1703.02757)

    Args:
        honest_grad: Reference honest gradient.
        rng:         Unused; kept for uniform signature.

    Returns:
        Poisoned gradient: -5 × honest_grad.
    """
    return -5.0 * honest_grad


def _attack_gradient_scale(honest_grad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Byzantine client sends the honest gradient scaled up 50×.

    Inflates gradient norm to dominate FedAvg aggregation.
    (Fang et al., arXiv:1811.12470)

    Args:
        honest_grad: Reference honest gradient.
        rng:         Unused; kept for uniform signature.

    Returns:
        Poisoned gradient: 50 × honest_grad.
    """
    return 50.0 * honest_grad


# Map attack name -> callable(honest_grad, rng) -> poisoned_grad
AttackFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]

ATTACKS: dict[str, AttackFn] = {
    "none":             _attack_none,
    "random_noise":     _attack_random_noise,
    "sign_flip":        _attack_sign_flip,
    "gradient_scale":   _attack_gradient_scale,
}


# ===========================================================================
# Defense wrappers
# ===========================================================================

def _wrap_for_aggregation(grads: list[np.ndarray]) -> list[list[np.ndarray]]:
    """Convert a list of flat gradient arrays to GradientList format.

    server.aggregation expects GradientList = list[list[np.ndarray]], where
    each inner list represents one "layer" per client.  For flat vectors there
    is a single layer, so we wrap each array in a singleton list.

    Args:
        grads: List of 1-D gradient arrays, one per client.

    Returns:
        GradientList where each client has a single-element layer list.
    """
    return [[g] for g in grads]


def _unwrap_aggregated(aggregated: list[np.ndarray]) -> np.ndarray:
    """Extract the flat gradient from a single-layer Gradients result.

    Args:
        aggregated: Gradients = list[np.ndarray] with one element.

    Returns:
        The single 1-D gradient array.
    """
    return aggregated[0]


def _run_fedavg(grads: list[np.ndarray], num_byzantine: int) -> np.ndarray:
    """FedAvg aggregation (uniform weights, no filtering).

    Args:
        grads:         All client gradients including Byzantine ones.
        num_byzantine: Ignored — FedAvg has no Byzantine filtering.

    Returns:
        Averaged gradient.
    """
    return _unwrap_aggregated(fedavg(_wrap_for_aggregation(grads)))


def _run_krum(grads: list[np.ndarray], num_byzantine: int) -> np.ndarray:
    """Krum aggregation — selects one gradient closest to its neighbours.

    Args:
        grads:         All client gradients.
        num_byzantine: f parameter (Byzantine tolerance) passed to krum().

    Returns:
        Krum-selected gradient.
    """
    f = max(1, num_byzantine)
    return _unwrap_aggregated(krum(_wrap_for_aggregation(grads), f=f))


def _run_trimmed_mean(grads: list[np.ndarray], num_byzantine: int) -> np.ndarray:
    """Trimmed mean — removes extreme values per coordinate before averaging.

    trim_fraction is set to byzantine_fraction so we trim at least as many
    values as there are Byzantine clients on each tail.

    Args:
        grads:         All client gradients.
        num_byzantine: Used to set trim_fraction = num_byzantine / len(grads).

    Returns:
        Trimmed mean gradient.
    """
    n = len(grads)
    # Trim exactly the Byzantine fraction, clamped safely below 0.5
    trim_frac = min(num_byzantine / n, 0.45)
    return _unwrap_aggregated(trimmed_mean(_wrap_for_aggregation(grads), trim_fraction=trim_frac))


def _run_coordinate_median(grads: list[np.ndarray], num_byzantine: int) -> np.ndarray:
    """Coordinate-wise median aggregation.

    Args:
        grads:         All client gradients.
        num_byzantine: Ignored — median is parameter-free.

    Returns:
        Coordinate-wise median gradient.
    """
    return _unwrap_aggregated(coordinate_median(_wrap_for_aggregation(grads)))


def _run_flame(grads: list[np.ndarray], num_byzantine: int) -> np.ndarray:
    """FLAME: norm-clip + cosine clustering + Gaussian DP noise.

    A high epsilon (low noise) is used to measure robustness under minimal
    privacy noise; the clustering step is the primary defense mechanism here.
    (Nguyen et al., arXiv:2101.02281)

    Args:
        grads:         All client gradients.
        num_byzantine: Ignored — FLAME uses clustering to identify attackers.

    Returns:
        FLAME-aggregated gradient.
    """
    return _unwrap_aggregated(flame(_wrap_for_aggregation(grads), epsilon=3000.0))


# Map defense name -> callable(grads, num_byzantine) -> aggregated_grad
DefenseFn = Callable[[list[np.ndarray], int], np.ndarray]

DEFENSES: dict[str, DefenseFn] = {
    "fedavg":             _run_fedavg,
    "krum":               _run_krum,
    "trimmed_mean":       _run_trimmed_mean,
    "coordinate_median":  _run_coordinate_median,
    "flame":              _run_flame,
}


# ===========================================================================
# Robustness metric
# ===========================================================================

def robustness_score(aggregated: np.ndarray, expected: np.ndarray) -> float:
    """Compute 1 - normalised L2 error between aggregated and expected gradient.

    Score = 1  →  perfect recovery (aggregated == expected).
    Score = 0  →  L2 error equals ||expected|| (complete failure baseline).
    Score < 0  →  error larger than the reference norm (defense made it worse).

    Normalisation by ||expected|| makes the score independent of gradient scale
    and comparable across different attack magnitudes.

    Args:
        aggregated: Aggregated gradient returned by the defense.
        expected:   Ground-truth honest mean gradient.

    Returns:
        Float robustness score, typically in [-inf, 1.0].
    """
    if expected.shape != aggregated.shape:
        raise ValueError(
            f"Shape mismatch: aggregated {aggregated.shape} vs expected {expected.shape}"
        )
    l2_error = float(np.linalg.norm(aggregated - expected))
    ref_norm = float(np.linalg.norm(expected))
    if ref_norm < 1e-12:
        # Degenerate case: expected gradient is essentially zero
        return 1.0 if l2_error < 1e-12 else 0.0
    return float(1.0 - l2_error / ref_norm)


# ===========================================================================
# Single trial
# ===========================================================================

def run_trial(
    defense_name: str,
    defense_fn: DefenseFn,
    attack_name: str,
    attack_fn: AttackFn,
    byzantine_fraction: float,
    seed: int,
    num_rounds: int = NUM_ROUNDS,
) -> dict:
    """Run one trial (multiple rounds) for a single defense/attack/fraction/seed.

    Each round independently samples honest gradients so per-round variance is
    captured. The trial score is the mean across rounds.

    Args:
        defense_name:      String label for logging.
        defense_fn:        Callable(grads, num_byzantine) -> aggregated_grad.
        attack_name:       String label for logging.
        attack_fn:         Callable(honest_grad, rng) -> poisoned_grad.
        byzantine_fraction: Fraction of NUM_CLIENTS that are Byzantine.
        seed:              Master seed; each round uses seed + round_idx.
        num_rounds:        Number of rounds to average over.

    Returns:
        Dict with keys: defense, attack, byzantine_fraction, seed,
        round_scores, mean_score, std_score.

    Raises:
        ValueError: If byzantine_fraction is outside [0, 1).
    """
    if not (0.0 <= byzantine_fraction < 1.0):
        raise ValueError(f"byzantine_fraction must be in [0, 1), got {byzantine_fraction}")

    num_byzantine = int(round(byzantine_fraction * NUM_CLIENTS))
    num_honest = NUM_CLIENTS - num_byzantine

    round_scores: list[float] = []

    for round_idx in range(num_rounds):
        rng = np.random.default_rng(seed + round_idx * 1000)

        # ── Generate honest gradients ────────────────────────────────────────
        honest_grads = [_honest_gradient(rng) for _ in range(num_honest)]

        # ── Compute the ground-truth target: mean of honest gradients ────────
        target = _honest_mean(honest_grads)

        # ── Generate Byzantine gradients using a representative honest grad ──
        # Use the mean honest gradient as the "reference" seen by Byzantine
        # clients (simulates a well-calibrated attacker who knows the update).
        byzantine_grads = [attack_fn(target, rng) for _ in range(num_byzantine)]

        # ── Combine and shuffle so positional bias does not affect Krum ──────
        all_grads = honest_grads + byzantine_grads
        shuffle_idx = rng.permutation(NUM_CLIENTS)
        all_grads = [all_grads[i] for i in shuffle_idx]

        # ── Aggregate ────────────────────────────────────────────────────────
        try:
            aggregated = defense_fn(all_grads, num_byzantine)
        except Exception as exc:
            log.warning(
                "Defense '%s' failed on round %d (attack='%s', f=%.1f, seed=%d): %s",
                defense_name, round_idx, attack_name, byzantine_fraction, seed, exc,
            )
            # Treat a crashed defense as zero score for that round
            aggregated = np.zeros_like(target)

        score = robustness_score(aggregated, target)
        round_scores.append(score)

    return {
        "defense":            defense_name,
        "attack":             attack_name,
        "byzantine_fraction": byzantine_fraction,
        "seed":               seed,
        "round_scores":       round_scores,
        "mean_score":         float(np.mean(round_scores)),
        "std_score":          float(np.std(round_scores)),
    }


# ===========================================================================
# Full benchmark matrix
# ===========================================================================

def run_benchmark() -> list[dict]:
    """Run the full attack-defense matrix benchmark.

    Iterates over all combinations of:
        DEFENSES × ATTACKS × BYZANTINE_FRACTIONS × SEEDS

    and returns a flat list of trial result dicts.

    Returns:
        List of trial result dicts (one per combination).
    """
    defense_names = list(DEFENSES.keys())
    attack_names = list(ATTACKS.keys())

    total = len(defense_names) * len(attack_names) * len(BYZANTINE_FRACTIONS) * len(SEEDS)
    log.info(
        "Starting benchmark: %d defenses × %d attacks × %d fractions × %d seeds = %d trials",
        len(defense_names), len(attack_names), len(BYZANTINE_FRACTIONS), len(SEEDS), total,
    )

    results: list[dict] = []
    completed = 0
    t0 = time.monotonic()

    for defense_name, defense_fn in DEFENSES.items():
        for attack_name, attack_fn in ATTACKS.items():
            for byz_frac in BYZANTINE_FRACTIONS:
                for seed in SEEDS:
                    trial = run_trial(
                        defense_name=defense_name,
                        defense_fn=defense_fn,
                        attack_name=attack_name,
                        attack_fn=attack_fn,
                        byzantine_fraction=byz_frac,
                        seed=seed,
                    )
                    results.append(trial)
                    completed += 1

                    if completed % 10 == 0 or completed == total:
                        elapsed = time.monotonic() - t0
                        log.info(
                            "[%3d/%d]  %-20s  %-16s  byz=%.1f  seed=%d  "
                            "score=%.4f  (%.1fs elapsed)",
                            completed, total,
                            defense_name, attack_name,
                            byz_frac, seed,
                            trial["mean_score"],
                            elapsed,
                        )

    log.info("Benchmark complete. %d trials in %.1fs.", total, time.monotonic() - t0)
    return results


# ===========================================================================
# Results aggregation
# ===========================================================================

def aggregate_results(results: list[dict]) -> dict:
    """Aggregate trial results into a nested summary dict.

    The summary maps defense -> attack -> byzantine_fraction ->
    {mean_score, std_score, n_seeds} by averaging across seeds.

    Args:
        results: Flat list of trial result dicts from run_benchmark().

    Returns:
        Nested dict: summary[defense][attack][byz_frac_str] = metrics_dict.
        Also includes a top-level "overall" key with the grand mean.
    """
    # Group by (defense, attack, byz_frac)
    from collections import defaultdict
    groups: dict[tuple, list[float]] = defaultdict(list)

    for r in results:
        key = (r["defense"], r["attack"], r["byzantine_fraction"])
        groups[key].append(r["mean_score"])

    summary: dict = {}
    all_scores: list[float] = []

    for (defense, attack, byz_frac), scores in groups.items():
        summary.setdefault(defense, {}).setdefault(attack, {})[str(byz_frac)] = {
            "mean_score": float(np.mean(scores)),
            "std_score":  float(np.std(scores)),
            "n_seeds":    len(scores),
        }
        all_scores.extend(scores)

    summary["_meta"] = {
        "overall_mean_score": float(np.mean(all_scores)),
        "num_trials":         len(results),
        "defenses":           list(DEFENSES.keys()),
        "attacks":            list(ATTACKS.keys()),
        "byzantine_fractions": BYZANTINE_FRACTIONS,
        "seeds":              SEEDS,
        "num_clients":        NUM_CLIENTS,
        "grad_dim":           GRAD_DIM,
        "num_rounds":         NUM_ROUNDS,
        "honest_noise_std":   HONEST_NOISE_STD,
    }

    return summary


# ===========================================================================
# Heatmap: defense (rows) × attack (cols), averaged over fractions and seeds
# ===========================================================================

def build_heatmap_matrix(
    summary: dict,
    defense_names: list[str],
    attack_names: list[str],
) -> np.ndarray:
    """Build the 2-D score matrix for the defense × attack heatmap.

    Scores are averaged over all Byzantine fractions and seeds to produce a
    single representative value per (defense, attack) cell.

    Args:
        summary:       Nested summary dict from aggregate_results().
        defense_names: Ordered list of defense names (row order).
        attack_names:  Ordered list of attack names (column order).

    Returns:
        2-D float array of shape (len(defense_names), len(attack_names)).
    """
    matrix = np.zeros((len(defense_names), len(attack_names)), dtype=float)

    for row_idx, defense in enumerate(defense_names):
        for col_idx, attack in enumerate(attack_names):
            frac_means = []
            for byz_frac in BYZANTINE_FRACTIONS:
                entry = summary.get(defense, {}).get(attack, {}).get(str(byz_frac))
                if entry is not None:
                    frac_means.append(entry["mean_score"])
            matrix[row_idx, col_idx] = float(np.mean(frac_means)) if frac_means else 0.0

    return matrix


def plot_heatmap(
    matrix: np.ndarray,
    defense_names: list[str],
    attack_names: list[str],
    save_path: Path,
) -> None:
    """Render and save the defense × attack robustness heatmap.

    Rows = defenses, columns = attacks. Each cell is annotated with the
    mean robustness score (averaged over Byzantine fractions and seeds).
    A diverging colormap (RdYlGn) maps low scores (red) to high scores (green).

    Args:
        matrix:        2-D score matrix, shape (n_defenses, n_attacks).
        defense_names: Row labels.
        attack_names:  Column labels.
        save_path:     File path for the saved PNG.
    """
    n_defenses, n_attacks = matrix.shape
    fig_width = max(8, n_attacks * 2.0)
    fig_height = max(5, n_defenses * 1.2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Clamp colormap at [0, 1] for a clean red-green scale; scores outside
    # this range (defense did better or much worse than baseline) still render
    # but use the extreme colors.
    vmin, vmax = 0.0, 1.0
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # Axis ticks and labels
    ax.set_xticks(range(n_attacks))
    ax.set_yticks(range(n_defenses))
    ax.set_xticklabels(
        [a.replace("_", "\n") for a in attack_names],
        fontsize=11, fontweight="bold",
    )
    ax.set_yticklabels(
        [d.replace("_", "\n") for d in defense_names],
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Attack", fontsize=13, labelpad=10)
    ax.set_ylabel("Defense", fontsize=13, labelpad=10)
    ax.set_title(
        "Aggregation Robustness: Defense × Attack Matrix\n"
        "(score = 1 − normalised L2 error; higher = more robust)",
        fontsize=13, pad=14,
    )

    # Annotate each cell with the numeric score
    for row in range(n_defenses):
        for col in range(n_attacks):
            score = matrix[row, col]
            # Use dark text on light cells, light text on dark cells
            brightness = cmap(norm(score))[0] * 0.299 + cmap(norm(score))[1] * 0.587 + cmap(norm(score))[2] * 0.114
            text_color = "black" if brightness > 0.5 else "white"
            ax.text(
                col, row, f"{score:.3f}",
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                color=text_color,
            )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Robustness Score", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Heatmap saved to %s", save_path)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """Run the full benchmark, save JSON results, and produce the heatmap.

    Outputs
    -------
    experiments/results/aggregation_benchmark.json
        Full trial-level results plus aggregated summary.
    experiments/results/aggregation_heatmap.png
        Defense × attack heatmap (mean robustness score).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Results directory: %s", RESULTS_DIR)

    # ── Run benchmark ────────────────────────────────────────────────────────
    trials = run_benchmark()

    # ── Aggregate and save JSON ──────────────────────────────────────────────
    summary = aggregate_results(trials)

    output = {
        "summary": summary,
        "trials":  trials,
    }
    with open(JSON_PATH, "w") as fh:
        json.dump(output, fh, indent=2)
    log.info("Results saved to %s  (%d bytes)", JSON_PATH, JSON_PATH.stat().st_size)

    # ── Build heatmap matrix ─────────────────────────────────────────────────
    defense_names = list(DEFENSES.keys())
    attack_names = list(ATTACKS.keys())

    matrix = build_heatmap_matrix(summary, defense_names, attack_names)

    # ── Print ASCII table to console ─────────────────────────────────────────
    col_w = 16
    header = f"{'Defense':<20}" + "".join(f"{a:>{col_w}}" for a in attack_names)
    log.info("Robustness score matrix (mean over fractions and seeds):")
    log.info("%s", header)
    log.info("%s", "-" * len(header))
    for row_idx, defense in enumerate(defense_names):
        row_str = f"{defense:<20}" + "".join(f"{matrix[row_idx, col_idx]:>{col_w}.4f}" for col_idx in range(len(attack_names)))
        log.info("%s", row_str)

    # ── Render and save heatmap ──────────────────────────────────────────────
    plot_heatmap(matrix, defense_names, attack_names, HEATMAP_PATH)

    log.info("Benchmark finished successfully.")


if __name__ == "__main__":
    main()
