"""
Aggregation strategies for federated learning.

Implements FedAvg, Krum, Trimmed Mean, Median, and FLAME.

References:
  - FedAvg:       arXiv:1602.05629 (McMahan et al., 2017)
  - Krum:         NeurIPS 2017, Blanchard et al. (arXiv:1703.02757)
  - Trimmed Mean: arXiv:1803.01498 (Yin et al., 2018)
  - FLAME:        arXiv:2101.02281 (Nguyen et al., 2022)
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)


# ── Type alias ────────────────────────────────────────────────────────────────
Gradients = list[np.ndarray]   # one array per layer
GradientList = list[Gradients] # one per client


# ── FedAvg ────────────────────────────────────────────────────────────────────

def fedavg(gradients: GradientList, weights: list[float] | None = None) -> Gradients:
    """Weighted average of client gradients (McMahan et al., arXiv:1602.05629).

    Args:
        gradients: List of per-client gradient arrays.
        weights:   Per-client weights (e.g. dataset sizes). Uniform if None.

    Returns:
        Aggregated gradient arrays.
    """
    if not gradients:
        raise ValueError("gradients list is empty")

    n = len(gradients)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    aggregated = []
    for layer_idx in range(len(gradients[0])):
        layer_agg = sum(
            w * g[layer_idx] for w, g in zip(weights, gradients)
        )
        aggregated.append(layer_agg)
    return aggregated


# ── Krum ──────────────────────────────────────────────────────────────────────

def krum(gradients: GradientList, f: int = 1) -> Gradients:
    """Krum: select the gradient closest to its n-f-2 neighbours.

    Byzantine-robust for f < n/2 - 1 malicious clients.
    (Blanchard et al., NeurIPS 2017, arXiv:1703.02757)

    Args:
        gradients: Per-client gradients.
        f:         Number of Byzantine clients to tolerate.

    Returns:
        The single gradient selected by Krum.
    """
    if not gradients:
        raise ValueError("gradients list is empty")

    n = len(gradients)
    if f >= n // 2:
        log.warning("Krum: f=%d >= n/2=%d — tolerance exceeded, falling back to f=0", f, n // 2)
        f = max(0, n // 2 - 1)

    # Flatten each client's gradient into a single vector
    flat = [np.concatenate([g.flatten() for g in grads]) for grads in gradients]

    scores = []
    for i in range(n):
        dists = sorted(
            np.linalg.norm(flat[i] - flat[j]) ** 2
            for j in range(n) if j != i
        )
        # Sum of (n - f - 2) smallest distances
        scores.append(sum(dists[: n - f - 2]))

    selected = int(np.argmin(scores))
    log.debug("Krum selected client %d (score=%.4f)", selected, scores[selected])
    return gradients[selected]


def multi_krum(gradients: GradientList, f: int = 1, m: int | None = None) -> Gradients:
    """Multi-Krum: select m candidates via Krum scoring, then average them.

    Args:
        gradients: Per-client gradients.
        f:         Byzantine tolerance.
        m:         Number of candidates to select. Defaults to n - f.

    Returns:
        Averaged gradients of the m selected candidates.
    """
    n = len(gradients)
    m = m or (n - f)
    flat = [np.concatenate([g.flatten() for g in grads]) for grads in gradients]

    scores = []
    for i in range(n):
        dists = sorted(
            np.linalg.norm(flat[i] - flat[j]) ** 2
            for j in range(n) if j != i
        )
        scores.append(sum(dists[: n - f - 2]))

    selected_indices = sorted(range(n), key=lambda i: scores[i])[:m]
    selected = [gradients[i] for i in selected_indices]
    return fedavg(selected)


# ── Trimmed Mean ──────────────────────────────────────────────────────────────

def trimmed_mean(gradients: GradientList, trim_fraction: float = 0.1) -> Gradients:
    """Coordinate-wise trimmed mean.

    Removes the top and bottom `trim_fraction` of values per coordinate,
    then averages the remainder. (Yin et al., arXiv:1803.01498)

    Args:
        gradients:     Per-client gradients.
        trim_fraction: Fraction to trim from each tail (0 < trim_fraction < 0.5).

    Returns:
        Trimmed mean gradient arrays.
    """
    if not (0 <= trim_fraction < 0.5):
        raise ValueError(f"trim_fraction must be in [0, 0.5), got {trim_fraction}")
    if not gradients:
        raise ValueError("gradients list is empty")

    n = len(gradients)
    k = int(np.floor(trim_fraction * n))  # number to trim each side

    aggregated = []
    for layer_idx in range(len(gradients[0])):
        stacked = np.stack([g[layer_idx] for g in gradients], axis=0)  # [n, *shape]
        if k > 0:
            stacked = np.sort(stacked, axis=0)[k: n - k]
        aggregated.append(stacked.mean(axis=0))

    return aggregated


# ── Coordinate-wise Median ────────────────────────────────────────────────────

def coordinate_median(gradients: GradientList) -> Gradients:
    """Coordinate-wise median aggregation.

    Byzantine-robust: the median is influenced by at most 50% of clients.

    Args:
        gradients: Per-client gradients.

    Returns:
        Coordinate-wise median gradient arrays.
    """
    if not gradients:
        raise ValueError("gradients list is empty")

    aggregated = []
    for layer_idx in range(len(gradients[0])):
        stacked = np.stack([g[layer_idx] for g in gradients], axis=0)
        aggregated.append(np.median(stacked, axis=0))
    return aggregated


# ── FLAME ─────────────────────────────────────────────────────────────────────

def flame(gradients: GradientList, epsilon: float = 3000.0) -> Gradients:
    """FLAME: clustering + noise-based defense against model poisoning.

    Uses cosine similarity clustering to filter outliers, then adds
    calibrated noise. (Nguyen et al., arXiv:2101.02281)

    Args:
        gradients: Per-client gradients.
        epsilon:   Noise magnitude bound.

    Returns:
        Filtered and noised aggregate.
    """
    if not gradients:
        raise ValueError("gradients list is empty")

    from sklearn.cluster import AgglomerativeClustering

    # Flatten gradients for clustering
    flat = np.array([np.concatenate([g.flatten() for g in grads]) for grads in gradients])

    # Normalise to unit vectors for cosine distance
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    flat_norm = flat / norms

    # Agglomerative clustering with cosine distance
    n = len(gradients)
    n_clusters = max(2, n // 2)
    try:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average"
        )
        labels = clustering.fit_predict(flat_norm)
        # Keep the largest cluster
        from collections import Counter
        dominant_label = Counter(labels).most_common(1)[0][0]
        selected_indices = [i for i, l in enumerate(labels) if l == dominant_label]
    except Exception as exc:
        log.warning("FLAME clustering failed (%s), falling back to FedAvg", exc)
        selected_indices = list(range(n))

    log.debug("FLAME: kept %d/%d clients", len(selected_indices), n)
    selected = [gradients[i] for i in selected_indices]

    # Aggregate selected
    agg = fedavg(selected)

    # Add calibrated Gaussian noise
    sensitivity = np.median([np.linalg.norm(flat[i]) for i in selected_indices])
    noise_std = sensitivity * epsilon / n
    noised = [layer + np.random.normal(0, noise_std, layer.shape) for layer in agg]
    return noised


# ── Registry ──────────────────────────────────────────────────────────────────

def get_aggregation_fn(name: str, **kwargs) -> Callable[[GradientList], Gradients]:
    """Return aggregation function by name.

    Args:
        name:   One of 'fedavg', 'krum', 'multi_krum', 'trimmed_mean', 'median', 'flame'.
        kwargs: Passed to the aggregation function.

    Returns:
        A callable that takes GradientList and returns Gradients.
    """
    registry: dict[str, Callable] = {
        "fedavg":        lambda g: fedavg(g),
        "krum":          lambda g: krum(g, f=kwargs.get("f", 1)),
        "multi_krum":    lambda g: multi_krum(g, f=kwargs.get("f", 1)),
        "trimmed_mean":  lambda g: trimmed_mean(g, trim_fraction=kwargs.get("trim_fraction", 0.1)),
        "median":        lambda g: coordinate_median(g),
        "flame":         lambda g: flame(g, epsilon=kwargs.get("epsilon", 3000.0)),
    }
    if name not in registry:
        raise ValueError(f"Unknown aggregation '{name}'. Choose from: {list(registry)}")
    return registry[name]
