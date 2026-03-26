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
    """FLAME: norm-clipping + cosine clustering + DP noise defense against poisoning.

    Implements the three core steps from Nguyen et al., arXiv:2101.02281:
      1. Clip each gradient to the median L2 norm (removes scale-based attacks).
      2. Binary cosine-similarity clustering to isolate the dominant cluster.
      3. FedAvg the dominant cluster + calibrated Gaussian noise.

    Args:
        gradients: Per-client gradients.
        epsilon:   DP noise budget — larger epsilon means less noise added.

    Returns:
        Filtered and noised aggregate.
    """
    if not gradients:
        raise ValueError("gradients list is empty")

    from sklearn.cluster import AgglomerativeClustering
    from collections import Counter

    n = len(gradients)

    # Step 1: Compute per-client L2 norms on the flattened gradient and clip to
    # the median norm.  This neutralises scale-based poisoning (e.g. a gradient
    # that is 500× larger than honest clients) before clustering.
    flat_raw = np.array([np.concatenate([g.flatten() for g in grads]) for grads in gradients])
    raw_norms = np.linalg.norm(flat_raw, axis=1)
    clip_bound = float(np.median(raw_norms))

    def _clip(grads: Gradients, raw_norm: float) -> Gradients:
        scale = min(1.0, clip_bound / (raw_norm + 1e-9))
        return [layer * scale for layer in grads]

    clipped = [_clip(grads, float(raw_norms[i])) for i, grads in enumerate(gradients)]

    # Step 2: Cluster clipped (unit-normalised) gradients with binary cosine
    # agglomerative clustering and keep the dominant cluster.
    flat_clipped = np.array([np.concatenate([g.flatten() for g in grads]) for grads in clipped])
    norms_clipped = np.linalg.norm(flat_clipped, axis=1, keepdims=True)
    norms_clipped = np.where(norms_clipped == 0, 1.0, norms_clipped)
    flat_norm = flat_clipped / norms_clipped

    try:
        clustering = AgglomerativeClustering(
            n_clusters=2, metric="cosine", linkage="average"
        )
        labels = clustering.fit_predict(flat_norm)
        dominant_label = Counter(labels).most_common(1)[0][0]
        selected_indices = [i for i, lbl in enumerate(labels) if lbl == dominant_label]
    except Exception as exc:
        log.warning("FLAME clustering failed (%s), falling back to FedAvg", exc)
        selected_indices = list(range(n))

    log.debug("FLAME: kept %d/%d clients", len(selected_indices), n)
    selected = [clipped[i] for i in selected_indices]

    # Step 3: Aggregate selected clipped gradients and add DP Gaussian noise.
    # noise_std = clip_bound / (epsilon * sqrt(n)) follows the standard DP Gaussian
    # mechanism: larger epsilon → less noise → less privacy but more utility.
    agg = fedavg(selected)
    noise_std = clip_bound / (epsilon * np.sqrt(n))
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
