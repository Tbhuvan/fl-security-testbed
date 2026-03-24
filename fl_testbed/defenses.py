"""
defenses.py — Robust aggregation defenses for Byzantine-resilient FL.

Implemented aggregators:

1. FedAvg       — standard weighted mean (no defense, baseline)
   Ref: McMahan et al., "Communication-Efficient Learning of Deep Networks"
        arXiv:1602.05629

2. Krum          — selects the update with smallest sum-of-distances to f+1 neighbors
   Ref: Blanchard et al., "Machine Learning with Adversaries" NeurIPS 2017

3. MultiKrum     — Krum but returns average of top-m selected updates

4. TrimmedMean   — coordinate-wise trimmed mean (drops top-β and bottom-β per coord)
   Ref: Yin et al., "Byzantine-Robust Distributed Learning" ICML 2018
        arXiv:1803.01498

5. Median        — coordinate-wise median
   Ref: Yin et al., ibid.

6. FLAME         — Clustering + noise injection defense against backdoors
   Ref: Nguyen et al., "FLAME: Taming Backdoors in FL" USENIX Security 2022
        arXiv:2101.02281

7. FLTrust       — Server-side reference update to score client updates by cosine similarity
   Ref: Cao et al., "FLTrust: Byzantine-robust FL via Trust Bootstrapping"
        arXiv:2012.13995

All aggregators conform to the Aggregator protocol:
    agg.aggregate(updates: List[List[Tensor]], weights: List[float]) -> List[Tensor]
"""

from __future__ import annotations

import math
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Aggregator(ABC):
    """Base class for FL aggregation strategies."""

    @abstractmethod
    def aggregate(
        self,
        updates: List[List[torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> List[torch.Tensor]:
        """Aggregate client updates into a single global update.

        Args:
            updates: List of per-client parameter update lists.
            weights: Optional per-client sample counts for weighted averaging.
                     If None, uniform weights are used.

        Returns:
            Aggregated parameter list (same structure as a single update).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def _uniform_weights(n: int) -> List[float]:
    return [1.0 / n] * n


def _normalize_weights(weights: List[float]) -> List[float]:
    total = sum(weights)
    return [w / total for w in weights]


def _flatten(params: List[torch.Tensor]) -> torch.Tensor:
    """Flatten parameter list into a 1D tensor."""
    return torch.cat([p.flatten() for p in params])


def _unflatten(flat: torch.Tensor, reference: List[torch.Tensor]) -> List[torch.Tensor]:
    """Restore flat 1D tensor to a list matching reference shapes."""
    result = []
    offset = 0
    for ref in reference:
        numel = ref.numel()
        result.append(flat[offset:offset + numel].view(ref.shape))
        offset += numel
    return result


# ---------------------------------------------------------------------------
# 1. FedAvg
# ---------------------------------------------------------------------------

class FedAvg(Aggregator):
    """Standard federated averaging — weighted mean of all updates."""

    def aggregate(
        self,
        updates: List[List[torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> List[torch.Tensor]:
        if not updates:
            raise ValueError("Cannot aggregate empty updates list")
        n = len(updates)
        w = _normalize_weights(weights if weights is not None else _uniform_weights(n))
        result = [torch.zeros_like(p) for p in updates[0]]
        for client_update, wi in zip(updates, w):
            for i, param in enumerate(client_update):
                result[i] += wi * param
        return result

    def __repr__(self) -> str:
        return "FedAvg()"


# ---------------------------------------------------------------------------
# 2. Krum / MultiKrum
# ---------------------------------------------------------------------------

class Krum(Aggregator):
    """Krum aggregation — selects the single update closest to its neighbors.

    Filters out f Byzantine updates by choosing the candidate with minimum
    sum of squared Euclidean distances to its (n - f - 2) nearest neighbors.

    Args:
        num_byzantine: Expected number of Byzantine clients (f).
        multi_k: If > 1, returns average of top-multi_k selected candidates (MultiKrum).
    """

    def __init__(self, num_byzantine: int = 1, multi_k: int = 1) -> None:
        if num_byzantine < 0:
            raise ValueError(f"num_byzantine must be >= 0, got {num_byzantine}")
        self.num_byzantine = num_byzantine
        self.multi_k = multi_k

    def aggregate(
        self,
        updates: List[List[torch.Tensor]],
        weights: Optional[List[float]] = None,  # Krum ignores weights
    ) -> List[torch.Tensor]:
        n = len(updates)
        f = self.num_byzantine
        if n < 2 * f + 3:
            raise ValueError(
                f"Krum requires n >= 2f+3, but n={n}, f={f}. "
                f"Reduce num_byzantine or add more clients."
            )

        flat_updates = [_flatten(u) for u in updates]
        k = n - f - 2  # number of nearest neighbors to consider

        # Compute pairwise squared distances
        scores = torch.zeros(n)
        for i in range(n):
            dists = torch.tensor([
                (flat_updates[i] - flat_updates[j]).pow(2).sum().item()
                for j in range(n) if j != i
            ])
            # Take sum of k smallest distances
            scores[i] = dists.topk(k, largest=False).values.sum()

        # Select multi_k best candidates
        selected_indices = scores.argsort()[:self.multi_k].tolist()
        selected_updates = [updates[i] for i in selected_indices]

        # Average the selected updates
        aggregator = FedAvg()
        return aggregator.aggregate(selected_updates)

    def __repr__(self) -> str:
        return f"Krum(f={self.num_byzantine}, multi_k={self.multi_k})"


class MultiKrum(Krum):
    """MultiKrum — averages the top-m Krum-selected candidates."""

    def __init__(self, num_byzantine: int = 1, m: int = 3) -> None:
        super().__init__(num_byzantine=num_byzantine, multi_k=m)

    def __repr__(self) -> str:
        return f"MultiKrum(f={self.num_byzantine}, m={self.multi_k})"


# ---------------------------------------------------------------------------
# 3. Coordinate-wise Trimmed Mean
# ---------------------------------------------------------------------------

class TrimmedMean(Aggregator):
    """Coordinate-wise trimmed mean — drops extreme values before averaging.

    For each parameter coordinate, sorts client values and removes the top
    and bottom `trim_fraction` of values, then takes the mean of the remainder.

    Args:
        trim_fraction: Fraction to trim from each end (e.g., 0.1 = 10% each side).
                       Must be in [0, 0.5).
    """

    def __init__(self, trim_fraction: float = 0.1) -> None:
        if not 0 <= trim_fraction < 0.5:
            raise ValueError(f"trim_fraction must be in [0, 0.5), got {trim_fraction}")
        self.trim_fraction = trim_fraction

    def aggregate(
        self,
        updates: List[List[torch.Tensor]],
        weights: Optional[List[float]] = None,  # TrimmedMean ignores weights
    ) -> List[torch.Tensor]:
        if not updates:
            raise ValueError("Cannot aggregate empty updates list")
        n = len(updates)
        k = max(1, int(n * self.trim_fraction))  # number to trim from each end

        result = []
        for layer_idx in range(len(updates[0])):
            stacked = torch.stack([u[layer_idx] for u in updates], dim=0)
            # Sort along client dimension, trim, then mean
            sorted_vals, _ = stacked.sort(dim=0)
            trimmed = sorted_vals[k:n - k] if n - 2 * k > 0 else sorted_vals
            result.append(trimmed.mean(dim=0))

        return result

    def __repr__(self) -> str:
        return f"TrimmedMean(trim={self.trim_fraction})"


# ---------------------------------------------------------------------------
# 4. Coordinate-wise Median
# ---------------------------------------------------------------------------

class CoordinateMedian(Aggregator):
    """Coordinate-wise median aggregation.

    Provably Byzantine-resilient for up to f < n/2 attackers per coordinate.
    More robust than TrimmedMean but slower for large models.
    """

    def aggregate(
        self,
        updates: List[List[torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> List[torch.Tensor]:
        if not updates:
            raise ValueError("Cannot aggregate empty updates list")
        result = []
        for layer_idx in range(len(updates[0])):
            stacked = torch.stack([u[layer_idx] for u in updates], dim=0)
            result.append(stacked.median(dim=0).values)
        return result

    def __repr__(self) -> str:
        return "CoordinateMedian()"


# ---------------------------------------------------------------------------
# 5. FLAME — Clustering + noise defense
# ---------------------------------------------------------------------------

class FLAME(Aggregator):
    """FLAME: Federated Learning AMEndments defense against backdoor attacks.

    Steps:
        1. Compute cosine similarity between all pairs of updates.
        2. Cluster with DBSCAN to detect and remove outlier backdoor updates.
        3. Clip remaining updates to median L2 norm.
        4. Add Gaussian noise scaled to clipping threshold (local DP guarantee).
        5. Aggregate with FedAvg.

    Ref: Nguyen et al., USENIX Security 2022 (arXiv:2101.02281)

    Args:
        noise_sigma: Standard deviation of Gaussian noise relative to clip threshold.
        min_cluster_size: DBSCAN min_samples parameter.
        eps: DBSCAN epsilon (cosine distance threshold for same cluster).
    """

    def __init__(
        self,
        noise_sigma: float = 0.001,
        min_cluster_size: int = 3,
        eps: float = 0.5,
    ) -> None:
        self.noise_sigma = noise_sigma
        self.min_cluster_size = min_cluster_size
        self.eps = eps

    def aggregate(
        self,
        updates: List[List[torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> List[torch.Tensor]:
        if not updates:
            raise ValueError("Cannot aggregate empty updates list")

        flat_updates = [_flatten(u) for u in updates]
        n = len(flat_updates)

        # Step 1: cosine distance matrix
        norms = torch.stack([f.norm(2) for f in flat_updates])
        normalized = torch.stack([
            f / (norm + 1e-8) for f, norm in zip(flat_updates, norms)
        ])
        cos_sim = normalized @ normalized.T
        cos_dist = (1.0 - cos_sim.clamp(-1.0, 1.0)).cpu().numpy()

        # Step 2: DBSCAN clustering on cosine distances
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_cluster_size,
            metric="precomputed",
        )
        labels = dbscan.fit_predict(cos_dist)

        # Keep updates in the largest cluster (label != -1)
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(unique_labels) == 0:
            # All outliers — fall back to full FedAvg (degenerate case)
            selected_indices = list(range(n))
        else:
            dominant_label = unique_labels[counts.argmax()]
            selected_indices = [i for i, l in enumerate(labels) if l == dominant_label]

        selected_updates = [updates[i] for i in selected_indices]
        selected_flat = [flat_updates[i] for i in selected_indices]

        # Step 3: Clip to median norm
        selected_norms = torch.stack([f.norm(2) for f in selected_flat])
        clip_threshold = selected_norms.median().item()

        clipped_updates = []
        for update, flat in zip(selected_updates, selected_flat):
            norm = flat.norm(2).item()
            scale = min(1.0, clip_threshold / (norm + 1e-8))
            clipped_updates.append([p * scale for p in update])

        # Step 4: FedAvg on clipped
        aggregator = FedAvg()
        aggregated = aggregator.aggregate(clipped_updates)

        # Step 5: Add Gaussian noise
        if self.noise_sigma > 0:
            noise_std = self.noise_sigma * clip_threshold
            aggregated = [
                p + torch.randn_like(p) * noise_std for p in aggregated
            ]

        return aggregated

    def __repr__(self) -> str:
        return f"FLAME(sigma={self.noise_sigma}, eps={self.eps})"


# ---------------------------------------------------------------------------
# 6. FLTrust
# ---------------------------------------------------------------------------

class FLTrust(Aggregator):
    """FLTrust — server-side trust scoring via cosine similarity.

    Server computes a reference update using a small clean dataset,
    then weights each client update by its cosine similarity to the reference.
    Negative similarities are zeroed out (untrusted clients).

    Ref: Cao et al., arXiv:2012.13995

    Args:
        server_update: The server's reference gradient update (from clean data).
                       Must be provided at init or before first aggregate() call.
    """

    def __init__(self, server_update: Optional[List[torch.Tensor]] = None) -> None:
        self.server_update = server_update

    def set_server_update(self, server_update: List[torch.Tensor]) -> None:
        self.server_update = server_update

    def aggregate(
        self,
        updates: List[List[torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> List[torch.Tensor]:
        if self.server_update is None:
            raise ValueError(
                "FLTrust requires server_update. "
                "Call set_server_update() before aggregate()."
            )
        if not updates:
            raise ValueError("Cannot aggregate empty updates list")

        flat_server = _flatten(self.server_update)
        flat_server_norm = flat_server / (flat_server.norm(2) + 1e-8)

        # Compute trust scores (ReLU of cosine similarity)
        trust_scores = []
        normalized_updates = []
        for update in updates:
            flat = _flatten(update)
            norm = flat.norm(2)
            flat_norm = flat / (norm + 1e-8)
            cos_sim = float((flat_norm @ flat_server_norm).clamp(min=0.0))
            trust_scores.append(cos_sim)
            # Scale client update to server update magnitude
            server_norm = flat_server.norm(2).item()
            scale = server_norm / (norm + 1e-8)
            normalized_updates.append([p * scale for p in update])

        total_trust = sum(trust_scores)
        if total_trust < 1e-8:
            # All clients have zero or negative cosine similarity — return server update
            return self.server_update

        trust_weights = [t / total_trust for t in trust_scores]

        aggregator = FedAvg()
        return aggregator.aggregate(normalized_updates, weights=trust_weights)

    def __repr__(self) -> str:
        return "FLTrust()"


# ---------------------------------------------------------------------------
# Defense registry
# ---------------------------------------------------------------------------

DEFENSE_REGISTRY = {
    "fedavg": FedAvg,
    "krum": Krum,
    "multikrum": MultiKrum,
    "trimmedmean": TrimmedMean,
    "median": CoordinateMedian,
    "flame": FLAME,
    "fltrust": FLTrust,
}


def get_defense(name: str, **kwargs) -> Aggregator:
    """Factory: return a defense aggregator by name.

    Args:
        name: One of 'fedavg', 'krum', 'multikrum', 'trimmedmean',
              'median', 'flame', 'fltrust'.
        **kwargs: Passed to the aggregator constructor.

    Returns:
        Aggregator instance.
    """
    key = name.lower()
    if key not in DEFENSE_REGISTRY:
        raise ValueError(f"Unknown defense '{name}'. Choose from: {list(DEFENSE_REGISTRY)}")
    return DEFENSE_REGISTRY[key](**kwargs)
