"""Aggregation rules operating on a stack of client updates.

Every function takes ``updates`` with shape ``(n_clients, n_params)`` (each client's
model flattened into one vector) and returns one aggregated vector. They are pure
numpy so they can be unit-tested without Flower.

* ``fedavg``        - example-weighted mean (no defence)
* ``median``        - coordinate-wise median (Yin et al., 2018)
* ``trimmed_mean``  - coordinate-wise trimmed mean (Yin et al., 2018)
* ``krum``          - pick the single most "central" update (Blanchard et al., 2017)
* ``multikrum``     - average the n - f most central updates
"""

from __future__ import annotations

import numpy as np

METHODS = ("fedavg", "median", "trimmed_mean", "krum", "multikrum")


def flatten_params(params: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in params])


def unflatten_params(flat: np.ndarray, shapes: list[tuple]) -> list[np.ndarray]:
    out, i = [], 0
    for shape in shapes:
        n = int(np.prod(shape))
        out.append(flat[i : i + n].reshape(shape).copy())
        i += n
    return out


def fedavg(updates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    return (w / w.sum()) @ updates


def coordinate_median(updates: np.ndarray) -> np.ndarray:
    return np.median(updates, axis=0)


def trimmed_mean(updates: np.ndarray, trim_ratio: float) -> np.ndarray:
    """Drop the ``trim_ratio`` largest and smallest values per coordinate, then average."""
    n = updates.shape[0]
    k = int(np.floor(trim_ratio * n + 1e-9))
    k = min(k, (n - 1) // 2)  # always keep at least one value
    if k <= 0:
        return updates.mean(axis=0)
    return np.sort(updates, axis=0)[k : n - k].mean(axis=0)


def _clamp_f(n: int, f: int) -> int:
    """Krum needs n >= 2f + 3; clamp f to the largest value that satisfies it."""
    return int(max(0, min(f, (n - 3) // 2)))


def krum_scores(updates: np.ndarray, f: int) -> np.ndarray:
    """Krum score = sum of squared distances to the n - f - 2 nearest other updates."""
    n = updates.shape[0]
    sq = np.sum(updates * updates, axis=1)
    dist = sq[:, None] + sq[None, :] - 2.0 * (updates @ updates.T)
    np.maximum(dist, 0.0, out=dist)
    np.fill_diagonal(dist, np.inf)
    k = max(1, n - _clamp_f(n, f) - 2)
    return np.sort(dist, axis=1)[:, :k].sum(axis=1)


def krum(updates: np.ndarray, f: int, m: int = 1) -> tuple[np.ndarray, list[int]]:
    """(Multi-)Krum. Returns ``(aggregate, selected_client_indices)``."""
    n = updates.shape[0]
    if n < 3:  # not enough clients for a meaningful distance-based rule
        return updates.mean(axis=0), list(range(n))
    selected = np.argsort(krum_scores(updates, f))[: int(np.clip(m, 1, n))]
    return updates[selected].mean(axis=0), [int(i) for i in selected]


def aggregate(
    method: str,
    updates: np.ndarray,
    weights: np.ndarray,
    f: int = 0,
    trim_ratio: float = 0.0,
) -> tuple[np.ndarray, list[int] | None]:
    """Dispatch to an aggregation rule. Second value = clients kept (Krum family)."""
    if method == "fedavg":
        return fedavg(updates, weights), None
    if method == "median":
        return coordinate_median(updates), None
    if method == "trimmed_mean":
        return trimmed_mean(updates, trim_ratio), None
    if method == "krum":
        return krum(updates, f, m=1)
    if method == "multikrum":
        n = updates.shape[0]
        return krum(updates, f, m=n - _clamp_f(n, f))
    raise ValueError(f"Unknown aggregation '{method}'. Choose from {METHODS}.")
