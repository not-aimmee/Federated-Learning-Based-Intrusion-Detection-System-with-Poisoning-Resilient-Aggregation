"""Poisoning attacks a malicious client can mount.

* ``label_flip``   (data poisoning)  - train on inverted labels, so attacks look benign
                                        and benign traffic looks like attacks.
* ``model_poison`` (update poisoning) - train honestly, then send the *opposite* of the
                                        update, amplified by ``scale``:
                                        ``w_sent = w_global - scale * (w_local - w_global)``.
"""

from __future__ import annotations

import numpy as np

ATTACKS = ("none", "label_flip", "model_poison")


def flip_labels(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Map class c -> (n_classes - 1 - c). For binary data this swaps 0 and 1."""
    return (n_classes - 1) - np.asarray(y)


def poison_update(
    global_params: list[np.ndarray], local_params: list[np.ndarray], scale: float
) -> list[np.ndarray]:
    return [g - scale * (l - g) for g, l in zip(global_params, local_params)]


def num_malicious(fraction: float, num_clients: int) -> int:
    """Clients with partition-id < this value are malicious."""
    return int(round(fraction * num_clients))
