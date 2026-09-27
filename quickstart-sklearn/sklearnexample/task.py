"""Data access, model helpers and metrics shared by client, server and scripts.

The model is a logistic-regression classifier trained with SGD
(``sklearn.linear_model.SGDClassifier``). Its state is just ``[coef_, intercept_]``,
which is exactly what Flower exchanges and what the robust aggregators operate on.

Data format (produced by ``scripts/preprocess_cicids2017.py`` or
``scripts/make_synthetic_data.py``) in ``data/processed/``::

    X_train.npy  y_train.npy  X_test.npy  y_test.npy  meta.json

Class 0 is always BENIGN; every other class id is an attack.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from sklearnexample.config import data_dir as default_data_dir

NDArrays = list[np.ndarray]
_FILES = ("X_train", "y_train", "X_test", "y_test")


# --------------------------------------------------------------------------- data
def _resolve(data_dir: str | Path | None) -> str:
    return str(Path(data_dir) if data_dir else default_data_dir())


@lru_cache(maxsize=2)
def load_meta(data_dir: str | None = None) -> dict:
    path = Path(_resolve(data_dir)) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/preprocess_cicids2017.py (real data) or "
            "scripts/make_synthetic_data.py (quick test data) first."
        )
    return json.loads(path.read_text())


@lru_cache(maxsize=2)
def _arrays(data_dir: str) -> dict[str, np.ndarray]:
    # Memory-mapped: each client only materialises its own partition.
    return {k: np.load(Path(data_dir) / f"{k}.npy", mmap_mode="r") for k in _FILES}


@lru_cache(maxsize=16)
def _partitions(data_dir: str, num_partitions: int, alpha: float, seed: int):
    """Split training-row indices across clients (IID if alpha <= 0, else Dirichlet)."""
    y = np.asarray(_arrays(data_dir)["y_train"])
    rng = np.random.default_rng(seed)
    if alpha <= 0:
        return tuple(np.array_split(rng.permutation(len(y)), num_partitions))

    out: list[np.ndarray] = []
    for _ in range(50):  # retry until every client has a usable amount of data
        parts: list[list[np.ndarray]] = [[] for _ in range(num_partitions)]
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            rng.shuffle(idx)
            props = rng.dirichlet(np.full(num_partitions, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for part, chunk in zip(parts, np.split(idx, cuts)):
                part.append(chunk)
        out = [np.concatenate(p) for p in parts]
        if min(len(o) for o in out) >= 100:
            break
    return tuple(out)


def load_client_data(
    partition_id: int,
    num_partitions: int,
    alpha: float = 0.0,
    seed: int = 42,
    val_fraction: float = 0.2,
    data_dir: str | Path | None = None,
):
    """Return ``(X_train, y_train, X_val, y_val)`` for one simulated organisation."""
    d = _resolve(data_dir)
    arrays = _arrays(d)
    idx = _partitions(d, int(num_partitions), float(alpha), int(seed))[partition_id]
    rng = np.random.default_rng(int(seed) + 1000 + int(partition_id))
    perm = rng.permutation(len(idx))
    n_val = max(1, int(len(idx) * val_fraction))
    val_idx = np.sort(idx[perm[:n_val]])
    tr_idx = np.sort(idx[perm[n_val:]])

    def take(name: str, rows: np.ndarray, dtype) -> np.ndarray:
        return np.asarray(arrays[name][rows], dtype=dtype)

    return (
        take("X_train", tr_idx, np.float64),
        take("y_train", tr_idx, np.int64),
        take("X_train", val_idx, np.float64),
        take("y_train", val_idx, np.int64),
    )


@lru_cache(maxsize=2)
def load_test_data(data_dir: str | None = None):
    """Held-out global test set (natural class distribution)."""
    arrays = _arrays(_resolve(data_dir))
    return (
        np.asarray(arrays["X_test"], dtype=np.float64),
        np.asarray(arrays["y_test"], dtype=np.int64),
    )


# -------------------------------------------------------------------------- model
def create_model(
    n_features: int, n_classes: int, learning_rate: float = 0.01, seed: int = 42
) -> SGDClassifier:
    """Logistic regression (SGD) with all-zero initial parameters."""
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        learning_rate="constant",
        eta0=learning_rate,
        random_state=seed,
        shuffle=True,
    )
    # One dummy step on zero inputs makes sklearn allocate every fitted attribute
    # (classes_, coef_, intercept_, t_ ...); we then reset the weights to zero.
    classes = np.arange(n_classes)
    model.partial_fit(np.zeros((n_classes, n_features)), classes, classes=classes)
    model.coef_ = np.zeros_like(model.coef_)
    model.intercept_ = np.zeros_like(model.intercept_)
    return model


def get_model_params(model: SGDClassifier) -> NDArrays:
    return [model.coef_.copy(), model.intercept_.copy()]


def set_model_params(model: SGDClassifier, params: NDArrays) -> SGDClassifier:
    model.coef_ = np.array(params[0], dtype=np.float64, order="C", copy=True)
    model.intercept_ = np.array(params[1], dtype=np.float64, order="C", copy=True)
    return model


def train_local(model: SGDClassifier, X, y, epochs: int = 1) -> None:
    """Run ``epochs`` SGD passes over the local data, continuing from current weights."""
    for _ in range(max(1, int(epochs))):
        model.partial_fit(X, y, classes=model.classes_)


# ---------------------------------------------------------------------- evaluation
def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Attack-vs-benign detection metrics (class 0 = BENIGN).

    ``accuracy`` is over the original labels; precision/recall/F1/FPR treat any
    non-zero class as "attack". ``fpr`` = share of benign flows flagged as attacks.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    t, p = y_true != 0, y_pred != 0
    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    tn = int(np.sum(~t & ~p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / (fp + tn) if fp + tn else 0.0),
    }


def evaluate_model(model: SGDClassifier, X, y) -> tuple[float, dict[str, float]]:
    """Return ``(log_loss, metrics)`` for a fitted model."""
    proba = model.predict_proba(X)
    pred = model.classes_[np.argmax(proba, axis=1)]
    loss = float(log_loss(y, proba, labels=model.classes_))
    return loss, compute_metrics(y, pred)