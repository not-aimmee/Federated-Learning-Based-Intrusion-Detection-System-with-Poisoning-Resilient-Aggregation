"""Tests for the Stage 1 data bridge and the extended metrics (no Flower needed)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearnexample.task import compute_metrics, create_model  # noqa: E402
import prepare_from_splits  # noqa: E402
import build_benchmark  # noqa: E402


def _write_splits(dir_: Path, leak: bool = False) -> None:
    rng = np.random.default_rng(1)
    labels = ["BENIGN", "DoS", "PortScan"]

    def make(n, start):
        df = pd.DataFrame(rng.lognormal(2, 1, (n, 4)), columns=["a", "b", "c", "d"])
        df["k"] = 7.0  # constant feature -> must be dropped
        df["Label"] = rng.choice(labels, n)
        df.insert(0, "Row_ID", np.arange(start, start + n))
        return df

    make(300, 0).to_csv(dir_ / "train_group_aware.csv", index=False)
    make(100, 0 if leak else 1000).to_csv(dir_ / "validation_group_aware.csv", index=False)
    make(100, 2000).to_csv(dir_ / "test_group_aware.csv", index=False)


def test_metrics_multiclass_and_binary_views():
    y = np.array([0, 0, 1, 2, 2, 1])
    p = np.array([0, 1, 1, 2, 0, 1])
    m = compute_metrics(y, p)
    assert m["accuracy"] == pytest.approx(4 / 6)
    for k in ("balanced_accuracy", "macro_f1", "weighted_f1", "precision", "recall", "f1", "fpr"):
        assert 0.0 <= m[k] <= 1.0
    assert m["fpr"] == pytest.approx(0.5)  # 1 of 2 benign flagged as attack


def test_create_model_uses_balanced_weights_with_partial_fit(tmp_path):
    y_train = np.array([0, 0, 0, 1, 1])
    np.save(tmp_path / "X_train.npy", np.zeros((len(y_train), 2)))
    np.save(tmp_path / "y_train.npy", y_train)
    np.save(tmp_path / "X_test.npy", np.zeros((1, 2)))
    np.save(tmp_path / "y_test.npy", np.array([0]))

    model = create_model(2, 2, data_dir=tmp_path)

    assert model.class_weight == pytest.approx({0: 5 / 6, 1: 1.25})


def test_federated_benchmark_reports_best_and_final_rounds(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    pd.DataFrame([
        {"attack": "none", "round": 0, "aggregation": "fedavg", "clients": 3,
         "partition_alpha": 0, "accuracy": 0.7, "balanced_accuracy": 0.3,
         "macro_f1": 0.2, "weighted_f1": 0.6},
        {"attack": "none", "round": 1, "aggregation": "fedavg", "clients": 3,
         "partition_alpha": 0, "accuracy": 0.8, "balanced_accuracy": 0.6,
         "macro_f1": 0.5, "weighted_f1": 0.75},
        {"attack": "none", "round": 2, "aggregation": "fedavg", "clients": 3,
         "partition_alpha": 0, "accuracy": 0.85, "balanced_accuracy": 0.55,
         "macro_f1": 0.45, "weighted_f1": 0.8},
    ]).to_csv(run_dir / "rounds.csv", index=False)

    rows = build_benchmark.federated_rows(tmp_path)

    assert [row["Selection"] for row in rows] == ["final", "best_test_macro_f1"]
    assert [row["Round"] for row in rows] == [2, 1]
    assert [row["Macro F1"] for row in rows] == [0.45, 0.5]


def test_prepare_from_splits_multiclass(tmp_path):
    _write_splits(tmp_path)
    out = tmp_path / "processed"
    prepare_from_splits.main(["--splits-dir", str(tmp_path), "--out-dir", str(out)])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_classes"] == 3 and meta["class_names"][0] == "BENIGN"
    assert meta["n_features"] == 4 and "k" in meta["dropped_constant_features"]
    X, y = np.load(out / "X_train.npy"), np.load(out / "y_train.npy")
    assert X.shape == (300, 4) and set(np.unique(y)) <= {0, 1, 2}
    assert abs(X.mean()) < 0.1  # standardised with train statistics
    assert np.load(out / "X_val.npy").shape[1] == np.load(out / "X_test.npy").shape[1] == 4


def test_prepare_from_splits_binary(tmp_path):
    _write_splits(tmp_path)
    out = tmp_path / "processed"
    prepare_from_splits.main(["--splits-dir", str(tmp_path), "--out-dir", str(out), "--label-mode", "binary"])
    assert set(np.unique(np.load(out / "y_train.npy"))) == {0, 1}


def test_prepare_from_splits_rejects_leaky_splits(tmp_path):
    _write_splits(tmp_path, leak=True)
    with pytest.raises(ValueError, match="leak"):
        prepare_from_splits.main(["--splits-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")])
