#!/usr/bin/env python
"""Centralised (non-federated) IDS baselines - the benchmark for everything else.

    python scripts/centralized_baseline.py
    python scripts/centralized_baseline.py --models logistic_sgd random_forest --max-train-rows 500000

Models
    logistic_sgd           same model family as the federated system (fair comparison)
    random_forest          strong classic IDS baseline
    hist_gradient_boosting gradient-boosting baseline

Writes results/centralized/metrics.csv and metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearnexample.config import data_dir, results_dir  # noqa: E402
from sklearnexample.task import compute_metrics, create_model, train_local  # noqa: E402

MODELS = ("logistic_sgd", "random_forest", "hist_gradient_boosting")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--sgd-epochs", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--max-train-rows", type=int, default=0, help="subsample training rows (0 = all)")
    ap.add_argument("--data-dir", type=Path, default=data_dir(ROOT))
    ap.add_argument("--out-dir", type=Path, default=results_dir(ROOT) / "centralized")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = args.data_dir
    meta = json.loads((d / "meta.json").read_text())
    X_tr, y_tr = np.load(d / "X_train.npy"), np.load(d / "y_train.npy")
    X_te, y_te = np.load(d / "X_test.npy"), np.load(d / "y_test.npy")
    if args.max_train_rows and len(X_tr) > args.max_train_rows:
        keep = np.random.default_rng(args.seed).choice(len(X_tr), args.max_train_rows, replace=False)
        X_tr, y_tr = X_tr[keep], y_tr[keep]
    print(f"train {X_tr.shape}, test {X_te.shape}, classes={meta['class_names']}")

    rows = []
    for name in args.models:
        print(f"\n== {name}")
        t0 = time.perf_counter()
        if name == "logistic_sgd":
            model = create_model(meta["n_features"], meta["n_classes"], args.learning_rate, args.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_local(model, X_tr.astype(np.float64), y_tr, args.sgd_epochs)
        elif name == "random_forest":
            model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=args.seed)
            model.fit(X_tr, y_tr)
        else:
            model = HistGradientBoostingClassifier(random_state=args.seed)
            model.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = model.predict(X_te.astype(np.float64) if name == "logistic_sgd" else X_te)
        predict_time = time.perf_counter() - t0

        row = {"model": name, **compute_metrics(y_te, pred),
               "train_time_s": train_time, "predict_time_s": predict_time}
        rows.append(row)
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "model"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "metrics.csv", index=False)
    (args.out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
