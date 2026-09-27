#!/usr/bin/env python
"""Generate a small synthetic, imbalanced 'traffic' dataset in the same format as the
real CICIDS2017 pipeline. Use it to test the whole system before the dataset is ready.

    python scripts/make_synthetic_data.py --samples 60000
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=60_000)
    ap.add_argument("--features", type=int, default=30)
    ap.add_argument("--attack-share", type=float, default=0.25)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y = make_classification(
        n_samples=args.samples,
        n_features=args.features,
        n_informative=max(4, args.features // 2),
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[1 - args.attack_share, args.attack_share],
        class_sep=1.0,
        flip_y=0.01,
        random_state=args.seed,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )
    mu, sd = X_tr.mean(0), X_tr.std(0)
    X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_train.npy", X_tr.astype(np.float32))
    np.save(out / "X_test.npy", X_te.astype(np.float32))
    np.save(out / "y_train.npy", y_tr.astype(np.int64))
    np.save(out / "y_test.npy", y_te.astype(np.int64))
    meta = {
        "source": "synthetic",
        "label_mode": "binary",
        "n_features": args.features,
        "n_classes": 2,
        "class_names": ["BENIGN", "ATTACK"],
        "feature_names": [f"f{i}" for i in range(args.features)],
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote synthetic dataset to {out} ({len(y_tr)} train / {len(y_te)} test rows)")


if __name__ == "__main__":
    main()
