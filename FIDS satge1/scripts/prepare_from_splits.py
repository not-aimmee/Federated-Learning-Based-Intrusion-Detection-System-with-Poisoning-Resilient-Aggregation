#!/usr/bin/env python
"""Turn the team's split CSVs (Checkpoint 2 output) into the arrays used by Checkpoints 3 and 4.

Input  (default)  data/splits/train_group_aware.csv, validation_group_aware.csv, test_group_aware.csv
                  columns: Row_ID, Label (text), <feature columns...>
Output            data/processed/X_{train,val,test}.npy, y_{train,val,test}.npy, meta.json

Why this exists: the centralised baseline (scripts/centralized_baseline.py) and the federated
system (Flower app) both read data/processed/. Building it from *the same split files* the
legacy experiments used guarantees every model sees identical train/validation/test rows, so
the Checkpoint 5 comparison is fair.

    python scripts/prepare_from_splits.py                          # multiclass, group-aware splits
    python scripts/prepare_from_splits.py --label-mode binary      # BENIGN vs ATTACK
    python scripts/prepare_from_splits.py --sample-frac 0.1        # quick development subset
    python scripts/prepare_from_splits.py --suffix ""              # use the plain (non group-aware) splits
    python scripts/prepare_from_splits.py --transform standard     # legacy StandardScaler only

Encoding: BENIGN is always class 0 (the metrics code relies on that); attack classes are
numbered 1..k in alphabetical order. Feature scaling statistics come from the TRAIN split only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID_COL, LABEL_COL = "Row_ID", "Label"
SPLITS = (("train", "train"), ("val", "validation"), ("test", "test"))


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--suffix", default="_group_aware", help="file suffix: <split><suffix>.csv")
    ap.add_argument("--label-mode", choices=["multiclass", "binary"], default="multiclass")
    ap.add_argument(
        "--transform",
        choices=["signedlog", "standard"],
        default="signedlog",
        help="signedlog = sign(x)*log1p|x|, standardise, clip to +-10 (robust for SGD; default)\n"
        "standard  = plain StandardScaler (what legacy/federated_evaluation.py used)",
    )
    ap.add_argument("--sample-frac", type=float, default=1.0, help="stratified subsample of every split")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------- helpers
def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Checkpoint 2 (TM2) must produce data/splits/{train,validation,test}<suffix>.csv first.\n"
            "No real data yet? Run scripts/make_synthetic_data.py for a synthetic stand-in."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for col in (LABEL_COL,):
        if col not in df.columns:
            raise KeyError(f"{path.name}: missing required column '{col}' (found {list(df.columns)[:5]}...)")
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    return df


def subsample(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if frac >= 1.0:
        return df
    parts = []
    for _, g in df.groupby(LABEL_COL, sort=False):
        parts.append(g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed))
    return pd.concat(parts).sort_index()


def build_label_map(labels: pd.Series, mode: str) -> dict[str, int]:
    names = sorted(labels.unique())
    benign = [n for n in names if n.upper() == "BENIGN"]
    if not benign:
        raise ValueError(f"No BENIGN label among {names}; class 0 must be BENIGN.")
    if mode == "binary":
        return {n: (0 if n.upper() == "BENIGN" else 1) for n in names}
    attacks = [n for n in names if n.upper() != "BENIGN"]
    return {benign[0]: 0, **{n: i + 1 for i, n in enumerate(attacks)}}


def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def main(argv=None) -> None:
    args = parse_args(argv)
    frames = {}
    for key, fname in SPLITS:
        df = load_split(args.splits_dir / f"{fname}{args.suffix}.csv")
        frames[key] = subsample(df, args.sample_frac, args.seed)
        print(f"loaded {key:<5} {frames[key].shape}")

    # ---- leakage checks (the reason the group-aware splits exist) -------------------------
    if all(ID_COL in f.columns for f in frames.values()):
        ids = {k: set(f[ID_COL]) for k, f in frames.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = len(ids[a] & ids[b])
            if overlap:
                raise ValueError(f"{overlap} Row_IDs appear in both {a} and {b} - splits leak.")
        print("PASS: no Row_ID overlap between train / validation / test")

    drop = [c for c in (ID_COL, LABEL_COL) if c in frames["train"].columns]
    feature_names = [c for c in frames["train"].columns if c not in drop]
    for k, f in frames.items():
        if [c for c in f.columns if c not in drop] != feature_names:
            raise ValueError(f"Feature columns of '{k}' differ from train - cannot align.")

    h = {k: pd.util.hash_pandas_object(f[feature_names], index=False) for k, f in frames.items()}
    dup = int(h["test"].isin(set(h["train"])).sum())
    print(f"info: {dup} test rows have a feature vector identical to a train row "
          f"({dup / max(1, len(frames['test'])):.2%}) - expected ~0 for group-aware splits")

    # ---- labels ---------------------------------------------------------------------------
    label_map = build_label_map(pd.concat([f[LABEL_COL] for f in frames.values()]), args.label_mode)
    unseen = set(frames["val"][LABEL_COL]) | set(frames["test"][LABEL_COL])
    unseen -= set(frames["train"][LABEL_COL])
    if unseen:
        print(f"warning: labels never seen in train: {sorted(unseen)}")
    y = {k: f[LABEL_COL].map(label_map).to_numpy(dtype=np.int64) for k, f in frames.items()}
    n_classes = int(max(label_map.values())) + 1
    class_names = ["BENIGN"] if args.label_mode == "multiclass" else ["BENIGN", "ATTACK"]
    if args.label_mode == "multiclass":
        class_names += [n for n, i in sorted(label_map.items(), key=lambda kv: kv[1]) if i > 0]

    # ---- features -------------------------------------------------------------------------
    X = {k: f[feature_names].to_numpy(dtype=np.float64) for k, f in frames.items()}
    n_bad = 0
    for k in X:
        bad = ~np.isfinite(X[k])
        n_bad += int(bad.sum())
        X[k][bad] = 0.0
    if n_bad:
        print(f"warning: replaced {n_bad} inf/NaN feature values with 0")

    if args.transform == "signedlog":
        X = {k: signed_log(v) for k, v in X.items()}
    mean = X["train"].mean(axis=0)
    std = X["train"].std(axis=0)
    keep = std > 1e-12  # constant-in-train features carry no signal and break scaling
    dropped = [n for n, k in zip(feature_names, keep) if not k]
    X = {k: (v[:, keep] - mean[keep]) / std[keep] for k, v in X.items()}
    if args.transform == "signedlog":
        X = {k: np.clip(v, -10.0, 10.0) for k, v in X.items()}
    feature_names = [n for n, k in zip(feature_names, keep) if k]
    print(f"features: kept {len(feature_names)}, dropped {len(dropped)} constant: {dropped}")

    # ---- save -----------------------------------------------------------------------------
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for k in X:
        np.save(out / f"X_{k}.npy", X[k].astype(np.float32))
        np.save(out / f"y_{k}.npy", y[k])
    meta = {
        "source": "cicids2017_splits",
        "splits_suffix": args.suffix,
        "label_mode": args.label_mode,
        "transform": args.transform,
        "sample_frac": args.sample_frac,
        "n_features": len(feature_names),
        "n_classes": n_classes,
        "class_names": class_names,
        "feature_names": feature_names,
        "dropped_constant_features": dropped,
        "n_train": int(len(y["train"])),
        "n_val": int(len(y["val"])),
        "n_test": int(len(y["test"])),
        "class_counts": {
            k: {class_names[c]: int(n) for c, n in zip(*np.unique(v, return_counts=True))}
            for k, v in y.items()
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved to {out}  (train {len(y['train'])}, val {len(y['val'])}, test {len(y['test'])}, "
          f"{n_classes} classes)")
    print("Next: python scripts/centralized_baseline.py   (Checkpoint 3)")


if __name__ == "__main__":
    sys.exit(main())
