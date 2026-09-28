#!/usr/bin/env python
"""Clean and feature-engineer CICIDS2017 (the 8 "MachineLearningCVE" CSV files).

Put the CSVs in ``data/raw/`` and run::

    python scripts/preprocess_cicids2017.py                     # full dataset
    python scripts/preprocess_cicids2017.py --sample-frac 0.2   # faster, for development
    python scripts/preprocess_cicids2017.py --label-mode multiclass --top-k 40

Pipeline (phase 1.2 of the project plan)
    1. load + strip column names, fix mangled label characters
    2. replace +/-inf with NaN, drop NaN rows and exact duplicate rows
    3. stratified train/test split (test set keeps the natural class distribution)
    4. drop constant features and highly correlated features (fit on train only)
    5. signed log1p transform (network features are extremely heavy-tailed),
       standardise with train statistics, clip to [-10, 10]
    6. optional: keep the top-k features by random-forest importance
    7. optional: undersample BENIGN in the *training* set to tame class imbalance
    8. save X/y .npy files + meta.json to data/processed/, EDA plots to reports/eda/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--eda-dir", type=Path, default=ROOT / "reports" / "eda")
    ap.add_argument("--label-mode", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--sample-frac", type=float, default=1.0, help="stratified subsample of the cleaned data")
    ap.add_argument("--corr-threshold", type=float, default=0.95, help="drop one of each feature pair above this |corr|")
    ap.add_argument("--top-k", type=int, default=0, help="keep only the k most important features (0 = keep all)")
    ap.add_argument("--undersample-ratio", type=float, default=3.0,
                    help="max BENIGN:attack ratio in the TRAIN set (<=0 disables)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-eda", action="store_true")
    return ap.parse_args()


# ------------------------------------------------------------------------ loading
def load_and_clean(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        sys.exit(
            f"No CSV files found in {raw_dir}.\n"
            "Download CICIDS2017 (MachineLearningCVE.zip from the UNB CIC page or Kaggle) and "
            "extract the CSVs there, or use scripts/make_synthetic_data.py for test data."
        )
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df.columns = df.columns.str.strip()
        print(f"  loaded {f.name}: {len(df):,} rows")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop(columns=[c for c in df.columns if c.endswith(".1")])  # duplicated column

    stats: dict = {"rows_raw": int(len(df))}
    df["Label"] = df["Label"].astype(str).map(lambda s: re.sub(r"[^\x00-\x7F]+", "-", s).strip())
    feats = [c for c in df.columns if c != "Label"]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    stats["rows_with_nan_or_inf"] = int(df[feats].isna().any(axis=1).sum())
    df = df.dropna(subset=feats)
    stats["rows_after_dropna"] = int(len(df))
    df = df.drop_duplicates()
    stats["rows_after_dedup"] = int(len(df))
    df[feats] = df[feats].astype(np.float32)
    return df.reset_index(drop=True), stats


def stratified_sample(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if frac >= 1.0:
        return df
    def pick(g: pd.DataFrame) -> pd.DataFrame:
        n = min(len(g), max(int(round(len(g) * frac)), 20))  # never wipe out rare attacks
        return g.sample(n=n, random_state=seed)
    return df.groupby("Label", group_keys=False).apply(pick).reset_index(drop=True)


# -------------------------------------------------------------------- transforms
def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def select_features(X: pd.DataFrame, corr_threshold: float, seed: int) -> tuple[list[str], dict]:
    const = [c for c in X.columns if X[c].max() == X[c].min()]
    keep = [c for c in X.columns if c not in const]
    sample = X[keep].sample(min(len(X), 200_000), random_state=seed)
    corr = sample.apply(signed_log1p).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    correlated = [c for c in upper.columns if (upper[c] > corr_threshold).any()]
    final = [c for c in keep if c not in correlated]
    return final, {"dropped_constant": const, "dropped_correlated": correlated}


# ------------------------------------------------------------------------- EDA
def make_eda(out: Path, labels: pd.Series, feats_scaled: np.ndarray, names: list[str],
             y_bin: np.ndarray, importances: np.ndarray | None) -> None:
    out.mkdir(parents=True, exist_ok=True)

    counts = labels.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(counts) + 1.5))
    ax.barh(counts.index, counts.values, color="#3b6ea5")
    ax.set_xscale("log")
    ax.set_xlabel("flows (log scale)")
    ax.set_title("CICIDS2017 - class distribution after cleaning")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=150)
    plt.close(fig)

    df = pd.DataFrame(feats_scaled, columns=names)
    corr_with_label = df.corrwith(pd.Series(y_bin)).abs().sort_values(ascending=False)
    top = list(corr_with_label.index[:25])
    cm = df[top].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top)), top, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top)), top, fontsize=7)
    ax.set_title("Feature correlations (top 25 features by |corr| with attack label)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "feature_correlation.png", dpi=150)
    plt.close(fig)

    best = list(corr_with_label.index[:6])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, feat in zip(axes.ravel(), best):
        ax.hist(df.loc[y_bin == 0, feat].clip(-5, 5), bins=60, alpha=0.6, label="benign", density=True)
        ax.hist(df.loc[y_bin == 1, feat].clip(-5, 5), bins=60, alpha=0.6, label="attack", density=True)
        ax.set_title(feat, fontsize=9)
    axes[0, 0].legend()
    fig.suptitle("Most attack-discriminative features (standardised, log-scaled)")
    fig.tight_layout()
    fig.savefig(out / "top_feature_distributions.png", dpi=150)
    plt.close(fig)

    if importances is not None:
        order = np.argsort(importances)[-20:]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(np.array(names)[order], importances[order], color="#3b6ea5")
        ax.set_title("Random-forest feature importance (top 20)")
        fig.tight_layout()
        fig.savefig(out / "feature_importance.png", dpi=150)
        plt.close(fig)


# -------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("Loading and cleaning ...")
    df, stats = load_and_clean(args.raw_dir)
    df = stratified_sample(df, args.sample_frac, args.seed)
    stats["rows_used"] = int(len(df))

    names_multi = ["BENIGN"] + sorted(set(df["Label"]) - {"BENIGN"})
    multi = df["Label"].map({n: i for i, n in enumerate(names_multi)}).to_numpy()
    if args.label_mode == "binary":
        y_all, class_names = (multi != 0).astype(np.int64), ["BENIGN", "ATTACK"]
    else:
        y_all, class_names = multi.astype(np.int64), names_multi
    stats["class_counts_cleaned"] = {n: int((multi == i).sum()) for i, n in enumerate(names_multi)}

    feats = [c for c in df.columns if c != "Label"]
    X_df = df[feats]

    print("Splitting ...")
    idx = np.arange(len(df))
    try:
        tr, te = train_test_split(idx, test_size=args.test_size, stratify=multi, random_state=args.seed)
    except ValueError:  # a class too small to stratify on
        tr, te = train_test_split(idx, test_size=args.test_size, stratify=(multi != 0), random_state=args.seed)

    print("Selecting features (train only) ...")
    kept, dropped = select_features(X_df.iloc[tr], args.corr_threshold, args.seed)
    stats.update(dropped)

    Xtr = signed_log1p(X_df.iloc[tr][kept].to_numpy(np.float64))
    Xte = signed_log1p(X_df.iloc[te][kept].to_numpy(np.float64))
    mean, std = Xtr.mean(0), Xtr.std(0)
    std[std == 0] = 1.0
    Xtr = np.clip((Xtr - mean) / std, -10, 10)
    Xte = np.clip((Xte - mean) / std, -10, 10)
    ytr, yte = y_all[tr], y_all[te]

    importances = None
    if args.top_k and args.top_k < len(kept):
        print(f"Ranking features with a random forest (keeping top {args.top_k}) ...")
        sub = rng.choice(len(Xtr), size=min(len(Xtr), 200_000), replace=False)
        rf = RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=args.seed)
        rf.fit(Xtr[sub], ytr[sub])
        order = np.argsort(rf.feature_importances_)[::-1][: args.top_k]
        order = np.sort(order)
        importances = rf.feature_importances_[order]
        stats["dropped_low_importance"] = [kept[i] for i in range(len(kept)) if i not in set(order)]
        kept = [kept[i] for i in order]
        Xtr, Xte, mean, std = Xtr[:, order], Xte[:, order], mean[order], std[order]

    if not args.no_eda:
        print("Writing EDA plots ...")
        eda_sub = rng.choice(len(Xtr), size=min(len(Xtr), 100_000), replace=False)
        make_eda(args.eda_dir, df["Label"], Xtr[eda_sub], kept, (ytr[eda_sub] != 0).astype(int), importances)

    if args.undersample_ratio > 0:
        benign, attack = np.flatnonzero(ytr == 0), np.flatnonzero(ytr != 0)
        keep_benign = min(len(benign), int(args.undersample_ratio * len(attack)))
        chosen = np.sort(np.concatenate([attack, rng.choice(benign, keep_benign, replace=False)]))
        Xtr, ytr = Xtr[chosen], ytr[chosen]
        stats["benign_undersampled_to"] = int(keep_benign)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "X_train.npy", Xtr.astype(np.float32))
    np.save(args.out_dir / "X_test.npy", Xte.astype(np.float32))
    np.save(args.out_dir / "y_train.npy", ytr.astype(np.int64))
    np.save(args.out_dir / "y_test.npy", yte.astype(np.int64))
    meta = {
        "source": "CICIDS2017",
        "label_mode": args.label_mode,
        "n_features": len(kept),
        "n_classes": len(class_names),
        "class_names": class_names,
        "feature_names": kept,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "train_class_counts": {class_names[i]: int((ytr == i).sum()) for i in range(len(class_names))},
        "test_class_counts": {class_names[i]: int((yte == i).sum()) for i in range(len(class_names))},
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    args.eda_dir.mkdir(parents=True, exist_ok=True)
    (args.eda_dir / "cleaning_summary.json").write_text(json.dumps(stats, indent=2))

    print(f"\nDone. {meta['n_train']:,} train / {meta['n_test']:,} test rows, {len(kept)} features.")
    print(f"  train class counts: {meta['train_class_counts']}")
    print(f"  saved to {args.out_dir}  |  EDA + cleaning summary in {args.eda_dir}")


if __name__ == "__main__":
    main()
