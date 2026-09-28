#!/usr/bin/env python
"""Merge every Stage 1 result into ONE comparison table (hand-off to Checkpoint 5 / TM2).

    python scripts/build_benchmark.py

Sources
    legacy/results/final_model_benchmark.csv   TM2's earlier single-machine experiments
    results/centralized/benchmark.csv          Checkpoint 3 (this repo's centralized baselines)
    results/runs/*/rounds.csv                  Checkpoint 4 (final round of each clean federated run)

Output
    results/stage1_comparison.csv   Model / Source / Split / Accuracy / Balanced Accuracy / Macro F1 / Weighted F1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ["Model", "Source", "Split", "Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1"]


def legacy_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        {"Model": r["Model"], "Source": "legacy", "Split": "test",
         "Accuracy": r["Test Accuracy"], "Balanced Accuracy": r["Test Balanced Accuracy"],
         "Macro F1": r["Test Macro F1"], "Weighted F1": r["Test Weighted F1"]}
        for _, r in df.iterrows()
    ]


def centralized_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df["Source"] = "checkpoint3"
    return df.assign(Split=df["Split"].str.lower())[COLS].to_dict("records")


def federated_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(runs_dir.glob("*/rounds.csv")):
        df = pd.read_csv(f)
        if df.empty or str(df["attack"].iloc[-1]) != "none":
            continue  # attacked runs belong to Stage 2
        last = df.iloc[-1]
        if "balanced_accuracy" not in df.columns:
            continue  # produced before the metrics upgrade - re-run it
        n = int(last["clients"]) if pd.notna(last.get("clients")) else "?"
        alpha = float(last["partition_alpha"])
        dist = "IID" if alpha <= 0 else f"non-IID a={alpha:g}"
        rows.append({
            "Model": f"Federated logistic_sgd ({last['aggregation']}, {n} clients, {dist}, {int(last['round'])} rounds)",
            "Source": "checkpoint4", "Split": "test", "Accuracy": last["accuracy"],
            "Balanced Accuracy": last["balanced_accuracy"], "Macro F1": last["macro_f1"],
            "Weighted F1": last["weighted_f1"],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--legacy", type=Path, default=ROOT / "legacy" / "results" / "final_model_benchmark.csv")
    args = ap.parse_args()

    rows = (legacy_rows(args.legacy)
            + centralized_rows(args.results_dir / "centralized" / "benchmark.csv")
            + federated_rows(args.results_dir / "runs"))
    if not rows:
        raise SystemExit("Nothing to merge - run centralized_baseline.py and a federated run first.")
    out = pd.DataFrame(rows)[COLS]
    out_path = args.results_dir / "stage1_comparison.csv"
    out.to_csv(out_path, index=False)
    print(out.round(4).to_string(index=False))
    print(f"\nSaved to {out_path}")
    print("Reminder: only compare rows trained on the SAME data (same splits / label mode).")


if __name__ == "__main__":
    main()
