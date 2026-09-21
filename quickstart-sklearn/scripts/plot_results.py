#!/usr/bin/env python
"""Turn results/runs/*/rounds.csv (+ results/centralized/metrics.csv) into figures and tables.

    python scripts/plot_results.py

Outputs
    results/summary.csv            one row per run (final metrics, overhead, convergence)
    results/summary.md             comparison tables (centralised vs federated vs defended)
    results/figures/convergence.png
    results/figures/attack_<attack>.png   F1 / recall / FPR vs % malicious clients
    results/figures/overhead.png          aggregation time and communication cost
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GROUP = ["aggregation", "attack", "malicious_fraction", "clients", "partition_alpha"]
COLORS = {"fedavg": "#d62728", "median": "#1f77b4", "trimmed_mean": "#2ca02c",
          "krum": "#9467bd", "multikrum": "#ff7f0e"}


def load_runs(runs_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(runs_dir.glob("*/rounds.csv")):
        df = pd.read_csv(path)
        if len(df) > 1:
            frames.append(df)
    if not frames:
        raise SystemExit(f"No round logs found in {runs_dir}. Run scripts/run_experiments.py first.")
    df = pd.concat(frames, ignore_index=True)
    df["clients"] = df.groupby("run_name")["clients"].transform("max").astype(int)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby("run_name"):
        g = g.sort_values("round")
        last = g.iloc[-1]
        final_f1 = last["f1"]
        hit = g[g["f1"] >= 0.95 * final_f1]["round"]
        trained = g[g["round"] > 0]
        rows.append({
            **{k: last[k] for k in GROUP + ["run_name", "seed"]},
            "rounds": int(last["round"]),
            **{m: last[m] for m in ["accuracy", "precision", "recall", "f1", "fpr", "loss"]},
            "rounds_to_95pct_final_f1": int(hit.iloc[0]) if final_f1 > 0.05 and len(hit) else np.nan,
            "mean_agg_time_ms": 1000 * trained["agg_time_s"].mean(),
            "total_comm_MB": (trained["bytes_up"] + trained["bytes_down"]).sum() / 1e6,
            "wall_time_s": last["elapsed_s"],
            "malicious_in_round": trained["malicious_in_round"].mean(),
            "malicious_kept": trained["malicious_kept"].mean() if "malicious_kept" in trained else np.nan,
        })
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def mean_std(g: pd.Series, digits: int = 4) -> str:
    if len(g) > 1:
        return f"{g.mean():.{digits}f} ± {g.std():.{digits}f}"
    return f"{g.iloc[0]:.{digits}f}"


def pivot(summary: pd.DataFrame, attack: str, metric: str, clients: int) -> pd.DataFrame:
    sub = summary[(summary["clients"] == clients) & (summary["attack"].isin(["none", attack]))]
    table = sub.pivot_table(index="malicious_fraction", columns="aggregation", values=metric, aggfunc=mean_std)
    table.index = [f"{int(round(100 * i))}%" for i in table.index]
    return table.reset_index().rename(columns={"index": "malicious"})


def plot_convergence(df: pd.DataFrame, central: pd.DataFrame | None, out: Path) -> None:
    clean = df[(df["attack"] == "none")]
    if clean.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (agg, n), g in clean.groupby(["aggregation", "clients"]):
        m = g.groupby("round")["f1"].mean()
        ax.plot(m.index, m.values, marker="o", ms=3, color=COLORS.get(agg), label=f"{agg}, {n} clients",
                ls="-" if n == clean["clients"].min() else "--")
    if central is not None:
        for _, r in central.iterrows():
            ax.axhline(r["f1"], ls=":", lw=1, color="gray")
            ax.text(clean["round"].max(), r["f1"], f" {r['model']}", va="center", fontsize=7, color="gray")
    ax.set_xlabel("federated round")
    ax.set_ylabel("global test F1 (attack detection)")
    ax.set_title("Convergence without attackers (dotted = centralised baselines)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=150)
    plt.close(fig)


def plot_attack(summary: pd.DataFrame, attack: str, clients: int, central: pd.DataFrame | None, out: Path) -> None:
    sub = summary[(summary["clients"] == clients) & (summary["attack"].isin(["none", attack]))]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (metric, label) in zip(axes, [("f1", "F1"), ("recall", "attack recall (detection rate)"),
                                          ("fpr", "false positive rate")]):
        for agg, g in sub.groupby("aggregation"):
            stat = g.groupby("malicious_fraction")[metric].agg(["mean", "std"]).fillna(0)
            ax.errorbar(100 * stat.index, stat["mean"], yerr=stat["std"], marker="o", capsize=3,
                        color=COLORS.get(agg), label=agg)
        if central is not None and metric in central:
            best = central.loc[central["f1"].idxmax()]
            ax.axhline(best[metric], ls=":", color="gray", label=f"centralised ({best['model']})")
        ax.set_xlabel("% malicious clients")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Effect of {attack.replace('_', ' ')} attack ({clients} clients)")
    fig.tight_layout()
    fig.savefig(out / f"attack_{attack}.png", dpi=150)
    plt.close(fig)


def plot_overhead(summary: pd.DataFrame, out: Path) -> None:
    clean = summary[summary["attack"] == "none"]
    if clean.empty:
        return
    g = clean.groupby("aggregation")[["mean_agg_time_ms", "total_comm_MB"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, col, title in zip(axes, g.columns, ["Mean aggregation time per round (ms)", "Total communication (MB)"]):
        ax.bar(g.index, g[col], color=[COLORS.get(a, "gray") for a in g.index])
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "overhead.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    res = args.results_dir
    fig_dir = res / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(res / "runs")
    summary = summarise(df)
    summary.to_csv(res / "summary.csv", index=False)
    central_path = res / "centralized" / "metrics.csv"
    central = pd.read_csv(central_path) if central_path.exists() else None

    plot_convergence(df, central, fig_dir)
    plot_overhead(summary, fig_dir)

    attacks = [a for a in summary["attack"].unique() if a != "none"]
    lines = ["# Results summary\n"]
    if central is not None:
        lines += ["## Centralised baselines\n", md_table(central.round(4)), ""]

    clean = summary[summary["attack"] == "none"]
    if not clean.empty:
        show = clean.groupby(["clients", "aggregation"]).agg(
            f1=("f1", mean_std), fpr=("fpr", mean_std), recall=("recall", mean_std),
            rounds_to_95pct=("rounds_to_95pct_final_f1", "mean"),
            agg_ms=("mean_agg_time_ms", "mean"), comm_MB=("total_comm_MB", "mean"),
        ).reset_index()
        lines += ["## Federated, no attackers\n", md_table(show.round(3)), ""]

    for attack in attacks:
        clients = int(summary[summary["attack"] == attack]["clients"].max())
        plot_attack(summary, attack, clients, central, fig_dir)
        lines.append(f"## {attack.replace('_', ' ').title()} attack ({clients} clients)\n")
        for metric in ("f1", "recall", "fpr"):
            lines += [f"**{metric}** (mean ± std over seeds; rows = % malicious clients)\n",
                      md_table(pivot(summary, attack, metric, clients)), ""]
        kept = summary[(summary["attack"] == attack) & summary["aggregation"].isin(["krum", "multikrum"])]
        if not kept.empty:
            t = kept.groupby(["aggregation", "malicious_fraction"]).agg(
                malicious_per_round=("malicious_in_round", "mean"), admitted_by_krum=("malicious_kept", "mean"))
            lines += ["**Attackers admitted into the Krum aggregate (mean per round)**\n", md_table(t.reset_index().round(2)), ""]

    (res / "summary.md").write_text("\n".join(lines))
    print(f"Wrote {res / 'summary.csv'}, {res / 'summary.md'} and figures in {fig_dir}")


if __name__ == "__main__":
    main()
