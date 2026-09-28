#!/usr/bin/env python
"""Run a grid of federated experiments with `flwr run` and collect their round logs.

Presets
    stage1   FedAvg, no attack, 3 and 5 clients             (federated baseline)
    stage2   every aggregation x {label_flip, model_poison} x {10,20,30} % malicious,
             10 clients, plus the clean (0 %) reference       (attack + defence study)

Examples
    python scripts/run_experiments.py --preset stage1 --rounds 20
    python scripts/run_experiments.py --preset stage2 --rounds 20 --seeds 42 43 44
    python scripts/run_experiments.py --aggregations fedavg krum --attacks label_flip \\
        --fractions 0.2 --clients 10 --rounds 10
    python scripts/run_experiments.py --preset stage2 --dry-run       # just list the runs

Each run writes results/runs/<run-name>/rounds.csv (see scripts/plot_results.py).
Runs that already finished are skipped unless you pass --force.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearnexample.aggregators import METHODS  # noqa: E402
from sklearnexample.attacks import ATTACKS  # noqa: E402

PRESETS = {
    "stage1": dict(clients=[3, 5], aggregations=["fedavg"], attacks=["none"], fractions=[0.0]),
    "stage2": dict(clients=[10], aggregations=list(METHODS),
                   attacks=["label_flip", "model_poison"], fractions=[0.1, 0.2, 0.3]),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), help="starting point; other flags override it")
    ap.add_argument("--clients", type=int, nargs="+")
    ap.add_argument("--aggregations", nargs="+", choices=METHODS)
    ap.add_argument("--attacks", nargs="+", choices=[a for a in ATTACKS if a != "none"] + ["none"])
    ap.add_argument("--fractions", type=float, nargs="+", help="malicious-client shares, e.g. 0.1 0.2 0.3")
    ap.add_argument("--no-clean-reference", action="store_true", help="skip the 0 %% malicious run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.0, help="Dirichlet non-IID strength (0 = IID)")
    ap.add_argument("--poison-scale", type=float, default=3.0)
    ap.add_argument("--cpus-per-client", type=int, default=0, help="Ray CPUs per simulated client (0 = leave as is)")
    ap.add_argument("--force", action="store_true", help="re-run finished experiments")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = PRESETS.get(args.preset or "", {})
    for key in ("clients", "aggregations", "attacks", "fractions"):
        if getattr(args, key) is None:
            setattr(args, key, base.get(key))
    if not args.clients or not args.aggregations:
        ap.error("choose --preset or give --clients and --aggregations")
    args.attacks = args.attacks or ["none"]
    args.fractions = args.fractions or [0.0]
    return args


def build_runs(args: argparse.Namespace) -> list[dict]:
    runs = []
    for n, agg, seed in itertools.product(args.clients, args.aggregations, args.seeds):
        combos: list[tuple[str, float]] = []
        if not args.no_clean_reference or args.attacks == ["none"]:
            combos.append(("none", 0.0))
        for attack, frac in itertools.product([a for a in args.attacks if a != "none"], args.fractions):
            if frac > 0:
                combos.append((attack, frac))
        for attack, frac in combos:
            name = f"{agg}_{attack}_{frac:.2f}_c{n}_s{seed}"
            if args.alpha > 0:
                name += f"_a{args.alpha:g}"
            runs.append({
                "name": name, "clients": n,
                "config": {
                    "num-server-rounds": args.rounds, "local-epochs": args.local_epochs,
                    "learning-rate": args.learning_rate, "partition-alpha": args.alpha,
                    "seed": seed, "attack": attack, "malicious-fraction": frac,
                    "poison-scale": args.poison_scale, "aggregation": agg,
                    "run-name": name, "project-dir": ROOT.as_posix(),
                },
            })
    return runs


def fmt(value) -> str:
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def is_finished(name: str, rounds: int) -> bool:
    path = ROOT / "results" / "runs" / name / "rounds.csv"
    if not path.exists():
        return False
    with path.open() as fh:
        last = list(csv.DictReader(fh))[-1:]
    return bool(last) and int(last[0]["round"]) >= rounds


def main() -> None:
    args = parse_args()
    runs = build_runs(args)
    print(f"{len(runs)} experiment(s) planned")
    if args.dry_run:
        for r in runs:
            print("  ", r["name"])
        return

    flwr = shutil.which("flwr") or str(Path(sys.executable).parent / "flwr")
    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    current_clients = None
    failures = []

    for i, run in enumerate(sorted(runs, key=lambda r: r["clients"]), 1):
        name = run["name"]
        if not args.force and is_finished(name, args.rounds):
            print(f"[{i}/{len(runs)}] skip (done)  {name}")
            continue
        if run["clients"] != current_clients:
            cmd = [flwr, "federation", "simulation-config", "--num-supernodes", str(run["clients"])]
            if args.cpus_per_client:
                cmd += ["--client-resources-num-cpus", str(args.cpus_per_client)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            current_clients = run["clients"]

        cfg = " ".join(f"{k}={fmt(v)}" for k, v in run["config"].items())
        cmd = [flwr, "run", str(ROOT), "--stream", "--run-config", cfg]
        print(f"[{i}/{len(runs)}] running     {name} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        (log_dir / f"{name}.log").write_text(proc.stdout + "\n" + proc.stderr)
        ok = proc.returncode == 0 and is_finished(name, args.rounds)
        print(f"{'ok' if ok else 'FAILED'} ({time.perf_counter() - t0:.0f}s)")
        if not ok:
            failures.append(name)

    if failures:
        print(f"\n{len(failures)} run(s) failed - see results/logs/<run>.log:\n  " + "\n  ".join(failures))
    print("\nNext: python scripts/plot_results.py")


if __name__ == "__main__":
    main()
