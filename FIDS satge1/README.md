# Federated Intrusion Detection with Poisoning-Resilient Aggregation

A network-intrusion detector (CICIDS2017) trained collaboratively by several simulated organisations
with [Flower](https://flower.ai) + scikit-learn, without sharing raw traffic. Stage 1 builds and benchmarks
the system; Stage 2 adds malicious clients and robust aggregation (Krum, Multi-Krum, trimmed mean, median).

## Repository layout

```
FIDS/
├── pyproject.toml            Flower app config + dependencies   (`flwr run .` is run from HERE)
├── sklearnexample/           the federated system (Checkpoint 4; Stage 2 defences already inside)
│   ├── task.py               data loading, partitioning, model, metrics
│   ├── client_app.py         ClientApp - one simulated organisation
│   ├── server_app.py         ServerApp - rounds + global evaluation
│   ├── strategy.py           FedAvg with pluggable aggregation + per-round CSV log
│   ├── aggregators.py        fedavg / median / trimmed_mean / krum / multikrum   (Stage 2)
│   ├── attacks.py            label_flip / model_poison                          (Stage 2)
│   └── config.py             defaults + path resolution
├── scripts/
│   ├── prepare_from_splits.py     splits CSV  -> data/processed/*.npy           (bridge from Checkpoint 2)
│   ├── preprocess_cicids2017.py   raw CSV     -> data/processed/*.npy           (alternative, self-contained)
│   ├── make_synthetic_data.py     fake data for smoke tests
│   ├── centralized_baseline.py    Checkpoint 3 - centralized benchmark
│   ├── run_experiments.py         Checkpoint 4/5 - grid of federated runs
│   ├── build_benchmark.py         merge legacy + centralized + federated into one table
│   └── plot_results.py            summary tables + figures
├── tests/                    pytest: aggregators, attacks, data bridge, metrics
├── data/                     (git-ignored contents)
│   ├── raw/                  original CICIDS2017 CSVs
│   ├── splits/               train/validation/test_group_aware.csv   (from TM2)
│   └── processed/            X_*.npy, y_*.npy, meta.json
├── results/                  (git-ignored) centralized/, runs/<run>/rounds.csv, stage1_comparison.csv, figures/
├── reports/                  (git-ignored) EDA plots
├── docs/                     blueprint PDF, checkpoint 1 write-up, notes
└── legacy/                   original single-machine scripts + result CSVs (kept for reference, not imported)
```

Everything Flower needs sits at the repo root (it was previously nested in `quickstart-sklearn/`), so there is
one project, one `pyproject.toml`, one virtual environment.

## Stage 1 checkpoint map

| # | Checkpoint | Owner | Where it lives | Status |
|---|-----------|-------|----------------|--------|
| 1 | Team Ready | All | `docs/checkpoint1_team_ready.md` | doc written - team ticks the checklist |
| 2 | Data Pipeline Ready | TM2 | `legacy/data_inspection*.py` -> `data/splits/*_group_aware.csv` | done by TM2 |
| 3 | Centralized Baseline | TM1 | `scripts/centralized_baseline.py` -> `results/centralized/` | ready to run |
| 4 | Federated System Working | TM1 + TM3 | `sklearnexample/` + `scripts/run_experiments.py` -> `results/runs/` | ready to run |
| 5 | Stage 1 Complete | TM2 | `scripts/build_benchmark.py`, `scripts/plot_results.py` | inputs prepared |

## Execution steps

**Windows (PowerShell), from the repo root. On macOS/Linux use `source venv/bin/activate`.**

### 0. One-time setup
```powershell
python -m venv venv
venv\Scripts\activate
pip install -e ".[dev]"
pytest                                    # 14 tests should pass
flwr federation simulation-config --num-supernodes 5
flwr federation simulation-config --client-resources-num-cpus 1     # avoids "ActorPool is empty"
```

### 1. Smoke test with synthetic data (2 minutes, no dataset needed)
```powershell
python scripts/make_synthetic_data.py
python scripts/centralized_baseline.py
flwr run . --stream --run-config "num-server-rounds=5 run-name='smoke'"
```

### 2. Put the real data in place
Copy TM2's three files into `data/splits/`:
`train_group_aware.csv`, `validation_group_aware.csv`, `test_group_aware.csv`
(columns: `Row_ID`, `Label`, then features). Then build the arrays every model will share:
```powershell
python scripts/prepare_from_splits.py --sample-frac 0.1      # fast development subset
python scripts/prepare_from_splits.py                        # full data (final numbers)
# BENIGN-vs-ATTACK instead of all classes:   add  --label-mode binary
```
It checks that no `Row_ID` appears in two splits, fits scaling on train only, and writes `meta.json`.

### 3. Checkpoint 3 - centralized baseline (TM1)
```powershell
python scripts/centralized_baseline.py
```
Outputs `results/centralized/{metrics.csv, metrics.json, benchmark.csv}` for validation and test.
`logistic_sgd` is the fair comparison for the federated system (same model family); random forest and
gradient boosting are stronger reference points.

### 4. Checkpoint 4 - federated system (TM1 + TM3)
```powershell
# single run, 5 organisations, IID data
flwr run . --stream --run-config "num-server-rounds=20 run-name='fed_c5_iid'"

# TM3: heterogeneous (non-IID) organisations - the realistic case
flwr run . --stream --run-config "num-server-rounds=20 partition-alpha=0.5 run-name='fed_c5_noniid'"

# the full Stage 1 grid (3 and 5 clients, FedAvg, no attackers), 3 seeds
python scripts/run_experiments.py --preset stage1 --rounds 20 --seeds 42 43 44
python scripts/run_experiments.py --preset stage1 --rounds 20 --alpha 0.5    # non-IID grid
```
Per-round metrics land in `results/runs/<run>/rounds.csv`; the config used is saved beside it.

### 5. Hand-off to Checkpoint 5 (TM2)
```powershell
python scripts/build_benchmark.py         # -> results/stage1_comparison.csv
python scripts/plot_results.py            # -> results/summary.md + results/figures/*.png
```

## Reading the results honestly
- Compare models only when trained on the **same data** (same splits, same `--label-mode`). The `legacy/`
  numbers were produced on a different pipeline (StandardScaler only, `saga`/`lbfgs` solvers), so they are
  context, not a like-for-like benchmark.
- CICIDS2017 is ~80 % BENIGN. Look at **balanced accuracy and macro-F1**, not accuracy.
- Linear models (`logistic_sgd`) will trail random forests; that gap is a property of the model family, not of
  federation. The Stage 1 claim to test is *federated logistic_sgd ~ centralized logistic_sgd*.
- Random forests cannot be averaged across clients, so they are centralized references only.

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| `ActorPool is empty` | `flwr federation simulation-config --client-resources-num-cpus 1` |
| Data/results end up in the wrong folder | add `project-dir='C:/path/to/FIDS'` to `--run-config` |
| `FileNotFoundError ... meta.json` | run step 1 (synthetic) or step 2 (`prepare_from_splits.py`) first |
| Out of memory on full data | use `--sample-frac 0.2`, or fewer clients |
| Changed client count | re-run the `simulation-config --num-supernodes N` command |

See `docs/` for the blueprint and Checkpoint 1 notes; the original Stage 2 usage (attacks/defences) is in
`scripts/run_experiments.py --preset stage2`.
