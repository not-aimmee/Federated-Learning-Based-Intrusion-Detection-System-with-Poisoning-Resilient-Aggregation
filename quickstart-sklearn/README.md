# Federated Intrusion Detection with Poisoning-Resilient Aggregation

A federated network-intrusion detector (CICIDS2017) built on [Flower](https://flower.ai) and
scikit-learn. Several simulated organisations train one shared classifier without sharing raw
traffic. Stage 2 adds malicious clients and robust aggregation (Krum, Multi-Krum, trimmed mean,
median).

```
sklearnexample/
  task.py          data loading + partitioning, model helpers, metrics (accuracy/precision/recall/F1/FPR)
  config.py        default run settings and path resolution
  client_app.py    ClientApp: local training, plus malicious behaviour when configured
  attacks.py       label_flip and model_poison attacks
  aggregators.py   fedavg, median, trimmed_mean, krum, multikrum (pure numpy)
  strategy.py      Flower strategy with a pluggable aggregator + per-round CSV logging
  server_app.py    ServerApp: runs the rounds and scores the global model on the test set
scripts/
  preprocess_cicids2017.py   clean + feature-engineer + EDA plots       (Phase 1.2)
  make_synthetic_data.py     fake data in the same format, for testing
  centralized_baseline.py    random forest / gradient boosting / logistic baselines (Phase 1.3)
  run_experiments.py         grid of federated runs, stage1 / stage2 presets  (Phases 1.4, 1.5, 2.2, 2.4)
  plot_results.py            summary tables + figures
tests/test_aggregators.py    unit tests for the defences and attacks
```

## 1. Setup

```powershell
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -e ".[dev]"
```

Requires Flower >= 1.36 (message-based `ClientApp` / `ServerApp` API).
Run `pytest` to check the aggregators and attacks.

**One-time simulation settings.** Flower stores the simulation size in its local SuperLink:

```powershell
flwr federation simulation-config --num-supernodes 5      # 5 simulated organisations
# If you see "ActorPool is empty", give each client fewer CPUs:
flwr federation simulation-config --client-resources-num-cpus 1
```

## 2. Try it in 2 minutes with synthetic data

```powershell
python scripts/make_synthetic_data.py
python scripts/centralized_baseline.py
flwr run . --stream
```

## 3. Stage 1 - baseline

**Phase 1.2: data.** Download the CICIDS2017 *MachineLearningCVE* CSVs into `data/raw/`, then:

```powershell
python scripts/preprocess_cicids2017.py --sample-frac 0.2     # development subset
python scripts/preprocess_cicids2017.py                       # full dataset (needs several GB of RAM)
```

Cleaning: strips column names, fixes mangled label characters, removes inf/NaN and duplicate
rows, drops constant and highly correlated features, applies signed-log + standardisation, and
undersamples BENIGN in the *training* split only (test set keeps the natural class balance).
Plots and a cleaning summary land in `reports/eda/`. Useful flags: `--label-mode multiclass`,
`--top-k 40`, `--undersample-ratio 3`.

**Phase 1.3: centralised benchmark.**

```powershell
python scripts/centralized_baseline.py
```

`logistic_sgd` is the same model family as the federated system (the fair comparison);
`random_forest` and `hist_gradient_boosting` are the stronger reference points.

**Phase 1.4: federated training.**

```powershell
flwr run . --stream
flwr run . --stream --run-config "num-server-rounds=30 local-epochs=2 partition-alpha=0.5"
```

**Phase 1.5: comparison** (3 and 5 clients, FedAvg, no attackers):

```powershell
python scripts/run_experiments.py --preset stage1 --rounds 20 --seeds 42 43 44
python scripts/plot_results.py
```

## 4. Stage 2 - attacks and defences

Malicious clients are the ones with `partition-id < round(malicious-fraction * clients)`.

| `attack`        | what a malicious client does                                              |
|-----------------|---------------------------------------------------------------------------|
| `label_flip`    | trains on inverted labels (data poisoning)                                |
| `model_poison`  | sends `w_global - scale * (w_local - w_global)` (update poisoning)        |

| `aggregation`  | rule                                                                    |
|----------------|-------------------------------------------------------------------------|
| `fedavg`       | example-weighted mean (undefended)                                      |
| `median`       | coordinate-wise median                                                  |
| `trimmed_mean` | drop the largest/smallest share of values per coordinate, then average  |
| `krum`         | keep the single update closest to its neighbours                        |
| `multikrum`    | average the `n - f` updates with the best Krum scores                   |

```powershell
# one run: undefended, then defended, at 30 % malicious clients
flwr federation simulation-config --num-supernodes 10
flwr run . --stream --run-config "attack='model_poison' malicious-fraction=0.3 aggregation='fedavg' run-name='demo_fedavg'"
flwr run . --stream --run-config "attack='model_poison' malicious-fraction=0.3 aggregation='multikrum' run-name='demo_mkrum'"

# the whole study (10 clients, every defence, both attacks, 0/10/20/30 %)
python scripts/run_experiments.py --preset stage2 --rounds 20 --seeds 42 43 44
python scripts/plot_results.py
```

The defences are given the attacker share `f` (`defense-fraction`, default = the true
`malicious-fraction`), the usual assumption in the Krum / trimmed-mean literature.

## 5. Outputs

| path                                   | contents                                                       |
|----------------------------------------|----------------------------------------------------------------|
| `results/runs/<run>/rounds.csv`        | per round: test metrics, aggregation time, bytes, attackers kept |
| `results/centralized/metrics.csv`      | centralised baseline metrics                                   |
| `results/summary.csv`, `summary.md`    | one row per run; comparison tables (mean ± std over seeds)      |
| `results/figures/*.png`                | convergence, attack impact, overhead                           |
| `reports/eda/*.png`                    | class distribution, correlations, feature distributions         |

## 6. Run settings (`[tool.flwr.app.config]`)

| key | default | meaning |
|-----|---------|---------|
| `num-server-rounds` | 20 | federated rounds |
| `local-epochs` | 1 | local SGD passes per round |
| `learning-rate` | 0.01 | client SGD step size |
| `partition-alpha` | 0.0 | 0 = IID split; e.g. 0.5 = non-IID (Dirichlet over classes) |
| `attack` / `malicious-fraction` / `poison-scale` | none / 0.0 / 3.0 | Stage 2 attack |
| `aggregation` | fedavg | see table above |
| `defense-fraction`, `trim-ratio` | -1 | -1 = derive from `malicious-fraction` |
| `fraction-evaluate` | 0.0 | 1.0 also collects per-client validation metrics |
| `run-name` | default | results go to `results/runs/<run-name>/` |
| `project-dir`, `data-dir`, `results-dir` | "" | override auto-detected paths |

## 7. Notes and known limits

- **Run `flwr run` from the project folder.** If the SuperLink was first started elsewhere and
  data/results end up in the wrong place, pass `--run-config "project-dir='C:/path/to/quickstart-sklearn'"`
  (`run_experiments.py` always does this).
- **Krum needs `n >= 2f + 3`.** With 5 clients it tolerates one attacker, so Stage 2 uses 10 clients
  (up to 3 attackers). `f` is clamped automatically when the federation is too small.
- **Label flipping is a mild attack on a linear model.** With IID data a minority of flipped clients
  is out-voted, so FedAvg barely degrades below ~40 % attackers. Model poisoning breaks FedAvg at 10-30 %.
  Add `--fractions 0.4 0.5` or use `partition-alpha` to see label flipping bite.
- Not implemented (future work): backdoor attacks, secure aggregation / differential privacy,
  neural-network models.
- The model is logistic regression trained with SGD. Random forests do not average across clients,
  so they are used only as centralised references.
