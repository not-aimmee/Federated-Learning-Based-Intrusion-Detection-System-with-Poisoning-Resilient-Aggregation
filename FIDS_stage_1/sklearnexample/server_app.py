"""sklearnexample: ServerApp - runs federated rounds with a pluggable aggregation rule.

After every round the global model is scored on the held-out test set, and the
results are written to ``results/runs/<run-name>/rounds.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from sklearnexample.config import get_config
from sklearnexample.strategy import ResilientFedAvg
from sklearnexample.task import (
    create_model,
    evaluate_model,
    get_model_params,
    load_meta,
    load_test_data,
    load_validation_data,
    set_model_params,
)

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = get_config(context)
    ddir = cfg["data-dir"]
    meta = load_meta(ddir)
    X_test, y_test = load_test_data(ddir)
    X_val, y_val = load_validation_data(ddir)

    mal_frac = float(cfg["malicious-fraction"])
    attack = str(cfg["attack"])
    if attack == "none":
        mal_frac = 0.0
    defense_frac = float(cfg["defense-fraction"])
    if defense_frac < 0:
        defense_frac = mal_frac

    run_info = {
        "run_name": cfg["run-name"],
        "aggregation": cfg["aggregation"],
        "attack": attack,
        "malicious_fraction": mal_frac,
        "partition_alpha": float(cfg["partition-alpha"]),
        "local_epochs": int(cfg["local-epochs"]),
        "seed": int(cfg["seed"]),
    }
    log_path = Path(cfg["results-dir"]) / "runs" / str(cfg["run-name"]) / "rounds.csv"

    strategy = ResilientFedAvg(
        aggregation=str(cfg["aggregation"]),
        defense_fraction=defense_frac,
        trim_ratio=float(cfg["trim-ratio"]),
        run_info=run_info,
        log_path=log_path,
        fraction_train=float(cfg["fraction-train"]),
        fraction_evaluate=float(cfg["fraction-evaluate"]),
    )

    # Global model used for centralised evaluation on the held-out test set.
    eval_model = create_model(
        meta["n_features"], meta["n_classes"], seed=int(cfg["seed"]), data_dir=ddir
    )

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        set_model_params(eval_model, arrays.to_numpy_ndarrays())
        val_loss, val_metrics = evaluate_model(eval_model, X_val, y_val)
        loss, metrics = evaluate_model(eval_model, X_test, y_test)
        combined = {**metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        strategy.log_round(server_round, loss, combined)
        return MetricRecord({"loss": loss, **combined})

    # Everything clients need travels in the training message.
    train_config = ConfigRecord(
        {
            "data-dir": ddir,
            "learning-rate": float(cfg["learning-rate"]),
            "local-epochs": int(cfg["local-epochs"]),
            "partition-alpha": float(cfg["partition-alpha"]),
            "seed": int(cfg["seed"]),
            "attack": attack,
            "malicious-fraction": mal_frac,
            "poison-scale": float(cfg["poison-scale"]),
        }
    )

    initial = ArrayRecord(
        get_model_params(create_model(
            meta["n_features"], meta["n_classes"], seed=int(cfg["seed"]), data_dir=ddir
        ))
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=initial,
        num_rounds=int(cfg["num-server-rounds"]),
        train_config=train_config,
        evaluate_config=train_config,
        evaluate_fn=evaluate_fn,
    )

    (log_path.parent / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"\nRound log written to {log_path}")
    final = result.evaluate_metrics_serverapp.get(int(cfg["num-server-rounds"]))
    print(f"Final global test metrics: {final}")