"""sklearnexample: ClientApp - one simulated organisation with its own local IDS data.

Clients whose ``partition-id`` is below ``round(malicious-fraction * num-partitions)``
behave maliciously according to the ``attack`` setting (see attacks.py). All settings
arrive from the ServerApp inside the training message.
"""

from __future__ import annotations

import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from sklearnexample.attacks import flip_labels, num_malicious, poison_update
from sklearnexample.task import (
    create_model,
    evaluate_model,
    get_model_params,
    load_client_data,
    load_meta,
    set_model_params,
    train_local,
)

app = ClientApp()


def _setup(msg: Message, context: Context):
    cfg = msg.content["config"]
    data_dir = str(cfg["data-dir"])
    meta = load_meta(data_dir)
    model = create_model(
        meta["n_features"], meta["n_classes"], float(cfg["learning-rate"]), int(cfg["seed"])
    )
    set_model_params(model, msg.content["arrays"].to_numpy_ndarrays())
    pid = int(context.node_config["partition-id"])
    n_parts = int(context.node_config["num-partitions"])
    data = load_client_data(
        pid, n_parts, float(cfg["partition-alpha"]), int(cfg["seed"]), data_dir=data_dir
    )
    malicious = (
        str(cfg["attack"]) != "none"
        and pid < num_malicious(float(cfg["malicious-fraction"]), n_parts)
    )
    return cfg, meta, model, data, malicious


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg, meta, model, (X_tr, y_tr, X_val, y_val), malicious = _setup(msg, context)
    attack = str(cfg["attack"])
    global_params = get_model_params(model)

    y_fit = flip_labels(y_tr, meta["n_classes"]) if malicious and attack == "label_flip" else y_tr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_local(model, X_tr, y_fit, int(cfg["local-epochs"]))

    params = get_model_params(model)
    if malicious and attack == "model_poison":
        params = poison_update(global_params, params, float(cfg["poison-scale"]))

    # Honest-view training loss on a small slice (cheap diagnostic only).
    sub = slice(0, min(len(X_tr), 20_000))
    loss, _ = evaluate_model(model, X_tr[sub], y_tr[sub])

    metrics = MetricRecord(
        {
            "num-examples": len(X_tr),
            "train-loss": float(loss),
            # Ground-truth flag: used ONLY for evaluation logging, never by defences.
            "malicious": int(malicious),
        }
    )
    content = RecordDict({"arrays": ArrayRecord(params), "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    cfg, _, model, (_, _, X_val, y_val), _ = _setup(msg, context)
    loss, m = evaluate_model(model, X_val, y_val)
    metrics = MetricRecord(
        {
            "num-examples": len(X_val),
            "val-loss": float(loss),
            "val-accuracy": m["accuracy"],
            "val-f1": m["f1"],
            "val-fpr": m["fpr"],
        }
    )
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)