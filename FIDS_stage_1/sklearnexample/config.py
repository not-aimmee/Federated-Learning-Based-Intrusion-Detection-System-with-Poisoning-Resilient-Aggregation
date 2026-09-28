"""Run configuration for the federated IDS.

Resolution order (later wins): ``DEFAULTS`` below, then ``[tool.flwr.app.config]`` in
pyproject.toml, then ``flwr run --run-config`` overrides.

Only the ServerApp reads this. It forwards everything the clients need (learning rate,
attack settings, data location ...) inside the training message.

Where are data/ and results/?  The Flower SuperLink may be running from a different
working directory than your shell, so paths are resolved in this order:
``project-dir`` run-config key > ``FIDS_HOME`` env var > current directory (if it has a
pyproject.toml) > the folder containing this package.
"""

from __future__ import annotations

import os
from pathlib import Path

from flwr.app import Context

DEFAULTS: dict = {
    "num-server-rounds": 20,
    "fraction-train": 1.0,  # keep 1.0 so the malicious share is exact every round
    "fraction-evaluate": 0.0,  # 1.0 -> also collect per-client validation metrics
    "local-epochs": 1,
    "learning-rate": 0.01,
    "partition-alpha": 0.0,  # 0 = IID; small (e.g. 0.5) = strongly non-IID
    "seed": 42,
    "attack": "none",  # none | label_flip | model_poison
    "malicious-fraction": 0.0,
    "poison-scale": 3.0,  # strength of the model_poison attack
    "aggregation": "fedavg",  # fedavg | median | trimmed_mean | krum | multikrum
    "defense-fraction": -1.0,  # assumed attacker share; -1 = use malicious-fraction
    "trim-ratio": -1.0,  # trimmed-mean trim share; -1 = use the defense fraction
    "run-name": "default",
    "project-dir": "",  # "" = auto-detect (see module docstring)
    "data-dir": "",  # "" = <project-dir>/data/processed
    "results-dir": "",  # "" = <project-dir>/results
}


def project_root(override: str | None = None) -> Path:
    """Locate the project folder (the one holding data/, results/, scripts/)."""
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("FIDS_HOME")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        return cwd
    return Path(__file__).resolve().parents[1]


def data_dir(root: Path | None = None) -> Path:
    env = os.environ.get("FIDS_DATA_DIR")
    return Path(env) if env else (root or project_root()) / "data" / "processed"


def results_dir(root: Path | None = None) -> Path:
    env = os.environ.get("FIDS_RESULTS_DIR")
    return Path(env) if env else (root or project_root()) / "results"


def get_config(context: Context) -> dict:
    """Merge defaults with the Flower run-config and resolve absolute paths."""
    cfg = dict(DEFAULTS)
    cfg.update(dict(context.run_config))
    root = project_root(cfg["project-dir"] or None)
    cfg["project-dir"] = str(root)
    cfg["data-dir"] = str(Path(cfg["data-dir"]) if cfg["data-dir"] else data_dir(root))
    cfg["results-dir"] = str(
        Path(cfg["results-dir"]) if cfg["results-dir"] else results_dir(root)
    )
    return cfg
