"""A FedAvg strategy whose aggregation rule is pluggable (FedAvg / Krum / trimmed mean / ...).

It also records, per round, everything the Stage 1/2 evaluation needs (global test
metrics, aggregation time, bytes exchanged, how many attackers Krum let through) and
writes it to ``results/runs/<run-name>/rounds.csv``.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAvg

from sklearnexample.aggregators import (
    METHODS,
    aggregate,
    flatten_params,
    unflatten_params,
)


class ResilientFedAvg(FedAvg):
    def __init__(
        self,
        *,
        aggregation: str = "fedavg",
        defense_fraction: float = 0.0,
        trim_ratio: float = -1.0,
        run_info: dict | None = None,
        log_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if aggregation not in METHODS:
            raise ValueError(f"aggregation must be one of {METHODS}, got '{aggregation}'")
        self.aggregation = aggregation
        self.defense_fraction = float(defense_fraction)  # assumed attacker share
        self.trim_ratio = float(trim_ratio)
        self.run_info = dict(run_info or {})
        self.log_path = Path(log_path) if log_path else None
        self._round_stats: dict[int, dict] = {}
        self._rows: list[dict] = []
        self._t0: float | None = None

    # ------------------------------------------------------------------ aggregation
    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid:
            return None, None

        contents = [m.content for m in valid]
        params = [next(iter(c.array_records.values())).to_numpy_ndarrays() for c in contents]
        metrics = [next(iter(c.metric_records.values())) for c in contents]
        weights = np.array([float(m[self.weighted_by_key]) for m in metrics])
        is_malicious = np.array([int(m.get("malicious", 0)) for m in metrics])

        shapes = [p.shape for p in params[0]]
        updates = np.stack([flatten_params(p) for p in params])
        n = len(updates)
        f = int(round(self.defense_fraction * n))
        trim = self.trim_ratio if self.trim_ratio >= 0 else f / n

        t0 = time.perf_counter()
        agg_flat, selected = aggregate(self.aggregation, updates, weights, f=f, trim_ratio=trim)
        agg_time = time.perf_counter() - t0

        stats = {
            "clients": n,
            "malicious_in_round": int(is_malicious.sum()),
            "agg_time_s": agg_time,
            "bytes_up": int(updates.nbytes),  # client -> server
            "bytes_down": int(updates.nbytes),  # server -> client (same model size)
        }
        if selected is not None:
            stats["kept_clients"] = len(selected)
            stats["malicious_kept"] = int(is_malicious[selected].sum())
        self._round_stats[server_round] = stats

        train_loss = [float(m.get("train-loss", 0.0)) for m in metrics]
        out = MetricRecord(
            {
                "train-loss": float(np.average(train_loss, weights=weights)),
                "agg-time-s": float(agg_time),
            }
        )
        return ArrayRecord(unflatten_params(agg_flat, shapes)), out

    # ---------------------------------------------------------------------- logging
    def log_round(self, server_round: int, loss: float, metrics: dict[str, float]) -> None:
        """Record global-test results for a round and flush ``rounds.csv``."""
        if self._t0 is None:
            self._t0 = time.perf_counter()
        row = {
            **self.run_info,
            "round": server_round,
            "loss": loss,
            **metrics,
            "elapsed_s": time.perf_counter() - self._t0,
            **self._round_stats.get(server_round, {}),
        }
        self._rows.append(row)
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(k for r in self._rows for k in r))
        with self.log_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(self._rows)
