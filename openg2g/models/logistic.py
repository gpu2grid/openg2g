"""Logistic fit models for power, latency, and throughput as functions of log2(batch_size)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LogisticParams:
    """Four-parameter logistic: y = b0 + L * sigmoid(k * (x - x0)).

    *x* is typically log2(batch_size).
    """

    L: float
    x0: float
    k: float
    b0: float

    def eval_x(self, x: float) -> float:
        """Evaluate at continuous x (= log2(batch_size))."""
        a = self.k * (float(x) - self.x0)
        if a >= 0:
            ea = math.exp(-a)
            s = 1.0 / (1.0 + ea)
        else:
            ea = math.exp(a)
            s = ea / (1.0 + ea)
        return float(self.b0 + self.L * s)

    def deriv_wrt_x(self, x: float) -> float:
        """dy/dx for y = b0 + L * sigmoid(k*(x - x0))."""
        a = self.k * (float(x) - self.x0)
        if a >= 0:
            ea = math.exp(-a)
            s = 1.0 / (1.0 + ea)
        else:
            ea = math.exp(a)
            s = ea / (1.0 + ea)
        ds_dx = self.k * s * (1.0 - s)
        return float(self.L * ds_dx)

    def eval(self, batch: int) -> float:
        """Evaluate at an integer batch size (converted to log2)."""
        x = math.log2(max(int(batch), 1))
        return self.eval_x(x)


class LogisticFitBank:
    """Collection of per-model logistic fits for power, latency, throughput.

    Args:
        csv_by_model: Mapping from model_label to path of a CSV with columns
            ``metric, L, x0, k, b0`` where metric is one of
            ``"power"``, ``"latency"``, ``"throughput"``.
    """

    def __init__(self, *, csv_by_model: Mapping[str, str | Path]):
        self.csv_by_model = {str(k): str(v) for k, v in csv_by_model.items()}
        self._fits: dict[str, dict[str, LogisticParams]] = {}

    def load_all(self) -> None:
        """Parse all CSVs and populate the fit bank."""
        self._fits.clear()
        for model, p in self.csv_by_model.items():
            df = pd.read_csv(p)
            need = {"metric", "L", "x0", "k", "b0"}
            if not need.issubset(df.columns):
                raise ValueError(
                    f"{p} missing columns {sorted(need - set(df.columns))}; got {list(df.columns)}"
                )

            d: dict[str, LogisticParams] = {}
            for _, row in df.iterrows():
                metric = str(row["metric"]).strip().lower()
                if metric not in ("power", "latency", "throughput"):
                    continue
                d[metric] = LogisticParams(
                    L=float(row["L"]),  # type: ignore[arg-type]
                    x0=float(row["x0"]),  # type: ignore[arg-type]
                    k=float(row["k"]),  # type: ignore[arg-type]
                    b0=float(row["b0"]),  # type: ignore[arg-type]
                )

            missing = [m for m in ("power", "latency", "throughput") if m not in d]
            if missing:
                raise ValueError(f"{p}: missing metrics {missing}. Need power/latency/throughput.")
            self._fits[str(model)] = d

    def params(self, model_label: str, metric: str) -> LogisticParams:
        """Retrieve fit parameters for a given model and metric."""
        m = str(metric).strip().lower()
        mdl = str(model_label)
        if mdl not in self._fits:
            raise KeyError(f"Model {mdl!r} not loaded. Call load_all().")
        if m not in self._fits[mdl]:
            raise KeyError(f"Metric {m!r} not present for model {mdl!r}.")
        return self._fits[mdl][m]
