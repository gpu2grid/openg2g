"""Training workload overlay: typed trace data and periodic overlay evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainingTrace:
    """A single-GPU training power trace.

    Attributes:
        t_s: Time vector (seconds), monotonically increasing.
        power_w: Power vector (watts) for one GPU, same length as `t_s`.
    """

    COL_TIME = "t_s"
    COL_POWER = "power_W"

    t_s: np.ndarray
    power_w: np.ndarray

    def __post_init__(self) -> None:
        if len(self.t_s) != len(self.power_w):
            raise ValueError(f"t_s and power_w must have the same length, got {len(self.t_s)} and {len(self.power_w)}")
        if len(self.t_s) < 2:
            raise ValueError("Training trace must have >= 2 samples.")

    @classmethod
    def load(cls, csv_path: Path) -> TrainingTrace:
        """Load a training trace from CSV.

        Args:
            csv_path: Path to CSV with columns `t_s` and `power_W`.
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        if cls.COL_TIME not in df.columns or cls.COL_POWER not in df.columns:
            raise ValueError(
                f"{csv_path} must have columns {cls.COL_TIME!r} and {cls.COL_POWER!r}. Got {list(df.columns)}"
            )

        t = df[cls.COL_TIME].to_numpy(float)
        p = np.clip(df[cls.COL_POWER].to_numpy(float), 0.0, None)

        if np.any(np.diff(t) < 0):
            idx = np.argsort(t)
            t, p = t[idx], p[idx]

        return cls(t_s=t, power_w=p)


class TrainingOverlayCache:
    """Rescales a [`TrainingTrace`][..TrainingTrace] and provides
    periodic overlay evaluation.

    Args:
        trace: Single-GPU training power trace.
        target_peak_W_per_gpu: The trace is rescaled so its peak equals this
            value.
    """

    def __init__(self, trace: TrainingTrace, *, target_peak_W_per_gpu: float) -> None:
        training_time = np.asarray(trace.t_s, float)
        raw_power = np.asarray(trace.power_w, float)

        training_time = training_time - training_time[0]
        period = float(training_time[-1] - training_time[0])
        if period <= 0:
            raise ValueError("Training trace time span must be positive.")

        peak = float(np.max(raw_power))
        if peak <= 0:
            raise ValueError("Training trace has non-positive peak; cannot scale.")

        per_gpu_power = raw_power * (float(target_peak_W_per_gpu) / peak)

        self.training_time = training_time
        self.per_gpu_power = per_gpu_power
        self.period = period
        self.target_peak_W_per_gpu = float(target_peak_W_per_gpu)
        self.peak_1gpu_after_scaling = float(np.max(per_gpu_power))

    def eval_total_on_grid(
        self,
        t_global_s: np.ndarray,
        *,
        t_add_start: float,
        t_add_end: float,
        n_train_gpus: int,
    ) -> np.ndarray:
        """Evaluate total training power overlay on a time grid.

        Args:
            t_global_s: Global simulation time stamps (seconds).
            t_add_start: Global time when training becomes active.
            t_add_end: Global time when training stops.
            n_train_gpus: Number of GPUs running the training workload.

        Returns:
            Total training power (W) at each time step.
        """
        t = np.asarray(t_global_s, float)
        overlay_total = np.zeros_like(t, dtype=float)
        mask = (t >= float(t_add_start)) & (t <= float(t_add_end))
        if not np.any(mask):
            return overlay_total

        t_local = t[mask] - float(t_add_start)
        t_mod = np.mod(t_local, self.period)
        p_local_1gpu = np.interp(
            t_mod,
            self.training_time,
            self.per_gpu_power,
            left=float(self.per_gpu_power[0]),
            right=float(self.per_gpu_power[-1]),
        )
        overlay_total[mask] = p_local_1gpu * int(n_train_gpus)
        return overlay_total
