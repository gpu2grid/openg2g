"""Training workload overlay: loads a 1-GPU training trace and provides periodic overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TrainingOverlayCache:
    """Loads a 1-GPU training trace and provides periodic overlay on any time grid.

    Args:
        train_csv: Path to CSV with columns ``t_s`` and ``power_W``
            (1-GPU power trace).
        target_peak_W_per_gpu: The trace is rescaled so its peak equals this
            value.
    """

    def __init__(self, train_csv: Path, *, target_peak_W_per_gpu: float):
        import pandas as pd

        df = pd.read_csv(train_csv)
        if "t_s" not in df.columns or "power_W" not in df.columns:
            raise ValueError(
                f"{train_csv} must have columns: t_s, power_W. Got {list(df.columns)}"
            )

        t_tr = df["t_s"].to_numpy(float)
        p_tr = np.clip(df["power_W"].to_numpy(float), 0.0, None)

        if np.any(np.diff(t_tr) < 0):
            idx = np.argsort(t_tr)
            t_tr = t_tr[idx]
            p_tr = p_tr[idx]

        t_tr = t_tr - t_tr[0]
        if len(t_tr) < 2:
            raise ValueError("Training trace must have >=2 samples.")
        period = float(t_tr[-1] - t_tr[0])
        if period <= 0:
            raise ValueError("Training trace time span must be positive.")

        peak = float(np.max(p_tr))
        if peak <= 0:
            raise ValueError("Training trace has non-positive peak; cannot scale.")

        p_tr_1gpu = p_tr * (float(target_peak_W_per_gpu) / peak)

        self.train_csv = Path(train_csv)
        self.t_tr = t_tr
        self.p_tr_1gpu = p_tr_1gpu
        self.period = period
        self.target_peak_W_per_gpu = float(target_peak_W_per_gpu)
        self.peak_1gpu_after_scaling = float(np.max(p_tr_1gpu))

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
            self.t_tr,
            self.p_tr_1gpu,
            left=float(self.p_tr_1gpu[0]),
            right=float(self.p_tr_1gpu[-1]),
        )
        overlay_total[mask] = p_local_1gpu * int(n_train_gpus)
        return overlay_total
