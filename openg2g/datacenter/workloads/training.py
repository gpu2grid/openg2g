"""Training workload: typed trace data and periodic overlay evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class TrainingTraceParams(BaseModel):
    """Parameters for synthetic training-like power trace generation.

    Attributes:
        duration_s: Total duration (seconds).
        dt_s: Timestep (seconds).
        seed: Random seed.
        P_hi: High plateau power (W).
        P_lo: Low plateau power (W).
        sigma_hi: Noise std in high plateaus (W).
        sigma_lo: Noise std in low plateaus (W).
        seg_lo_range: Duration range for low segments (seconds).
        seg_hi_range: Duration range for high segments (seconds).
        dip_prob_per_sec: Expected brief dips per second.
        dip_depth_range: Depth range for brief dips (W below current level).
        dip_dur_range: Duration range for brief dips (seconds).
        smooth_window_s: Smoothing window width (seconds).
        ramp_s: Initial warm-up ramp duration (seconds).
        ramp_from: Power at ramp start (W).
    """

    model_config = ConfigDict(frozen=True)

    duration_s: float = 1000.0
    dt_s: float = 0.1
    seed: int = 2
    P_hi: float = 225.0
    P_lo: float = 175.0
    sigma_hi: float = 50.0
    sigma_lo: float = 50.0
    seg_lo_range: tuple[float, float] = (10.0, 15.0)
    seg_hi_range: tuple[float, float] = (35.0, 40.0)
    dip_prob_per_sec: float = 0.010
    dip_depth_range: tuple[float, float] = (120.0, 125.0)
    dip_dur_range: tuple[float, float] = (0.06, 0.14)
    smooth_window_s: float = 0.30
    ramp_s: float = 18.0
    ramp_from: float = 50.0


def _generate_training_like_trace(params: TrainingTraceParams) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic training-like per-GPU power trace.

    Args:
        params: Generation parameters.

    Returns:
        Tuple of (time_array, power_array).
    """
    rng = np.random.default_rng(params.seed)
    t = np.arange(0.0, params.duration_s, params.dt_s)
    n = t.size

    env = np.empty(n, dtype=float)
    i = 0
    state_hi = True

    while i < n:
        if state_hi:
            dur = rng.uniform(*params.seg_hi_range)
            level = params.P_hi
        else:
            dur = rng.uniform(*params.seg_lo_range)
            level = params.P_lo

        j = min(n, i + int(np.round(dur / params.dt_s)))
        env[i:j] = level
        i = j
        state_hi = not state_hi

    noise = np.zeros(n, dtype=float)
    hi_mask = env > (params.P_hi + params.P_lo) / 2
    noise[hi_mask] = rng.normal(0.0, params.sigma_hi, size=hi_mask.sum())
    noise[~hi_mask] = rng.normal(0.0, params.sigma_lo, size=(~hi_mask).sum())

    p = env + noise

    w = max(1, int(np.round(params.smooth_window_s / params.dt_s)))
    if w > 1:
        kernel = np.ones(w) / w
        p = np.convolve(p, kernel, mode="same")

    n_dips = rng.poisson(params.dip_prob_per_sec * params.duration_s)
    for _ in range(n_dips):
        t0 = rng.uniform(0.0, params.duration_s)
        k0 = int(t0 / params.dt_s)
        dur = rng.uniform(*params.dip_dur_range)
        k1 = min(n, k0 + int(np.round(dur / params.dt_s)))
        if k1 <= k0:
            continue
        depth = rng.uniform(*params.dip_depth_range)
        p[k0:k1] = np.maximum(p[k0:k1] - depth, 0.0)

    if params.ramp_s > 0:
        k_ramp = min(n, int(np.round(params.ramp_s / params.dt_s)))
        ramp = np.linspace(params.ramp_from, params.P_hi, k_ramp)
        p[:k_ramp] = np.minimum(p[:k_ramp], ramp)

    return t, p


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
    def generate(cls, params: TrainingTraceParams | None = None) -> TrainingTrace:
        """Generate a synthetic training-like power trace.

        Args:
            params: Generation parameters. Uses defaults if `None`.

        Returns:
            A new [`TrainingTrace`][.] with generated data.
        """
        if params is None:
            params = TrainingTraceParams()
        t, p = _generate_training_like_trace(params)
        return cls(t_s=t, power_w=p)

    def save(self, csv_path: Path) -> None:
        """Save the trace to a CSV file.

        Args:
            csv_path: Output CSV path.
        """
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({self.COL_TIME: self.t_s, self.COL_POWER: self.power_w})
        df.to_csv(csv_path, index=False)

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

    @classmethod
    def ensure(cls, csv_path: Path, params: TrainingTraceParams | None = None) -> TrainingTrace:
        """Load from `csv_path`, generating first if needed.

        Args:
            csv_path: Path to the training trace CSV.
            params: Generation parameters. Required when no cached file exists.
                Uses defaults if `None` and generation is needed.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            logger.info("Generating training trace to %s ...", csv_path)
            cls.generate(params).save(csv_path)
        return cls.load(csv_path)
