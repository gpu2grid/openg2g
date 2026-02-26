"""Generate a synthetic GPU training-like power trace.

Produces a CSV with columns ``t_s`` and ``power_W`` that can be used as a
training overlay in the simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from openg2g.datacenter.training_overlay import TrainingTrace


def generate_training_like_trace(
    T: float = 1400.0,
    dt: float = 0.02,
    seed: int = 0,
    P_hi: float = 250.0,
    P_lo: float = 150.0,
    sigma_hi: float = 50.0,
    sigma_lo: float = 40.0,
    seg_lo_range: tuple[float, float] = (8.0, 16.0),
    seg_hi_range: tuple[float, float] = (18.0, 35.0),
    dip_prob_per_sec: float = 0.015,
    dip_depth_range: tuple[float, float] = (150.0, 180.0),
    dip_dur_range: tuple[float, float] = (0.06, 0.20),
    smooth_window_s: float = 0.25,
    ramp_s: float = 20.0,
    ramp_from: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic training-like per-GPU power trace.

    Args:
        T: Total duration (seconds).
        dt: Timestep (seconds).
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

    Returns:
        Tuple of (time_array, power_array).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T, dt)
    n = t.size

    # 1) Build a two-state piecewise-constant envelope (low/high blocks)
    env = np.empty(n, dtype=float)
    i = 0
    state_hi = True

    while i < n:
        if state_hi:
            dur = rng.uniform(*seg_hi_range)
            level = P_hi
        else:
            dur = rng.uniform(*seg_lo_range)
            level = P_lo

        j = min(n, i + int(np.round(dur / dt)))
        env[i:j] = level
        i = j
        state_hi = not state_hi

    # 2) Add within-plateau noise
    noise = np.zeros(n, dtype=float)
    hi_mask = env > (P_hi + P_lo) / 2
    noise[hi_mask] = rng.normal(0.0, sigma_hi, size=hi_mask.sum())
    noise[~hi_mask] = rng.normal(0.0, sigma_lo, size=(~hi_mask).sum())

    p = env + noise

    # 3) Optional smoothing
    w = max(1, int(np.round(smooth_window_s / dt)))
    if w > 1:
        kernel = np.ones(w) / w
        p = np.convolve(p, kernel, mode="same")

    # 4) Brief dips
    n_dips = rng.poisson(dip_prob_per_sec * T)
    for _ in range(n_dips):
        t0 = rng.uniform(0.0, T)
        k0 = int(t0 / dt)
        dur = rng.uniform(*dip_dur_range)
        k1 = min(n, k0 + int(np.round(dur / dt)))
        if k1 <= k0:
            continue
        depth = rng.uniform(*dip_depth_range)
        p[k0:k1] = np.maximum(p[k0:k1] - depth, 0.0)

    # 5) Warm-up ramp
    if ramp_s > 0:
        k_ramp = min(n, int(np.round(ramp_s / dt)))
        ramp = np.linspace(ramp_from, P_hi, k_ramp)
        p[:k_ramp] = np.minimum(p[:k_ramp], ramp)

    return t, p


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic GPU training-like power trace")
    parser.add_argument("--out-csv", required=True, help="Output CSV path")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--T", type=float, default=1000.0, help="Duration (s)")
    parser.add_argument("--dt", type=float, default=0.1, help="Timestep (s)")
    parser.add_argument("--P-hi", type=float, default=225.0)
    parser.add_argument("--P-lo", type=float, default=175.0)
    parser.add_argument("--sigma-hi", type=float, default=50.0)
    parser.add_argument("--sigma-lo", type=float, default=50.0)
    parser.add_argument("--seg-lo-min", type=float, default=10.0)
    parser.add_argument("--seg-lo-max", type=float, default=15.0)
    parser.add_argument("--seg-hi-min", type=float, default=35.0)
    parser.add_argument("--seg-hi-max", type=float, default=40.0)
    parser.add_argument("--dip-prob", type=float, default=0.010)
    parser.add_argument("--dip-depth-min", type=float, default=120.0)
    parser.add_argument("--dip-depth-max", type=float, default=125.0)
    parser.add_argument("--dip-dur-min", type=float, default=0.06)
    parser.add_argument("--dip-dur-max", type=float, default=0.14)
    parser.add_argument("--smooth-window-s", type=float, default=0.30)
    parser.add_argument("--ramp-s", type=float, default=18.0)
    parser.add_argument("--ramp-from", type=float, default=50.0)
    parser.add_argument("--plot", type=str, default=None, help="Save plot to this path")
    args = parser.parse_args()

    t, p = generate_training_like_trace(
        T=args.T,
        dt=args.dt,
        seed=args.seed,
        P_hi=args.P_hi,
        P_lo=args.P_lo,
        sigma_hi=args.sigma_hi,
        sigma_lo=args.sigma_lo,
        seg_lo_range=(args.seg_lo_min, args.seg_lo_max),
        seg_hi_range=(args.seg_hi_min, args.seg_hi_max),
        dip_prob_per_sec=args.dip_prob,
        dip_depth_range=(args.dip_depth_min, args.dip_depth_max),
        dip_dur_range=(args.dip_dur_min, args.dip_dur_max),
        smooth_window_s=args.smooth_window_s,
        ramp_s=args.ramp_s,
        ramp_from=args.ramp_from,
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({TrainingTrace.COL_TIME: t, TrainingTrace.COL_POWER: p})
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} samples to {out_path}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 3.2), dpi=140)
        ax.plot(t, p, lw=1.0)
        ax.set_title("Synthetic GPU training-like power trace")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power per GPU (W)")
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        fig.tight_layout()
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
