"""RL scenario generation, randomization, and experiment factories.

What lives here:

- `DCSite`, `PVSystemSpec`, `TimeVaryingLoadSpec`: dataclasses describing the
  per-site simulation components that are randomized across scenarios.
- `ScenarioRecord`: scalar snapshot of one randomized scenario, produced by
  `build_library.py` and replayed by `train_ppo.py` and `evaluate.py`.
- `save_library` / `load_library_data`: serialize a list of `ScenarioRecord`
  to a `metadata.json` + `traces.npz` pair (no pickle).
- `ScenarioOpenDSSGrid`: `OpenDSSGrid` subclass that injects PV systems and
  time-varying loads at arbitrary buses.
- `pv_profile_random` / `tvl_profile_random` / `eval_profile`: parameterized
  PV/TVL profile generators used at runtime to materialize ScenarioRecord
  parameters into instantaneous power values.
- `randomize_scenario` / `materialize_scenario`: build a ScenarioRecord from
  a seed and turn it back into the concrete simulation configs each episode
  needs.
- `EXPERIMENTS` / `ieee*_experiment`: per-feeder factories that combine the
  shared feeder definitions from `examples/offline/systems.py` with the
  model deployments and DC layouts this study uses.

Shared simulation constants and feeder factories (ieee13/ieee34/ieee123,
DT_*, V_MIN/V_MAX, model specs, ...) come from the local `systems.py`,
which is a vendored copy of `examples/offline/systems.py` so the example
is self-contained.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from openg2g.controller.ofo import OFOConfig
from openg2g.datacenter.config import (
    ModelDeployment,
    ReplicaSchedule,
    TrainingRun,
)
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.config import TapPosition, TapSchedule
from openg2g.grid.opendss import OpenDSSGrid

from systems import (
    SYSTEMS,
    TOTAL_DURATION_S,
    V_MAX,
    V_MIN,
    _irregular_fluct,
    _smooth_bump,
    _smoothstep,
    deploy,
    load_profile_kw,
    pv_profile_kw,
    tap,
    with_ramp,
)


@dataclass
class DCSite:
    """One datacenter site for simulation setup.

    Attributes:
        bus: Distribution bus where the datacenter is connected.
        bus_kv: Bus voltage level (kV).
        base_kw_per_phase: Constant base load per phase (kW).
        total_gpu_capacity: Total physical GPUs installed at this site.
        models: (deployment, replica_schedule) pairs at this site. Replica
            counts and runtime ramps both live on the ReplicaSchedule
            (matching master's unpack_deployments() pattern).
        seed: Random seed for layout generation.
        connection_type: Grid connection type (`"wye"` or `"delta"`).
        load_shift_headroom: Fraction of extra server capacity for load shifting.
    """

    bus: str
    bus_kv: float
    base_kw_per_phase: float
    total_gpu_capacity: int
    models: tuple[tuple[ModelDeployment, ReplicaSchedule], ...] = ()
    seed: int = 0
    connection_type: str = "wye"
    load_shift_headroom: float = 0.0


@dataclass
class PVSystemSpec:
    """PV system at a distribution bus (used by ScenarioOpenDSSGrid)."""

    bus: str
    bus_kv: float = 4.16
    peak_kw: float = 1000.0
    csv_path: Path | None = None
    power_factor: float = 1.0
    peak_t_shift_s: float = 0.0
    time_warp: float = 1.0
    profile_kind: str = "default"
    profile_params: dict | None = None


@dataclass
class TimeVaryingLoadSpec:
    """Time-varying load at a distribution bus (used by ScenarioOpenDSSGrid)."""

    bus: str
    bus_kv: float = 4.16
    peak_kw: float = 500.0
    csv_path: Path | None = None
    power_factor: float = 0.96
    peak_t_shift_s: float = 0.0
    time_warp: float = 1.0
    profile_kind: str = "default"
    profile_params: dict | None = None


@dataclass
class ScenarioRecord:
    """One randomized scenario, ready to be replayed at training time.

    Built by `build_library.py` after baseline + OFO screening, and consumed
    by `train_ppo.py` (sampled per episode) and `evaluate.py` (replayed
    deterministically). `ofo_voltage_pen_per_step` is the per-second OFO
    voltage penalty trace and is what gets subtracted from PPO's per-step
    voltage_pen during training.
    """

    seed: int
    pv_scale: float
    load_scale: float
    training_overlay: dict | None
    baseline_integral: float
    ofo_integral: float
    baseline_violation_time_s: float
    ofo_violation_time_s: float
    recovery_frac: float
    ofo_voltage_pen_per_step: np.ndarray = field(repr=False)
    baseline_voltage_pen_per_step: np.ndarray = field(repr=False)
    t_control_start: int = 0
    t_control_end: int = 3600
    bl_undervoltage_time_s: float = 0.0
    bl_overvoltage_time_s: float = 0.0
    randomize_ramps: bool | None = None
    ramp_frac_min: float | None = None
    ramp_frac_max: float | None = None
    ramp_start_min: float | None = None
    ramp_start_max: float | None = None
    ramp_dur_min: float | None = None
    ramp_dur_max: float | None = None


def pv_profile_random(t: float, peak_kw: float, params: dict) -> float:
    """Multi-shape PV profile with random cloud events (1-hour episode).

    params["shape"] picks the envelope:
        "flat"             constant baseline 0.75-0.95
        "rising_falling"   smooth bump from low_baseline up to high_baseline and back
        "morning_ramp"     low → high over a short ramp window, sustained high after
        "afternoon_decline" sustained high then ramp down at the end
        "midday_dip"       high baseline with a substantial mid-episode dip
    """
    shape = params.get("shape", "flat")
    T = float(TOTAL_DURATION_S)

    if shape == "flat":
        env = float(params.get("baseline", 0.85))
    elif shape == "rising_falling":
        lo = float(params.get("low_baseline", 0.50))
        hi = float(params.get("high_baseline", 0.95))
        peak_t = float(params.get("peak_t", T / 2))
        half_width = float(params.get("half_width", 1200.0))
        env = lo + (hi - lo) * _smooth_bump(t, peak_t, half_width)
    elif shape == "morning_ramp":
        lo = float(params.get("low_baseline", 0.25))
        hi = float(params.get("high_baseline", 0.75))
        ramp_start = float(params.get("ramp_start", 100.0))
        ramp_end = float(params.get("ramp_end", 1200.0))
        env = lo + (hi - lo) * _smoothstep(t, ramp_start, ramp_end)
    elif shape == "afternoon_decline":
        lo = float(params.get("low_baseline", 0.25))
        hi = float(params.get("high_baseline", 0.75))
        ramp_start = float(params.get("ramp_start", 2400.0))
        ramp_end = float(params.get("ramp_end", T - 100.0))
        env = hi - (hi - lo) * _smoothstep(t, ramp_start, ramp_end)
    elif shape == "midday_dip":
        hi = float(params.get("high_baseline", 0.90))
        dip_t = float(params.get("dip_t", T / 2))
        dip_half_width = float(params.get("dip_half_width", 700.0))
        dip_depth = float(params.get("dip_depth", 0.55))
        env = hi - dip_depth * _smooth_bump(t, dip_t, dip_half_width)
    else:
        env = 0.85

    for tc, hw, depth in params.get("clouds", ()):
        env -= float(depth) * _smooth_bump(t, float(tc), float(hw))

    env = max(0.15, env)
    noise_amp = float(params.get("noise_amp", 0.0))
    if noise_amp > 0:
        f = _irregular_fluct(t, seed=float(params.get("noise_seed", 0.0)))
        env *= 1.0 + (noise_amp / 0.20) * (f - 1.0)
    return max(0.0, peak_kw * env)


def tvl_profile_random(t: float, peak_kw: float, params: dict) -> float:
    """Multi-shape TVL profile (1-hour episode)."""
    shape = params.get("shape", "peaked")
    T = float(TOTAL_DURATION_S)
    if shape == "flat":
        base = float(params.get("level", 0.7))
    elif shape == "increasing":
        lo = float(params.get("lo", 0.2))
        hi = float(params.get("hi", 0.9))
        base = lo + (hi - lo) * min(1.0, max(0.0, t / T))
    elif shape == "decreasing":
        lo = float(params.get("lo", 0.2))
        hi = float(params.get("hi", 0.9))
        base = hi - (hi - lo) * min(1.0, max(0.0, t / T))
    elif shape == "peaked":
        peak_t = float(params.get("peak_t", T / 2))
        peak_w = float(params.get("peak_w", 1400.0))
        baseline = float(params.get("baseline", 0.15))
        amp = float(params.get("amp", 0.85))
        base = baseline + amp * _smooth_bump(t, peak_t, peak_w)
    elif shape == "valley":
        valley_t = float(params.get("valley_t", T / 2))
        valley_w = float(params.get("valley_w", 1200.0))
        high = float(params.get("high", 0.85))
        depth = float(params.get("depth", 0.55))
        base = high - depth * _smooth_bump(t, valley_t, valley_w)
    else:
        base = 0.5
    base = max(0.0, base)
    noise_amp = float(params.get("noise_amp", 0.0))
    if noise_amp > 0:
        f = _irregular_fluct(t, seed=float(params.get("noise_seed", 0.0)))
        base *= 1.0 + (noise_amp / 0.20) * (f - 1.0)
    return max(0.0, peak_kw * base)


def load_csv_profile(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def eval_profile(
    t,
    *,
    peak_kw,
    csv_data,
    profile_fn,
    site_idx,
    peak_t_shift_s: float = 0.0,
    time_warp: float = 1.0,
    profile_kind: str = "default",
    profile_params: dict | None = None,
):
    """Evaluate a PV/TVL profile at simulated time `t`.

    Dispatch order:
      1. `profile_kind="random_flat"` + `profile_params` -> pv_profile_random
      2. `profile_kind="random_shape"` + `profile_params` -> tvl_profile_random
      3. `csv_data` not None -> CSV interpolation
      4. `profile_fn` (analytical pv_profile_kw / load_profile_kw)
    """
    if profile_kind == "random_flat" and profile_params is not None:
        return pv_profile_random(t, peak_kw, profile_params)
    if profile_kind == "random_shape" and profile_params is not None:
        return tvl_profile_random(t, peak_kw, profile_params)
    if time_warp <= 0:
        time_warp = 1.0
    t_eff = (t - peak_t_shift_s) / time_warp
    if csv_data is not None:
        return float(np.interp(t_eff, csv_data[0], csv_data[1]))
    return profile_fn(t_eff, peak_kw, site_idx)


class ScenarioOpenDSSGrid(OpenDSSGrid):
    """OpenDSSGrid with PV systems and external loads at arbitrary buses."""

    def __init__(
        self, *, pv_systems=None, time_varying_loads=None, source_pu=None, constant_pv: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self._pv_specs = list(pv_systems or [])
        self._load_specs = list(time_varying_loads or [])
        self._source_pu = source_pu
        self._constant_pv = constant_pv

        self._pv_csv = [load_csv_profile(s.csv_path) if s.csv_path else None for s in self._pv_specs]
        self._load_csv = [load_csv_profile(s.csv_path) if s.csv_path else None for s in self._load_specs]
        self._pv_load_names = [(f"PV_{i}_A", f"PV_{i}_B", f"PV_{i}_C") for i in range(len(self._pv_specs))]
        self._ext_load_names = [
            (f"ExtLoad_{i}_A", f"ExtLoad_{i}_B", f"ExtLoad_{i}_C") for i in range(len(self._load_specs))
        ]

    def _init_dss(self) -> None:
        super()._init_dss()
        from openg2g.grid.opendss import dss

        if self._source_pu is not None:
            dss.Text.Command(f"Edit Vsource.source pu={self._source_pu}")

        for i, spec in enumerate(self._pv_specs):
            kv_ln = spec.bus_kv / math.sqrt(3.0)
            for ph, name in zip((1, 2, 3), self._pv_load_names[i], strict=False):
                dss.Text.Command(
                    f"New Load.{name} bus1={spec.bus}.{ph} phases=1 "
                    f"conn=wye kV={kv_ln:.6f} kW=0 kvar=0 model=1 vminpu=0.85"
                )

        for i, spec in enumerate(self._load_specs):
            kv_ln = spec.bus_kv / math.sqrt(3.0)
            for ph, name in zip((1, 2, 3), self._ext_load_names[i], strict=False):
                dss.Text.Command(
                    f"New Load.{name} bus1={spec.bus}.{ph} phases=1 "
                    f"conn=wye kV={kv_ln:.6f} kW=0 kvar=0 model=1 vminpu=0.85"
                )

    def step(self, clock, power_samples_w, events):
        from openg2g.grid.opendss import dss

        for i, spec in enumerate(self._pv_specs):
            if self._constant_pv:
                kw = spec.peak_kw
            else:
                kw = eval_profile(
                    clock.time_s,
                    peak_kw=spec.peak_kw,
                    csv_data=self._pv_csv[i],
                    profile_fn=pv_profile_kw,
                    site_idx=i,
                    peak_t_shift_s=getattr(spec, "peak_t_shift_s", 0.0),
                    time_warp=getattr(spec, "time_warp", 1.0),
                    profile_kind=getattr(spec, "profile_kind", "default"),
                    profile_params=getattr(spec, "profile_params", None),
                )
            pf = max(min(spec.power_factor, 0.999999), 1e-6)
            kvar = kw * math.tan(math.acos(pf))
            for name in self._pv_load_names[i]:
                dss.Loads.Name(name)
                dss.Loads.kW(-kw)
                dss.Loads.kvar(-kvar)

        for i, spec in enumerate(self._load_specs):
            kw = eval_profile(
                clock.time_s,
                peak_kw=spec.peak_kw,
                csv_data=self._load_csv[i],
                profile_fn=load_profile_kw,
                site_idx=i,
                peak_t_shift_s=getattr(spec, "peak_t_shift_s", 0.0),
                time_warp=getattr(spec, "time_warp", 1.0),
                profile_kind=getattr(spec, "profile_kind", "default"),
                profile_params=getattr(spec, "profile_params", None),
            )
            pf = max(min(spec.power_factor, 0.999999), 1e-6)
            kvar = kw * math.tan(math.acos(pf))
            for name in self._ext_load_names[i]:
                dss.Loads.Name(name)
                dss.Loads.kW(kw)
                dss.Loads.kvar(kvar)

        return super().step(clock, power_samples_w, events)


def _site_inference_gpus(site: DCSite) -> int:
    """Sum of GPUs consumed by inference at a site (replicas × gpus_per_replica)."""
    return sum(sched.initial * md.spec.gpus_per_replica for md, sched in site.models)


def _randomize_ramps(
    dc_sites: dict[str, DCSite],
    rng: np.random.Generator,
    *,
    ramp_frac_min: float = 0.15,
    ramp_frac_max: float = 0.3,
    ramp_start_min: float = 500.0,
    ramp_start_max: float = 3000.0,
    ramp_dur_min: float = 300.0,
    ramp_dur_max: float = 800.0,
) -> dict[str, DCSite]:
    """Return a copy of dc_sites with randomized ramp targets and timing."""
    ramp_frac = rng.uniform(ramp_frac_min, ramp_frac_max)
    ramp_start = rng.uniform(ramp_start_min, ramp_start_max)
    ramp_dur = rng.uniform(ramp_dur_min, ramp_dur_max)
    ramp_end = ramp_start + ramp_dur

    new_sites: dict[str, DCSite] = {}
    for sid, site in dc_sites.items():
        new_models: list[tuple[ModelDeployment, ReplicaSchedule]] = []
        for md, sched in site.models:
            target = max(1, int(ramp_frac * sched.initial))
            new_sched = ReplicaSchedule(initial=sched.initial).ramp_to(
                target,
                t_start=ramp_start,
                t_end=ramp_end,
            )
            new_models.append((md, new_sched))
        new_sites[sid] = DCSite(
            bus=site.bus,
            bus_kv=site.bus_kv,
            base_kw_per_phase=site.base_kw_per_phase,
            total_gpu_capacity=site.total_gpu_capacity,
            models=tuple(new_models),
            seed=int(rng.integers(0, 10000)),
            connection_type=site.connection_type,
        )
    return new_sites


def _randomize_broad_ramps(
    dc_sites: dict[str, DCSite],
    rng: np.random.Generator,
    *,
    overlay_gpus_at_first_site: int,
    n_ramps_per_site_choices: tuple[int, ...] = (1, 2),
    ramp_up_prob: float = 0.5,
    ramp_down_frac_min: float = 0.15,
    ramp_down_frac_max: float = 0.5,
    ramp_up_frac_min: float = 1.05,
    ramp_up_frac_max: float = 1.5,
    ramp_start_min: float = 500.0,
    ramp_start_max: float = 3000.0,
    ramp_dur_min: float = 300.0,
    ramp_dur_max: float = 800.0,
    exclude_window: tuple[float, float] | None = None,
) -> dict[str, DCSite]:
    """Bidirectional, multi-ramp generator that respects DC GPU capacity."""
    if exclude_window is not None:
        ex_lo, ex_hi = exclude_window
        zone1_hi = ex_lo - ramp_dur_max
        zone2_lo = ex_hi
        zones: list[tuple[float, float]] = []
        if ramp_start_min < zone1_hi:
            zones.append((ramp_start_min, min(zone1_hi, ramp_start_max)))
        if zone2_lo < ramp_start_max:
            zones.append((max(zone2_lo, ramp_start_min), ramp_start_max))
        if not zones:
            zones = [(ramp_start_min, ramp_start_max)]
    else:
        zones = [(ramp_start_min, ramp_start_max)]

    total_width = sum(hi - lo for lo, hi in zones)

    new_sites: dict[str, DCSite] = {}
    sites_list = list(dc_sites.items())
    for site_idx, (sid, site) in enumerate(sites_list):
        current_gpus = _site_inference_gpus(site)
        overlay_here = overlay_gpus_at_first_site if site_idx == 0 else 0
        available_gpus = max(0, site.total_gpu_capacity - overlay_here)
        max_feasible_up = available_gpus / current_gpus if current_gpus > 0 else 1.0
        site_ramp_up_max = min(ramp_up_frac_max, max_feasible_up)
        can_up_ramp = site_ramp_up_max >= max(1.05, ramp_up_frac_min)

        n_ramps = int(rng.choice(n_ramps_per_site_choices))
        zone_counts = [int(n_ramps * (hi - lo) / total_width) for lo, hi in zones]
        remainder = n_ramps - sum(zone_counts)
        if remainder > 0:
            widest = max(range(len(zones)), key=lambda i: zones[i][1] - zones[i][0])
            zone_counts[widest] += remainder

        # Per-model schedule starts from the model's existing initial count.
        model_scheds: dict[str, ReplicaSchedule] = {
            md.spec.model_label: ReplicaSchedule(initial=sched.initial) for md, sched in site.models
        }
        for (z_lo, z_hi), z_count in zip(zones, zone_counts, strict=False):
            if z_count == 0:
                continue
            band_width = (z_hi - z_lo) / z_count
            for bi in range(z_count):
                band_lo = z_lo + bi * band_width
                band_hi = z_lo + (bi + 1) * band_width
                if band_width < ramp_dur_min:
                    continue
                t_dur = float(rng.uniform(ramp_dur_min, min(ramp_dur_max, band_width)))
                t_start = float(rng.uniform(band_lo, max(band_lo + 1.0, band_hi - t_dur)))
                t_end = t_start + t_dur

                if can_up_ramp and rng.random() < ramp_up_prob:
                    lo = max(1.05, ramp_up_frac_min)
                    hi = max(lo + 1e-3, site_ramp_up_max)
                    frac = float(rng.uniform(lo, hi))
                else:
                    frac = float(rng.uniform(ramp_down_frac_min, ramp_down_frac_max))

                for md, sched in site.models:
                    target = max(1, int(round(frac * sched.initial)))
                    label = md.spec.model_label
                    model_scheds[label] = model_scheds[label].ramp_to(
                        target,
                        t_start=t_start,
                        t_end=t_end,
                    )

        new_models = tuple((md, model_scheds[md.spec.model_label]) for md, _ in site.models)
        new_sites[sid] = DCSite(
            bus=site.bus,
            bus_kv=site.bus_kv,
            base_kw_per_phase=site.base_kw_per_phase,
            total_gpu_capacity=site.total_gpu_capacity,
            models=new_models,
            seed=int(rng.integers(0, 10000)),
            connection_type=site.connection_type,
        )
    return new_sites


def randomize_scenario(
    seed: int,
    *,
    dc_sites_base: dict[str, DCSite],
    pv_systems_base: list[PVSystemSpec],
    tvl_base: list[TimeVaryingLoadSpec],
    training_base: dict | None,
    randomize_ramps: bool = True,
    ramp_frac_min: float = 0.15,
    ramp_frac_max: float = 0.3,
    ramp_start_min: float = 500.0,
    ramp_start_max: float = 3000.0,
    ramp_dur_min: float = 300.0,
    ramp_dur_max: float = 800.0,
    randomization_profile: bool = True,
    pv_scale_min: float = 0.5,
    pv_scale_max: float = 2.0,
    load_scale_min: float = 0.5,
    load_scale_max: float = 2.0,
    pv_t_shift_max_s: float = 0.0,
    tvl_t_shift_max_s: float = 0.0,
    pv_warp_min: float = 1.0,
    pv_warp_max: float = 1.0,
    tvl_warp_min: float = 1.0,
    tvl_warp_max: float = 1.0,
    overlay_prob: float = 1.0,
    overlay_intensity_min: float = 1.0,
    overlay_intensity_max: float = 1.0,
    overlay_gpu_frac_min: float = 0.85,
    overlay_gpu_frac_max: float = 1.0,
    n_ramps_per_site_choices: tuple[int, ...] = (1,),
    ramp_up_prob: float = 0.0,
    ramp_down_frac_min: float = 0.15,
    ramp_down_frac_max: float = 0.5,
    ramp_up_frac_min: float = 1.05,
    ramp_up_frac_max: float = 1.5,
    randomize_pv_profile: bool = False,
    pv_shape_choices: tuple[str, ...] = (
        "flat",
        "rising_falling",
        "morning_ramp",
        "afternoon_decline",
        "midday_dip",
    ),
    pv_baseline_min: float = 0.75,
    pv_baseline_max: float = 0.95,
    pv_cloud_count_max: int = 3,
    pv_cloud_depth_min: float = 0.30,
    pv_cloud_depth_max: float = 0.70,
    pv_cloud_width_min: float = 60.0,
    pv_cloud_width_max: float = 300.0,
    randomize_tvl_profile: bool = False,
    tvl_shape_choices: tuple[str, ...] = ("flat", "increasing", "decreasing", "peaked", "valley"),
) -> dict:
    """Build a single randomized episode scenario from a seed."""
    rng = np.random.default_rng(seed=seed)
    is_broad = randomization_profile if isinstance(randomization_profile, bool) else (randomization_profile == "broad")

    training_run = None
    train_overlay_meta: dict | None = None
    overlay_gpus_at_first_site = 0
    if training_base is not None:
        if is_broad:
            overlay_on = bool(rng.random() < overlay_prob)
        else:
            overlay_on = True
        if overlay_on:
            train_dur = float(rng.uniform(500.0, 1200.0))
            if is_broad:
                train_start = float(rng.uniform(0.0, max(0.0, float(TOTAL_DURATION_S) - train_dur)))
            else:
                train_start = float(rng.uniform(500.0, 1500.0))
            gpu_frac = (
                float(rng.uniform(overlay_gpu_frac_min, overlay_gpu_frac_max))
                if is_broad
                else float(rng.uniform(0.85, 1.0))
            )
            train_gpus = int(gpu_frac * training_base["n_gpus"])
            intensity = float(rng.uniform(overlay_intensity_min, overlay_intensity_max)) if is_broad else 1.0
            target_peak = training_base["target_peak_W_per_gpu"] * intensity
            training_run = TrainingRun(
                n_gpus=train_gpus,
                trace=training_base["trace"],
                target_peak_W_per_gpu=target_peak,
            ).at(t_start=train_start, t_end=train_start + train_dur)
            train_overlay_meta = {
                "n_gpus": train_gpus,
                "target_peak_W_per_gpu": target_peak,
                "intensity": intensity,
                "t_start": train_start,
                "t_end": train_start + train_dur,
            }
            overlay_gpus_at_first_site = train_gpus

    if randomize_ramps:
        if is_broad:
            overlay_window: tuple[float, float] | None = None
            if train_overlay_meta is not None:
                overlay_window = (train_overlay_meta["t_start"], train_overlay_meta["t_end"])
            sites = _randomize_broad_ramps(
                dc_sites_base,
                rng,
                overlay_gpus_at_first_site=overlay_gpus_at_first_site,
                n_ramps_per_site_choices=tuple(n_ramps_per_site_choices),
                ramp_up_prob=ramp_up_prob,
                ramp_down_frac_min=ramp_down_frac_min,
                ramp_down_frac_max=ramp_down_frac_max,
                ramp_up_frac_min=ramp_up_frac_min,
                ramp_up_frac_max=ramp_up_frac_max,
                ramp_start_min=ramp_start_min,
                ramp_start_max=ramp_start_max,
                ramp_dur_min=ramp_dur_min,
                ramp_dur_max=ramp_dur_max,
                exclude_window=overlay_window,
            )
        else:
            sites = _randomize_ramps(
                dc_sites_base,
                rng,
                ramp_frac_min=ramp_frac_min,
                ramp_frac_max=ramp_frac_max,
                ramp_start_min=ramp_start_min,
                ramp_start_max=ramp_start_max,
                ramp_dur_min=ramp_dur_min,
                ramp_dur_max=ramp_dur_max,
            )
    else:
        sites = dict(dc_sites_base)

    pv_scale = float(rng.uniform(pv_scale_min, pv_scale_max)) if is_broad else float(rng.uniform(0.5, 2.0))
    load_scale = float(rng.uniform(load_scale_min, load_scale_max)) if is_broad else float(rng.uniform(0.5, 2.0))
    pv_t_shift = float(rng.uniform(-pv_t_shift_max_s, pv_t_shift_max_s)) if (is_broad and pv_t_shift_max_s > 0) else 0.0
    tvl_t_shift = (
        float(rng.uniform(-tvl_t_shift_max_s, tvl_t_shift_max_s)) if (is_broad and tvl_t_shift_max_s > 0) else 0.0
    )
    pv_warp = float(rng.uniform(pv_warp_min, pv_warp_max)) if is_broad else 1.0
    tvl_warp = float(rng.uniform(tvl_warp_min, tvl_warp_max)) if is_broad else 1.0

    def _sample_pv_profile(rng_) -> tuple[str, dict | None]:
        n_clouds = int(rng_.integers(0, pv_cloud_count_max + 1))
        clouds = [
            (
                float(rng_.uniform(120.0, 3480.0)),
                float(rng_.uniform(pv_cloud_width_min, pv_cloud_width_max)),
                float(rng_.uniform(pv_cloud_depth_min, pv_cloud_depth_max)),
            )
            for _ in range(n_clouds)
        ]
        shape = str(rng_.choice(list(pv_shape_choices)))
        params: dict = {
            "shape": shape,
            "clouds": clouds,
            "noise_amp": float(rng_.uniform(0.02, 0.06)),
            "noise_seed": float(rng_.uniform(0.0, 10.0)),
        }
        if shape == "flat":
            params["baseline"] = float(rng_.uniform(pv_baseline_min, pv_baseline_max))
        elif shape == "rising_falling":
            lo = float(rng_.uniform(0.15, 0.55))
            params["low_baseline"] = lo
            params["high_baseline"] = float(rng_.uniform(lo + 0.20, 1.00))
            params["peak_t"] = float(rng_.uniform(900.0, 2700.0))
            params["half_width"] = float(rng_.uniform(800.0, 1500.0))
        elif shape == "morning_ramp":
            lo = float(rng_.uniform(0.15, 0.45))
            params["low_baseline"] = lo
            params["high_baseline"] = float(rng_.uniform(lo + 0.20, 1.00))
            params["ramp_start"] = float(rng_.uniform(0.0, 400.0))
            params["ramp_end"] = float(rng_.uniform(800.0, 1800.0))
        elif shape == "afternoon_decline":
            lo = float(rng_.uniform(0.15, 0.45))
            params["low_baseline"] = lo
            params["high_baseline"] = float(rng_.uniform(lo + 0.20, 1.00))
            params["ramp_start"] = float(rng_.uniform(1800.0, 2800.0))
            params["ramp_end"] = float(rng_.uniform(3200.0, 3600.0))
        elif shape == "midday_dip":
            params["high_baseline"] = float(rng_.uniform(0.30, 1.00))
            params["dip_t"] = float(rng_.uniform(1200.0, 2400.0))
            params["dip_half_width"] = float(rng_.uniform(500.0, 1000.0))
            params["dip_depth"] = float(rng_.uniform(0.20, 0.55))
        return "random_flat", params

    tvl_profile_kind = "default"
    tvl_profile_params: dict | None = None
    if is_broad and randomize_tvl_profile:
        shape = str(rng.choice(list(tvl_shape_choices)))
        params: dict = {"shape": shape}
        if shape == "flat":
            params["level"] = float(rng.uniform(0.5, 0.85))
        elif shape == "increasing" or shape == "decreasing":
            params["lo"] = float(rng.uniform(0.10, 0.30))
            params["hi"] = float(rng.uniform(0.65, 0.95))
        elif shape == "peaked":
            params["peak_t"] = float(rng.uniform(800.0, 2800.0))
            params["peak_w"] = float(rng.uniform(800.0, 1800.0))
            params["baseline"] = float(rng.uniform(0.10, 0.30))
            params["amp"] = float(rng.uniform(0.55, 0.90))
        elif shape == "valley":
            params["valley_t"] = float(rng.uniform(800.0, 2800.0))
            params["valley_w"] = float(rng.uniform(800.0, 1500.0))
            params["high"] = float(rng.uniform(0.70, 0.90))
            params["depth"] = float(rng.uniform(0.40, 0.70))
        params["noise_amp"] = float(rng.uniform(0.02, 0.06))
        params["noise_seed"] = float(rng.uniform(0.0, 10.0))
        tvl_profile_kind = "random_shape"
        tvl_profile_params = params

    pv_systems_out = []
    for s in pv_systems_base:
        if is_broad and randomize_pv_profile:
            p_kind, p_params = _sample_pv_profile(rng)
        else:
            p_kind, p_params = "default", None
        pv_systems_out.append(
            PVSystemSpec(
                bus=s.bus,
                bus_kv=s.bus_kv,
                peak_kw=s.peak_kw * pv_scale,
                peak_t_shift_s=pv_t_shift,
                time_warp=pv_warp,
                profile_kind=p_kind,
                profile_params=p_params,
            )
        )
    tvl = [
        TimeVaryingLoadSpec(
            bus=s.bus,
            bus_kv=s.bus_kv,
            peak_kw=s.peak_kw * load_scale,
            peak_t_shift_s=tvl_t_shift,
            time_warp=tvl_warp,
            profile_kind=tvl_profile_kind,
            profile_params=tvl_profile_params,
        )
        for s in tvl_base
    ]

    batch_choices = [32, 64, 128]
    initial_batch_map: dict[str, int] = {}
    new_sites: dict[str, DCSite] = {}
    for sid, site in sites.items():
        new_models: list[tuple[ModelDeployment, ReplicaSchedule]] = []
        for md, sched in site.models:
            bs = int(rng.choice(batch_choices))
            initial_batch_map[md.spec.model_label] = bs
            new_models.append(
                (ModelDeployment(spec=md.spec, initial_batch_size=bs), sched),
            )
        new_sites[sid] = DCSite(
            bus=site.bus,
            bus_kv=site.bus_kv,
            base_kw_per_phase=site.base_kw_per_phase,
            total_gpu_capacity=site.total_gpu_capacity,
            models=tuple(new_models),
            seed=site.seed,
            connection_type=site.connection_type,
        )

    return {
        "seed": int(seed),
        "dc_sites": new_sites,
        "pv_systems": pv_systems_out,
        "tvl": tvl,
        "training_run": training_run,
        "params": {
            "pv_scale": pv_scale,
            "load_scale": load_scale,
            "pv_t_shift_s": pv_t_shift,
            "tvl_t_shift_s": tvl_t_shift,
            "pv_warp": pv_warp,
            "tvl_warp": tvl_warp,
            "training_overlay": train_overlay_meta,
            "initial_batch_sizes": initial_batch_map,
            "randomization_profile": randomization_profile,
            "tvl_profile_kind": tvl_profile_kind,
            "tvl_profile_params": tvl_profile_params,
        },
    }


def save_library(
    library_dir: Path,
    scenarios: list[ScenarioRecord],
    config: dict,
) -> None:
    """Save a scenario library to `library_dir` as `metadata.json` + `traces.npz`.

    The two files together capture everything needed to replay a library:
    `metadata.json` is human-inspectable and stores the build config plus
    every `ScenarioRecord`'s scalar fields; `traces.npz` stores the per-step
    voltage penalty arrays as numpy arrays (`ofo_<i>`, `baseline_<i>`).
    """
    import json

    library_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict = {
        "config": config,
        "scenarios": [],
    }
    traces: dict[str, np.ndarray] = {}
    for i, rec in enumerate(scenarios):
        scalar_fields: dict = {}
        for f in ScenarioRecord.__dataclass_fields__:
            if f in {"ofo_voltage_pen_per_step", "baseline_voltage_pen_per_step"}:
                continue
            scalar_fields[f] = getattr(rec, f)
        metadata["scenarios"].append(scalar_fields)
        traces[f"ofo_{i}"] = rec.ofo_voltage_pen_per_step
        traces[f"baseline_{i}"] = rec.baseline_voltage_pen_per_step
    (library_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    np.savez(library_dir / "traces.npz", **traces)


def load_library_data(library_dir: Path) -> tuple[list[ScenarioRecord], dict]:
    """Read a library directory written by `save_library`.

    Returns `(scenarios, config)`. Does not construct materialization base
    components: `ScenarioLibrary` does that on top.
    """
    import json

    metadata = json.loads((library_dir / "metadata.json").read_text())
    traces = np.load(library_dir / "traces.npz")
    scenarios: list[ScenarioRecord] = []
    for i, fields in enumerate(metadata["scenarios"]):
        scenarios.append(
            ScenarioRecord(
                ofo_voltage_pen_per_step=traces[f"ofo_{i}"],
                baseline_voltage_pen_per_step=traces[f"baseline_{i}"],
                **fields,
            )
        )
    return scenarios, metadata["config"]


def materialize_scenario(
    rec: ScenarioRecord,
    *,
    dc_sites_base: dict[str, DCSite],
    pv_systems_base: list[PVSystemSpec],
    tvl_base: list[TimeVaryingLoadSpec],
    training_base: dict | None,
    randomize_kwargs: dict,
) -> dict:
    """Replay `randomize_scenario` from a record's seed to obtain the same
    per-episode scenario dict the build pipeline saw.

    This is `randomize_scenario(seed=rec.seed, ...)` with the
    `*_base` configurations and `randomize_kwargs` that the library was built
    with: the RNG is seeded so the replay is bit-identical to the build-time
    output (including `dc_sites`, `pv_systems`, `tvl`, and the chosen scalar
    parameters). `ScenarioLibrary.materialize` wires the base components in
    automatically; call this directly only when constructing a scenario
    outside a library (e.g. evaluate.py's seed-driven test set).
    """
    return randomize_scenario(
        seed=rec.seed,
        dc_sites_base=dc_sites_base,
        pv_systems_base=pv_systems_base,
        tvl_base=tvl_base,
        training_base=training_base,
        **randomize_kwargs,
    )


def ieee13_experiment(training_trace: TrainingTrace | None = None) -> dict:
    """IEEE 13-bus: single DC at bus 671 with 5 LLM models."""
    sys = SYSTEMS["ieee13"]()
    ramp_targets = {
        "Llama-3.1-8B": 144,
        "Llama-3.1-70B": 36,
        "Llama-3.1-405B": 18,
        "Qwen3-30B-A3B": 96,
        "Qwen3-235B-A22B": 42,
    }
    base_models = (
        deploy("Llama-3.1-8B", 720),
        deploy("Llama-3.1-70B", 180),
        deploy("Llama-3.1-405B", 90),
        deploy("Qwen3-30B-A3B", 480),
        deploy("Qwen3-235B-A22B", 210),
    )
    models = tuple(
        (md, sched.ramp_to(ramp_targets[md.spec.model_label], t_start=2500, t_end=3000)) for md, sched in base_models
    )
    training_base = (
        {
            "trace": training_trace,
            "n_gpus": 2400,
            "target_peak_W_per_gpu": 400.0,
            "t_start": 1000.0,
            "t_end": 2000.0,
        }
        if training_trace is not None
        else None
    )
    return dict(
        sys=sys,
        dc_sites={
            "default": DCSite(
                bus="671",
                bus_kv=sys["bus_kv"],
                base_kw_per_phase=500.0,
                total_gpu_capacity=7200,
                models=models,
                seed=0,
            ),
        },
        pv_systems=[PVSystemSpec(bus="675", bus_kv=4.16, peak_kw=300.0)],
        time_varying_loads=[TimeVaryingLoadSpec(bus="680", bus_kv=4.16, peak_kw=300.0)],
        training_base=training_base,
        ofo_config=OFOConfig(
            primal_step_size=0.05,
            w_throughput=0.00001,
            w_switch=1.0,
            voltage_gradient_scale=1e6,
            v_min=V_MIN,
            v_max=V_MAX,
            voltage_dual_step_size=1.0,
            latency_dual_step_size=1.0,
            sensitivity_update_interval=300,
            sensitivity_perturbation_kw=100.0,
        ),
        tap_schedule=TapSchedule(
            (
                (1500, TapPosition(regulators={"creg1a": tap(16), "creg1b": tap(6), "creg1c": tap(17)})),
                (3300, TapPosition(regulators={"creg1a": tap(10), "creg1b": tap(6), "creg1c": tap(10)})),
            )
        ),
    )


def ieee34_experiment(training_trace: TrainingTrace | None = None) -> dict:
    """IEEE 34-bus: two DC sites (upstream/downstream)."""
    sys = SYSTEMS["ieee34"]()
    return dict(
        sys=sys,
        dc_sites={
            "upstream": DCSite(
                bus="850",
                bus_kv=24.9,
                base_kw_per_phase=250.0,
                models=(deploy("Llama-3.1-8B", 320), deploy("Llama-3.1-70B", 80), deploy("Llama-3.1-405B", 40)),
                seed=0,
                total_gpu_capacity=1200,
            ),
            "downstream": DCSite(
                bus="834",
                bus_kv=24.9,
                base_kw_per_phase=300.0,
                models=(deploy("Qwen3-30B-A3B", 216), deploy("Qwen3-235B-A22B", 96)),
                seed=42,
                total_gpu_capacity=1440,
            ),
        },
        pv_systems=[
            PVSystemSpec(bus="858", bus_kv=24.9, peak_kw=130.0),
            PVSystemSpec(bus="852", bus_kv=24.9, peak_kw=65.0),
        ],
        time_varying_loads=[
            TimeVaryingLoadSpec(bus="860", bus_kv=24.9, peak_kw=80.0),
            TimeVaryingLoadSpec(bus="844", bus_kv=24.9, peak_kw=120.0),
            TimeVaryingLoadSpec(bus="858", bus_kv=24.9, peak_kw=50.0),
        ],
        training_base=None,
        ofo_config=OFOConfig(
            primal_step_size=0.05,
            w_throughput=0.0001,
            w_switch=1.0,
            voltage_gradient_scale=1e6,
            voltage_dual_step_size=1.0,
            latency_dual_step_size=1.0,
            sensitivity_update_interval=300,
            sensitivity_perturbation_kw=50.0,
            v_min=V_MIN,
            v_max=V_MAX,
        ),
        tap_schedule=TapSchedule(
            (
                (
                    1800,
                    TapPosition(
                        regulators={
                            "creg2a": tap(10),
                            "creg2b": tap(10),
                            "creg2c": tap(10),
                        }
                    ),
                ),
            )
        ),
    )


def ieee123_experiment(training_trace: TrainingTrace | None = None) -> dict:
    """IEEE 123-bus: four DC sites across zones."""
    sys = SYSTEMS["ieee123"]()
    return dict(
        sys=sys,
        dc_sites={
            "z1_sw": DCSite(
                bus="8",
                bus_kv=4.16,
                base_kw_per_phase=280.0,
                models=(with_ramp(deploy("Llama-3.1-8B", 800), 1200, t_start=500, t_end=1000),),
                seed=0,
                total_gpu_capacity=1200,
            ),
            "z2_nw": DCSite(
                bus="23",
                bus_kv=4.16,
                base_kw_per_phase=280.0,
                models=(with_ramp(deploy("Qwen3-30B-A3B", 460), 600, t_start=1500, t_end=2500),),
                seed=17,
                total_gpu_capacity=1200,
            ),
            "z3_se": DCSite(
                bus="60",
                bus_kv=4.16,
                base_kw_per_phase=224.0,
                models=(
                    with_ramp(deploy("Llama-3.1-70B", 64), 96, t_start=700, t_end=1100),
                    deploy("Llama-3.1-405B", 72),
                ),
                seed=34,
                total_gpu_capacity=960,
            ),
            "z4_ne": DCSite(
                bus="105",
                bus_kv=4.16,
                base_kw_per_phase=224.0,
                models=(with_ramp(deploy("Qwen3-235B-A22B", 96), 56, t_start=2000, t_end=2500),),
                seed=51,
                total_gpu_capacity=960,
            ),
        },
        pv_systems=[
            PVSystemSpec(bus="18", bus_kv=4.16, peak_kw=100.0),
            PVSystemSpec(bus="48", bus_kv=4.16, peak_kw=250.0),
            PVSystemSpec(bus="57", bus_kv=4.16, peak_kw=200.0),
        ],
        time_varying_loads=[
            TimeVaryingLoadSpec(bus="13", bus_kv=4.16, peak_kw=20.0),
            TimeVaryingLoadSpec(bus="86", bus_kv=4.16, peak_kw=20.0),
            TimeVaryingLoadSpec(bus="114", bus_kv=4.16, peak_kw=20.0),
        ],
        training_base=None,
        ofo_config=OFOConfig(
            primal_step_size=0.05,
            w_throughput=0.0001,
            w_switch=1.0,
            voltage_gradient_scale=1e6,
            voltage_dual_step_size=0.3,
            latency_dual_step_size=1.0,
            sensitivity_update_interval=300,
            sensitivity_perturbation_kw=10.0,
            v_min=V_MIN,
            v_max=V_MAX,
        ),
        tap_schedule=None,
    )


EXPERIMENTS: dict[str, Callable[..., dict]] = {
    "ieee13": ieee13_experiment,
    "ieee34": ieee34_experiment,
    "ieee123": ieee123_experiment,
}
