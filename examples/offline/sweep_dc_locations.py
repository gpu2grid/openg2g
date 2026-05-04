"""Unified DC location sweep: 1-D, 2-D, or zone-constrained.

Automatically selects sweep mode based on the system configuration:
  - 1 DC site, no zones  -> 1-D sweep: sweep single DC across candidate buses,
                            with tap optimization and 4-case comparison
  - 2+ DC sites, no zones -> 2-D sweep: sweep all unordered pairs of candidate
                            buses, OFO control for each site, heatmap output
  - N DC sites with zones -> zone-constrained sweep (3 phases):
      Phase 1 (Screening):  Sweep each zone independently while holding other
                            zones at default buses.  Keep top-K per zone.
                            Uses coarse dt (default 60 s) over the full
                            simulation duration (e.g. 3600 s) for fast ranking.
      Phase 2 (Combination): Cartesian product of top-K per zone.  Uses native
                            config resolution (e.g. dt=0.1 s) but only 60 s
                            duration as a stress test with full DC capacity.
      Phase 3 (Refinement):  Optional (--refine).  Iteratively re-sweep each
                            zone from the Phase 2 winner.  Uses native config
                            resolution over the full simulation duration.

Usage:
    python sweep_dc_locations.py --system ieee13
    python sweep_dc_locations.py --system ieee34
    python sweep_dc_locations.py --system ieee123
    python sweep_dc_locations.py --system ieee123 --top-k 6 --refine
"""

from __future__ import annotations

import csv
import itertools
import logging
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from openg2g.controller.ofo import (
    LogisticModelStore,
    OFOBatchSizeController,
    OFOConfig,
)
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.config import (
    DatacenterConfig,
    InferenceModelSpec,
    PowerAugmentationConfig,
    ReplicaSchedule,
    TrainingRun,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.config import TapPosition, TapSchedule
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import compute_allbus_voltage_stats

from systems import (
    DT_CTRL,
    DT_DC,
    DT_GRID,
    POWER_AUG,
    SPECS_CACHE_DIR,
    SYSTEMS,
    TAP_STEP,
    TOTAL_DURATION_S,
    TRAINING_TRACE_PATH,
    V_MAX,
    V_MIN,
    DCSite,
    PVSystemSpec,
    TimeVaryingLoadSpec,
    all_model_specs,
    deploy,
    tap,
    with_ramp,
)

logger = logging.getLogger("sweep_dc_locations")

T_TOTAL_S = 3600

# Backward-compatible aliases (used by other scripts)
PVSystemConfig = PVSystemSpec
TimeVaryingLoadConfig = TimeVaryingLoadSpec


# ── Profile generation helpers ───────────────────────────────────────────────


def _smooth_bump(t: float, t_center: float, half_width: float) -> float:
    dt = abs(t - t_center)
    if dt >= half_width:
        return 0.0
    x = dt / half_width
    return (1 - x * x) ** 2


def _smoothstep(t: float, t_start: float, t_end: float) -> float:
    """Cubic Hermite smoothstep: zero-derivative at both endpoints.

    Returns 0 for ``t <= t_start``, 1 for ``t >= t_end``, and a smooth
    S-curve in between (3x² - 2x³). This is the standard ease-in/ease-out
    function used for physically plausible PV ramps — solar power doesn't
    transition linearly because the sun's elevation angle changes smoothly.
    """
    if t_end <= t_start:
        return 1.0 if t >= t_start else 0.0
    if t <= t_start:
        return 0.0
    if t >= t_end:
        return 1.0
    x = (t - t_start) / (t_end - t_start)
    return x * x * (3.0 - 2.0 * x)


def _irregular_fluct(t: float, seed: float = 0.0) -> float:
    """Irregular fluctuation via superposition of incommensurate frequencies.

    Returns a value centred around 1.0 with ~+-15% variation.
    The use of irrational-ratio periods avoids visible periodicity.
    """
    s = seed
    f1 = 0.06 * math.sin(2 * math.pi * t / 173.0 + s)
    f2 = 0.05 * math.sin(2 * math.pi * t / 97.3 + s * 2.3)
    f3 = 0.04 * math.sin(2 * math.pi * t / 251.7 + s * 0.7)
    f4 = 0.03 * math.sin(2 * math.pi * t / 41.9 + s * 4.1)
    f5 = 0.02 * math.sin(2 * math.pi * t / 317.3 + s * 1.9)
    return 1.0 + f1 + f2 + f3 + f4 + f5


def pv_profile_kw(t: float, peak_kw: float, site_idx: int = 0) -> float:
    """Solar PV output (kW per phase) with per-site cloud patterns.

    Each site has a distinct profile with different cloud events, trends, and
    fluctuation seeds so the curves are visually distinguishable.
    """
    T = T_TOTAL_S
    if site_idx == 0:
        # Site 0: Declining afternoon output with two cloud dips
        trend = 0.85 - 0.30 * (t / T)
        cloud = 1.0
        cloud -= 0.55 * _smooth_bump(t, 600, 120)
        cloud -= 0.40 * _smooth_bump(t, 2100, 180)
        fluct = _irregular_fluct(t, seed=0.3)
        return max(0.0, peak_kw * trend * max(cloud, 0.05) * fluct)
    elif site_idx == 1:
        # Site 1: Midday peak with passing cloud bank
        ramp = 0.55 + 0.40 * _smooth_bump(t, 1200, 900)
        cloud = 1.0
        cloud -= 0.60 * _smooth_bump(t, 1680, 240)
        cloud -= 0.25 * _smooth_bump(t, 2400, 150)
        fluct = _irregular_fluct(t, seed=2.1)
        return max(0.0, peak_kw * ramp * max(cloud, 0.05) * fluct)
    else:
        # Site 2+: Morning ramp, sustained high, late cloud event
        # Seed varies by site_idx so additional PV sites get distinct curves
        ramp = 0.30 + 0.65 * min(1.0, t / 900.0)
        cloud = 1.0
        cloud -= 0.70 * _smooth_bump(t, 2700, 300)
        cloud -= 0.30 * _smooth_bump(t, 1200, 100)
        fluct = _irregular_fluct(t, seed=2.0 + site_idx * 3.7)
        return max(0.0, peak_kw * ramp * max(cloud, 0.05) * fluct)


def load_profile_kw(t: float, peak_kw: float, site_idx: int = 0) -> float:
    fluct_period = 130.0 + site_idx * 37
    fluct = 1.0 + 0.06 * math.sin(2 * math.pi * t / fluct_period + site_idx * 1.4)

    if site_idx == 0:
        base = 0.15 + 0.85 * _smooth_bump(t, 2280, 1400)
        surge = 0.20 * _smooth_bump(t, 2280, 180)
        return max(0.0, peak_kw * (base + surge) * fluct)
    elif site_idx == 1:
        base = 0.10
        base += 0.50 * _smooth_bump(t, 1500, 600)
        base += 0.80 * _smooth_bump(t, 2880, 500)
        return max(0.0, peak_kw * base * fluct)
    elif site_idx == 2:
        base = 0.80 - 0.55 * _smooth_bump(t, 1800, 1200)
        surge = 0.70 * _smooth_bump(t, 2520, 400)
        return max(0.0, peak_kw * (base + surge) * fluct)
    elif site_idx == 3:
        base = 0.10 + 0.90 * _smooth_bump(t, 3120, 800)
        return max(0.0, peak_kw * base * fluct)
    else:
        base = 0.10
        base += 0.60 * _smooth_bump(t, 1080, 300)
        base += 0.75 * _smooth_bump(t, 2100, 350)
        base += 0.90 * _smooth_bump(t, 3300, 300)
        return max(0.0, peak_kw * base * fluct)


# ── Randomized profile generators ────────────────────────────────────────────
#
# These produce 1-hour episodes (T_TOTAL_S ≈ 3600 s). Unlike the legacy
# pv_profile_kw / load_profile_kw which were shaped like multi-hour curves
# (sunrise→noon→sunset, residential demand peak), these are tuned for the
# 1-hour episode regime where:
#   - PV power is roughly constant (sun position barely moves) with
#     occasional cloud dips as the dominant variability source.
#   - TVL has multiple plausible shapes (flat, increasing, decreasing,
#     peaked, valley) depending on the load type. The shape is chosen by
#     params["shape"] and the parameters drive its details.
#
# The params dict is generated once per scenario in train_ppo.randomize_scenario
# (broad mode) and stored on PVSystemSpec.profile_params / TVLSpec.profile_params
# so the simulator can replay deterministically.


def pv_profile_random(t: float, peak_kw: float, params: dict) -> float:
    """Multi-shape PV profile with random cloud events (1-hour episode).

    params["shape"] picks the envelope:
        "flat"             constant baseline 0.75-0.95
        "rising_falling"   smooth bump from low_baseline up to high_baseline and back
        "morning_ramp"     low → high over a short ramp window, sustained high after
        "afternoon_decline" sustained high then ramp down at the end
        "midday_dip"       high baseline with a substantial mid-episode dip

    Envelope-specific params (all optional, sensible defaults below):
        flat:              baseline
        rising_falling:    low_baseline, high_baseline, peak_t, half_width
        morning_ramp:      low_baseline, high_baseline, ramp_end
        afternoon_decline: low_baseline, high_baseline, ramp_start
        midday_dip:        high_baseline, dip_t, dip_half_width, dip_depth

    Common keys (applied on top of any envelope):
        clouds:            list of (t_center_s, half_width_s, depth_frac) extra cloud-dip events
        noise_amp:         amplitude of small irrational-frequency fluctuations (0.0-0.1)
        noise_seed:        seed for the fluctuation phase
    """
    shape = params.get("shape", "flat")
    T = float(T_TOTAL_S)

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
        env = 0.85  # safe fallback

    # Apply cloud events (independent of envelope shape)
    for tc, hw, depth in params.get("clouds", ()):
        env -= float(depth) * _smooth_bump(t, float(tc), float(hw))

    # Floor PV at 15% of nameplate even during heavy clouds — diffuse
    # irradiance keeps real PV output above zero in practice.
    env = max(0.15, env)
    noise_amp = float(params.get("noise_amp", 0.0))
    if noise_amp > 0:
        f = _irregular_fluct(t, seed=float(params.get("noise_seed", 0.0)))
        env *= 1.0 + (noise_amp / 0.20) * (f - 1.0)
    return max(0.0, peak_kw * env)


def tvl_profile_random(t: float, peak_kw: float, params: dict) -> float:
    """Multi-shape TVL profile (1-hour episode).

    params:
        shape:       one of "flat", "increasing", "decreasing", "peaked", "valley"
        ... shape-specific keys (see body below) ...
        noise_amp:   amplitude of small fluctuations (0.0-0.08)
        noise_seed:  seed for the fluctuation phase
    """
    shape = params.get("shape", "peaked")
    T = float(T_TOTAL_S)
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
        # Unknown shape -> safe default
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
    """Evaluate a PV/TVL profile at simulated time ``t``.

    Dispatch order:
      1. ``profile_kind="random_flat"`` + ``profile_params`` -> pv_profile_random
      2. ``profile_kind="random_shape"`` + ``profile_params`` -> tvl_profile_random
      3. ``csv_data`` not None -> CSV interpolation
      4. ``profile_fn`` (legacy analytical pv_profile_kw / load_profile_kw)

    The legacy path applies the optional ``peak_t_shift_s`` / ``time_warp``
    transform: ``t_eff = (t - peak_t_shift_s) / time_warp``. The randomized
    kinds bake their shape into profile_params and ignore these.
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


# ── Scenario Grid ────────────────────────────────────────────────────────────


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


# ── Bus discovery ────────────────────────────────────────────────────────────


def discover_candidate_buses(
    case_dir: Path,
    master_file: str,
    target_bus_kv: float,
    exclude: set[str],
) -> list[str]:
    """Discover all 3-phase buses at the target voltage level."""
    from opendssdirect import dss

    dss.Basic.ClearAll()
    master_path = str(Path(case_dir) / master_file)
    dss.Text.Command(f'Compile "{master_path}"')
    dss.Solution.SolveNoControl()

    target_kv_ln = target_bus_kv / math.sqrt(3.0)
    tolerance = 0.05 * target_kv_ln

    bus_phases: dict[str, set[int]] = {}
    for name in dss.Circuit.AllNodeNames():
        parts = name.split(".")
        bus = parts[0].lower()
        phase = int(parts[1]) if len(parts) > 1 else 0
        if bus not in bus_phases:
            bus_phases[bus] = set()
        bus_phases[bus].add(phase)

    exclude_lower = {b.lower() for b in exclude}
    candidates = []

    for bus_name in dss.Circuit.AllBusNames():
        if bus_name.lower() in exclude_lower:
            continue
        phases = bus_phases.get(bus_name.lower(), set())
        if not {1, 2, 3}.issubset(phases):
            continue
        dss.Circuit.SetActiveBus(bus_name)
        kv_base = dss.Bus.kVBase()
        if abs(kv_base - target_kv_ln) <= tolerance:
            candidates.append(bus_name)

    return sorted(candidates)


# ══════════════════════════════════════════════════════════════════════════════
# 1-D SWEEP (single DC site)
# ══════════════════════════════════════════════════════════════════════════════


MAX_TAP_ITERATIONS = 20


def extract_all_voltages(grid_states, exclude_buses=("rg60", "sourcebus")):
    exclude = {b.lower() for b in exclude_buses}
    result = {}
    for gs in grid_states:
        for bus in gs.voltages.buses():
            if bus.lower() in exclude:
                continue
            if bus not in result:
                result[bus] = {"a": [], "b": [], "c": []}
            pv = gs.voltages[bus]
            result[bus]["a"].append(pv.a)
            result[bus]["b"].append(pv.b)
            result[bus]["c"].append(pv.c)
    return result


def find_violations(voltages_dict, v_min=0.95, v_max=1.05):
    violations = []
    for bus, phases in voltages_dict.items():
        for phase_name, values in phases.items():
            arr = np.array(values)
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                continue
            vmin = float(np.min(valid))
            vmax = float(np.max(valid))
            if vmin < v_min:
                violations.append((bus, phase_name, "under", vmin, v_min - vmin))
            if vmax > v_max:
                violations.append((bus, phase_name, "over", vmax, vmax - v_max))
    return violations


def optimize_taps_for_bus(
    *,
    sys: dict,
    dc_bus: str,
    bus_kv: float,
    base_kw_per_phase: float,
    connection_type: str,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    training: TrainingRun | None,
    replica_schedules: dict[str, ReplicaSchedule] | None,
    initial_tap_position: TapPosition,
    total_gpu_capacity: int = 0,
    pv_systems: list | None = None,
    time_varying_loads: list | None = None,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    t_total_s: int = T_TOTAL_S,
    t_analysis_start: int = 0,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
) -> TapPosition:
    regs = dict(initial_tap_position.regulators)
    reg_names = list(regs.keys())

    dt_coarse = Fraction(60)

    for iteration in range(MAX_TAP_ITERATIONS):
        tap_pos = TapPosition(regulators=dict(regs))
        logger.debug(
            "  Tap opt iter %d: %s",
            iteration,
            ", ".join(f"{k}=%.5f (%+d)" % (regs[k], round((regs[k] - 1.0) / TAP_STEP)) for k in reg_names),
        )

        dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=base_kw_per_phase)

        workload_kwargs: dict = {"inference_data": inference_data}
        if replica_schedules is not None:
            workload_kwargs["replica_schedules"] = replica_schedules
        if training is not None:
            workload_kwargs["training"] = training
        workload = OfflineWorkload(**workload_kwargs)

        dc = OfflineDatacenter(
            dc_config,
            workload,
            name="dc",
            dt_s=dt_coarse,
            seed=0,
            power_augmentation=power_augmentation,
            total_gpu_capacity=total_gpu_capacity,
        )

        grid = ScenarioOpenDSSGrid(
            pv_systems=pv_systems or [],
            time_varying_loads=time_varying_loads or [],
            source_pu=sys["source_pu"],
            dss_case_dir=sys["dss_case_dir"],
            dss_master_file=sys["dss_master_file"],
            dt_s=dt_coarse,
            initial_tap_position=tap_pos,
        )
        grid.attach_dc(dc, bus=dc_bus, connection_type=connection_type, power_factor=dc_config.power_factor)

        ctrl = TapScheduleController(schedule=TapSchedule(()), dt_s=dt_coarse)
        coord = Coordinator(
            datacenters=[dc],
            grid=grid,
            controllers=[ctrl],
            total_duration_s=t_total_s,
        )
        log = coord.run()

        analysis_states = [gs for gs in log.grid_states if gs.time_s >= t_analysis_start]
        if len(analysis_states) < 2:
            analysis_states = log.grid_states

        voltages = extract_all_voltages(analysis_states)
        violations = find_violations(voltages, v_min=v_min, v_max=v_max)

        if not violations:
            logger.debug("    No violations. Taps optimized.")
            return tap_pos

        phase_adjust = {"a": 0, "b": 0, "c": 0}
        for _bus, phase, vtype, _val, _mag in violations:
            if vtype == "under":
                phase_adjust[phase] = max(phase_adjust[phase], 1)
            elif vtype == "over":
                phase_adjust[phase] = min(phase_adjust[phase], -1)

        has_conflict = False
        for phase in ("a", "b", "c"):
            under = any(v[1] == phase and v[2] == "under" for v in violations)
            over = any(v[1] == phase and v[2] == "over" for v in violations)
            if under and over:
                has_conflict = True
        if has_conflict:
            logger.debug("    Conflicting violations - cannot resolve. Stopping.")
            return tap_pos

        prev_regs = dict(regs)
        for rname in reg_names:
            rname_lower = rname.lower()
            if rname_lower.endswith("a"):
                regs[rname] += phase_adjust["a"] * TAP_STEP
            elif rname_lower.endswith("b"):
                regs[rname] += phase_adjust["b"] * TAP_STEP
            elif rname_lower.endswith("c"):
                regs[rname] += phase_adjust["c"] * TAP_STEP

        for rname in reg_names:
            regs[rname] = max(0.9, min(1.1, regs[rname]))

        if regs == prev_regs:
            logger.debug("    Taps clamped at limits. Stopping.")
            return tap_pos

    return TapPosition(regulators=dict(regs))


def optimize_taps_multiscenario(
    *,
    sys: dict,
    dc_bus: str,
    bus_kv: float,
    base_kw_per_phase: float,
    connection_type: str,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    training: TrainingRun | None,
    replica_schedules: dict[str, ReplicaSchedule] | None,
    initial_tap_position: TapPosition,
    tap_schedule_entries: list[tuple[float, TapPosition]] | None = None,
    training_t_start: float = 0,
    training_t_end: float = 0,
    ramp_t_end: float = 0,
    total_gpu_capacity: int = 0,
    pv_systems: list | None = None,
    time_varying_loads: list | None = None,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    total_duration_s: int = TOTAL_DURATION_S,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
) -> dict[str, TapPosition]:
    """Optimize taps for multiple scenarios (inference, training, low_load).

    Parameters
    ----------
    training_t_start, training_t_end : float
        Timing window for the training run (used for scenario splitting).
    ramp_t_end : float
        End time of the inference ramp (used for low_load scenario).
    """

    # Build scenario base taps from tap_schedule_entries
    bases: dict[str, TapPosition] = {"inference": initial_tap_position}
    sorted_entries = sorted(tap_schedule_entries or [], key=lambda e: e[0])

    if len(sorted_entries) >= 1 and training is not None:
        bases["training"] = sorted_entries[0][1]

    if len(sorted_entries) >= 2 and replica_schedules is not None:
        bases["low_load"] = sorted_entries[-1][1]

    COARSE_TICK = 60

    def _round_up(t: int) -> int:
        return ((t + COARSE_TICK - 1) // COARSE_TICK) * COARSE_TICK

    scenarios: list[tuple[str, int, int, TapPosition]] = []

    t_ts = int(training_t_start)
    t_te = int(training_t_end)

    if t_ts > 0:
        scenarios.append(("inference", _round_up(min(t_ts, 180)), 0, bases["inference"]))
    else:
        scenarios.append(("inference", _round_up(180), 0, bases["inference"]))

    if training is not None and "training" in bases:
        scenarios.append(("training", _round_up(t_te + 120), t_ts, bases["training"]))

    if replica_schedules is not None and "low_load" in bases:
        t_re = int(ramp_t_end) if ramp_t_end > 0 else total_duration_s
        scenarios.append(("low_load", _round_up(total_duration_s), t_re, bases["low_load"]))

    results = {}
    for name, t_total, t_start, init_taps in scenarios:
        taps_str = ", ".join(f"{k}=%+d" % round((v - 1.0) / TAP_STEP) for k, v in init_taps.regulators.items())
        logger.info(
            "    Scenario '%s': T=%d, analysis>=%d, base taps: %s",
            name,
            t_total,
            t_start,
            taps_str,
        )
        optimal = optimize_taps_for_bus(
            sys=sys,
            dc_bus=dc_bus,
            bus_kv=bus_kv,
            base_kw_per_phase=base_kw_per_phase,
            connection_type=connection_type,
            inference_data=inference_data,
            training_trace=training_trace,
            training=training,
            replica_schedules=replica_schedules,
            initial_tap_position=init_taps,
            total_gpu_capacity=total_gpu_capacity,
            pv_systems=pv_systems,
            time_varying_loads=time_varying_loads,
            power_augmentation=power_augmentation,
            t_total_s=t_total,
            t_analysis_start=t_start,
            v_min=v_min,
            v_max=v_max,
        )
        results[name] = optimal
        opt_str = ", ".join(f"{k}=%+d" % round((v - 1.0) / TAP_STEP) for k, v in optimal.regulators.items())
        logger.info("    -> %s taps: %s", name, opt_str)
    return results


def run_case_1d(
    *,
    sys: dict,
    dc_bus: str,
    bus_kv: float,
    base_kw_per_phase: float,
    case_name: str,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    all_models: tuple[InferenceModelSpec, ...],
    initial_taps: TapPosition,
    tap_schedule: TapSchedule | None,
    use_ofo: bool,
    ofo_config: OFOConfig | None = None,
    training: TrainingRun | None = None,
    replica_schedules: dict[str, ReplicaSchedule] | None = None,
    total_gpu_capacity: int = 0,
    pv_systems: list | None = None,
    time_varying_loads: list | None = None,
    total_duration_s: int = TOTAL_DURATION_S,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    connection_type: str = "wye",
    dt_dc: Fraction = DT_DC,
    dt_grid: Fraction = DT_GRID,
    dt_ctrl: Fraction = DT_CTRL,
    save_dir: Path | None = None,
) -> dict:
    """Run a single 1-D simulation case and return metrics dict."""
    dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=base_kw_per_phase)

    workload_kwargs: dict = {"inference_data": inference_data}
    if replica_schedules is not None:
        workload_kwargs["replica_schedules"] = replica_schedules
    if training is not None:
        workload_kwargs["training"] = training
    workload = OfflineWorkload(**workload_kwargs)

    dc = OfflineDatacenter(
        dc_config,
        workload,
        name="dc",
        dt_s=dt_dc,
        seed=0,
        power_augmentation=power_augmentation,
        total_gpu_capacity=total_gpu_capacity,
    )

    grid = ScenarioOpenDSSGrid(
        pv_systems=pv_systems or [],
        time_varying_loads=time_varying_loads or [],
        source_pu=sys["source_pu"],
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        dt_s=dt_grid,
        initial_tap_position=initial_taps,
    )
    grid.attach_dc(dc, bus=dc_bus, connection_type=connection_type, power_factor=dc_config.power_factor)

    controllers: list = []
    schedule = tap_schedule if tap_schedule else TapSchedule(())
    tap_ctrl = TapScheduleController(schedule=schedule, dt_s=dt_ctrl)
    controllers.append(tap_ctrl)

    if use_ofo and ofo_config is not None:
        ofo_ctrl = OFOBatchSizeController(
            all_models,
            datacenter=dc,
            grid=grid,
            models=logistic_models,
            config=ofo_config,
            dt_s=dt_ctrl,
        )
        controllers.append(ofo_ctrl)

    coord = Coordinator(
        datacenters=[dc],
        grid=grid,
        controllers=controllers,
        total_duration_s=total_duration_s,
    )
    log = coord.run()

    stats = compute_allbus_voltage_stats(log.grid_states, v_min=v_min, v_max=v_max)

    dc_kW = np.array([(s.power_w.a + s.power_w.b + s.power_w.c) / 3e3 for s in log.dc_states])
    avg_power = float(dc_kW.mean()) if dc_kW.size > 0 else 0.0

    result = {
        "violation_time_s": stats.violation_time_s,
        "worst_vmin": stats.worst_vmin,
        "worst_vmax": stats.worst_vmax,
        "integral_violation_pu_s": stats.integral_violation_pu_s,
        "avg_power_kw_per_phase": avg_power,
    }

    if save_dir is not None:
        from plot_all_figures import (
            extract_per_model_timeseries,
            plot_allbus_voltages_per_phase,
            plot_model_timeseries_4panel,
        )

        save_dir.mkdir(parents=True, exist_ok=True)
        time_s = np.array(log.time_s)
        plot_allbus_voltages_per_phase(
            log.grid_states,
            time_s,
            save_dir=save_dir,
            v_min=v_min,
            v_max=v_max,
            title_template=f"DC@{dc_bus} {case_name} — Voltage (Phase {{label}})",
        )
        if use_ofo:
            per_model = extract_per_model_timeseries(log.dc_states)
            plot_model_timeseries_4panel(
                per_model.time_s,
                per_model,
                model_labels=[m.model_label for m in all_models],
                regime_shading=False,
                save_path=save_dir / "OFO_results.png",
            )

    return result


def _plot_bus_comparison(all_rows: list[dict], save_path: Path) -> None:
    import matplotlib.pyplot as plt

    buses = [r["dc_bus"] for r in all_rows]
    cases = [
        ("baseline_no_tap", "Baseline (no tap)"),
        ("baseline_tap_change", "Baseline (tap change)"),
        ("ofo_no_tap", "OFO (no tap)"),
        ("ofo_tap_change", "OFO (tap change)"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(buses))
    width = 0.18

    for i, (prefix, label) in enumerate(cases):
        viol = [r.get(f"{prefix}_violation_time_s", float("nan")) for r in all_rows]
        integ = [r.get(f"{prefix}_integral_violation_pu_s", float("nan")) for r in all_rows]
        offset = (i - 1.5) * width
        ax1.bar(x + offset, viol, width, label=label)
        ax2.bar(x + offset, integ, width, label=label)

    ax1.set_xlabel("DC Bus")
    ax1.set_ylabel("Violation Time (s)")
    ax1.set_title("Voltage Violation Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(buses)
    ax1.legend(fontsize=8)

    ax2.set_xlabel("DC Bus")
    ax2.set_ylabel("Integral Violation (pu·s)")
    ax2.set_title("Integral Voltage Violation")
    ax2.set_xticks(x)
    ax2.set_xticklabels(buses)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison plot saved to: %s", save_path)


def main_1d(
    *,
    sys: dict,
    system: str,
    dc_site: DCSite,
    all_models: tuple[InferenceModelSpec, ...],
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    ofo_config: OFOConfig,
    training: TrainingRun | None = None,
    replica_schedules: dict[str, ReplicaSchedule] | None = None,
    tap_schedule_entries: list[tuple[float, TapPosition]] | None = None,
    training_t_start: float = 0,
    training_t_end: float = 0,
    ramp_t_end: float = 0,
    pv_systems: list | None = None,
    time_varying_loads: list | None = None,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    total_duration_s: int = TOTAL_DURATION_S,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    buses: list[str] | None = None,
    dt_override: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """1-D sweep: single DC site swept across candidate buses."""
    if replica_schedules is None:
        replica_schedules = {md.spec.model_label: sched for md, sched in dc_site.models}
    site_gpu_capacity = dc_site.total_gpu_capacity

    dt_dc = DT_DC
    dt_grid = DT_GRID
    dt_ctrl = DT_CTRL

    if dt_override is not None:
        frac = Fraction(dt_override) if "/" in dt_override else Fraction(int(dt_override))
        dt_dc = dt_grid = dt_ctrl = frac
        logger.info("Time resolution override: dt = %s s for all components", dt_override)

    save_dir = (
        output_dir if output_dir else (Path(__file__).resolve().parent / "outputs" / system / "sweep_dc_locations_1d")
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # Discover candidate buses
    if buses:
        candidate_buses = buses
        logger.info("Using user-specified buses: %s", candidate_buses)
    else:
        logger.info("Discovering candidate 3-phase buses at %.2f kV...", dc_site.bus_kv)
        candidate_buses = discover_candidate_buses(
            sys["dss_case_dir"],
            sys["dss_master_file"],
            dc_site.bus_kv,
            exclude=set(sys["exclude_buses"]),
        )
        logger.info("Found %d candidate buses: %s", len(candidate_buses), candidate_buses)

    if not candidate_buses:
        logger.error("No candidate buses found!")
        return

    initial_taps = sys["initial_taps"]

    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 1: Tap optimization for %d buses", len(candidate_buses))
    logger.info("=" * 70)

    optimal_taps: dict[str, dict[str, TapPosition]] = {}

    for dc_bus in candidate_buses:
        logger.info("")
        logger.info("  Bus %s: optimizing taps...", dc_bus)
        try:
            taps = optimize_taps_multiscenario(
                sys=sys,
                dc_bus=dc_bus,
                bus_kv=dc_site.bus_kv,
                base_kw_per_phase=dc_site.base_kw_per_phase,
                connection_type=dc_site.connection_type,
                inference_data=inference_data,
                training_trace=training_trace,
                training=training,
                replica_schedules=replica_schedules,
                initial_tap_position=initial_taps,
                tap_schedule_entries=tap_schedule_entries,
                training_t_start=training_t_start,
                training_t_end=training_t_end,
                ramp_t_end=ramp_t_end,
                total_gpu_capacity=site_gpu_capacity,
                pv_systems=pv_systems,
                time_varying_loads=time_varying_loads,
                power_augmentation=power_augmentation,
                total_duration_s=total_duration_s,
                v_min=v_min,
                v_max=v_max,
            )
            optimal_taps[dc_bus] = taps
        except Exception as e:
            logger.error("  Bus %s: tap optimization FAILED: %s", dc_bus, e)
            optimal_taps[dc_bus] = {"inference": initial_taps}

    # Save tap optimization results
    tap_csv_path = save_dir / "optimal_taps.csv"
    with open(tap_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        all_reg_names: list[str] = []
        for scenarios in optimal_taps.values():
            for tp in scenarios.values():
                for rn in tp.regulators:
                    if rn not in all_reg_names:
                        all_reg_names.append(rn)
        header = ["dc_bus", "scenario"]
        for rn in all_reg_names:
            header.extend([f"tap_{rn}", f"step_{rn}"])
        writer.writerow(header)
        for dc_bus, scenarios in optimal_taps.items():
            for scenario, tp in scenarios.items():
                row = [dc_bus, scenario]
                for rn in all_reg_names:
                    val = tp.regulators.get(rn, 1.0)
                    row.extend([f"{val:.5f}", round((val - 1.0) / TAP_STEP)])
                writer.writerow(row)
    logger.info("Tap optimization results saved to: %s", tap_csv_path)

    # Phase 2: Run 4 comparison cases per bus
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 2: Running 4 comparison cases for %d buses", len(candidate_buses))
    logger.info("=" * 70)

    all_rows = []
    total_cases = len(candidate_buses) * 4
    case_idx = 0

    # Build tap schedule from optimized per-bus taps
    for dc_bus in candidate_buses:
        bus_taps = optimal_taps.get(dc_bus, {})
        inference_taps = bus_taps.get("inference", initial_taps)

        bus_tap_schedule_entries = []
        if "training" in bus_taps and training is not None:
            bus_tap_schedule_entries.append((training_t_start, bus_taps["training"]))
        if "low_load" in bus_taps and replica_schedules is not None:
            t_low = ramp_t_end + 120 if ramp_t_end > 0 else total_duration_s
            bus_tap_schedule_entries.append((t_low, bus_taps["low_load"]))

        bus_tap_schedule = (
            TapSchedule(()) if not bus_tap_schedule_entries else TapSchedule(tuple(bus_tap_schedule_entries))
        )

        cases = [
            ("baseline_no_tap", False, False),
            ("baseline_tap_change", False, True),
            ("ofo_no_tap", True, False),
            ("ofo_tap_change", True, True),
        ]

        bus_results: dict[str, str | float] = {"dc_bus": dc_bus}

        for case_name, use_ofo, use_tap_change in cases:
            case_idx += 1
            logger.info("")
            logger.info("[%d/%d] Bus %s — %s", case_idx, total_cases, dc_bus, case_name)

            case_save_dir = save_dir / f"bus_{dc_bus}" / case_name

            try:
                result = run_case_1d(
                    sys=sys,
                    dc_bus=dc_bus,
                    bus_kv=dc_site.bus_kv,
                    base_kw_per_phase=dc_site.base_kw_per_phase,
                    case_name=case_name,
                    inference_data=inference_data,
                    training_trace=training_trace,
                    logistic_models=logistic_models,
                    all_models=all_models,
                    initial_taps=inference_taps,
                    tap_schedule=bus_tap_schedule if use_tap_change else None,
                    use_ofo=use_ofo,
                    ofo_config=ofo_config,
                    training=training,
                    replica_schedules=replica_schedules,
                    total_gpu_capacity=site_gpu_capacity,
                    pv_systems=pv_systems,
                    time_varying_loads=time_varying_loads,
                    total_duration_s=total_duration_s,
                    v_min=v_min,
                    v_max=v_max,
                    power_augmentation=power_augmentation,
                    connection_type=dc_site.connection_type,
                    dt_dc=dt_dc,
                    dt_grid=dt_grid,
                    dt_ctrl=dt_ctrl,
                    save_dir=case_save_dir,
                )
                logger.info(
                    "  -> viol=%.1fs  integral=%.4f pu·s  vmin=%.4f  vmax=%.4f",
                    result["violation_time_s"],
                    result["integral_violation_pu_s"],
                    result["worst_vmin"],
                    result["worst_vmax"],
                )
                for k, v in result.items():
                    bus_results[f"{case_name}_{k}"] = v
            except Exception as e:
                logger.error("  Bus %s — %s FAILED: %s", dc_bus, case_name, e)
                for k in [
                    "violation_time_s",
                    "worst_vmin",
                    "worst_vmax",
                    "integral_violation_pu_s",
                    "avg_power_kw_per_phase",
                ]:
                    bus_results[f"{case_name}_{k}"] = float("nan")

        all_rows.append(bus_results)

    # Save results
    if all_rows:
        csv_path = save_dir / f"results_{system}_sweep_dc_locations.csv"
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info("")
        logger.info("Results saved to: %s", csv_path)
        _plot_bus_comparison(all_rows, save_dir / f"comparison_{system}_sweep_dc_locations.png")

    # Summary table
    logger.info("")
    logger.info("=" * 160)
    logger.info("FULL COMPARISON: No-Tap vs Tap-Change vs OFO vs OFO+Tap-Change")
    logger.info("=" * 160)
    logger.info(
        "%-8s | %12s %12s | %12s %12s | %12s %12s | %12s %12s",
        "Bus",
        "NoTap V(s)",
        "NoTap I",
        "TapChg V(s)",
        "TapChg I",
        "OFO V(s)",
        "OFO I",
        "OFO+TC V(s)",
        "OFO+TC I",
    )
    logger.info("-" * 160)
    for row in all_rows:
        bus = row["dc_bus"]
        logger.info(
            "%-8s | %12.1f %12.4f | %12.1f %12.4f | %12.1f %12.4f | %12.1f %12.4f",
            bus,
            row.get("baseline_no_tap_violation_time_s", float("nan")),
            row.get("baseline_no_tap_integral_violation_pu_s", float("nan")),
            row.get("baseline_tap_change_violation_time_s", float("nan")),
            row.get("baseline_tap_change_integral_violation_pu_s", float("nan")),
            row.get("ofo_no_tap_violation_time_s", float("nan")),
            row.get("ofo_no_tap_integral_violation_pu_s", float("nan")),
            row.get("ofo_tap_change_violation_time_s", float("nan")),
            row.get("ofo_tap_change_integral_violation_pu_s", float("nan")),
        )

    logger.info("")
    logger.info("All outputs saved to: %s", save_dir)


# ══════════════════════════════════════════════════════════════════════════════
# 2-D SWEEP (multiple DC sites)
# ══════════════════════════════════════════════════════════════════════════════


def _plot_heatmaps(df: pd.DataFrame, buses: list[str], save_dir: Path) -> None:
    import matplotlib.pyplot as plt

    n = len(buses)
    bus_idx = {b: i for i, b in enumerate(buses)}

    for metric, label, fname in [
        ("violation_time_s", "Violation Time (s)", "heatmap_violation_time.png"),
        ("integral_violation_pu_s", "Integral Violation (pu·s)", "heatmap_integral_violation.png"),
    ]:
        mat = np.full((n, n), np.nan)

        for _, row in df.iterrows():
            i = bus_idx.get(row["bus_A"])
            j = bus_idx.get(row["bus_B"])
            if i is None or j is None:
                continue
            val = row[metric]
            mat[i, j] = val
            mat[j, i] = val

        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
        im = ax.imshow(mat, cmap="YlOrRd", aspect="equal", origin="lower")

        ax.set_xticks(range(n))
        ax.set_xticklabels(buses, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(buses, fontsize=9)

        ax.set_xlabel("DC2 Bus", fontsize=12)
        ax.set_ylabel("DC1 Bus", fontsize=12)
        ax.set_title(f"2-D DC Location Sweep: {label}", fontsize=14)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(label, fontsize=11)

        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                if np.isfinite(val):
                    text_color = "white" if val > (np.nanmax(mat) + np.nanmin(mat)) / 2 else "black"
                    fmt = f"{val:.0f}" if metric == "violation_time_s" else f"{val:.2f}"
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=7, color=text_color)

        fig.tight_layout()
        fig.savefig(save_dir / fname, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved heatmap: %s", fname)


def main_2d(
    *,
    sys: dict,
    system: str,
    dc_sites: dict[str, DCSite],
    all_models: tuple[InferenceModelSpec, ...],
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    ofo_config: OFOConfig,
    training: TrainingRun | None = None,
    pv_systems: list | None = None,
    time_varying_loads: list | None = None,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    total_duration_s: int = TOTAL_DURATION_S,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    dt_override: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """2-D sweep: multiple DC sites, sweep all unordered pairs of candidate buses."""
    dt_dc = DT_DC
    dt_grid = DT_GRID
    dt_ctrl = DT_CTRL

    if dt_override is not None:
        frac = Fraction(dt_override) if "/" in dt_override else Fraction(int(dt_override))
        dt_dc = dt_grid = dt_ctrl = frac
        logger.info("Time resolution override: dt = %s s for all components", dt_override)

    save_dir = (
        output_dir if output_dir else (Path(__file__).resolve().parent / "outputs" / system / "sweep_dc_locations_2d")
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # Discover candidate buses
    site_ids = list(dc_sites.keys())
    assert len(site_ids) >= 2, "Need at least 2 dc_sites for 2-D sweep"
    site_A_id = site_ids[0]
    site_B_id = site_ids[1]
    site_A = dc_sites[site_A_id]
    site_B = dc_sites[site_B_id]

    target_kv = site_A.bus_kv

    logger.info("Discovering candidate 3-phase buses at %.1f kV...", target_kv)
    candidate_buses = discover_candidate_buses(
        sys["dss_case_dir"],
        sys["dss_master_file"],
        target_kv,
        exclude=set(sys["exclude_buses"]),
    )
    logger.info("Found %d candidate buses: %s", len(candidate_buses), candidate_buses)

    if len(candidate_buses) < 2:
        logger.error("Need at least 2 candidate buses for 2-D sweep!")
        return

    specs_A = tuple(md.spec for md, _ in site_A.models)
    specs_B = tuple(md.spec for md, _ in site_B.models)
    inference_A = inference_data.filter_models(specs_A)
    inference_B = inference_data.filter_models(specs_B)

    initial_taps = sys["initial_taps"]
    exclude_buses = tuple(sys["exclude_buses"])

    # Generate all unordered pairs
    pairs = list(itertools.combinations(candidate_buses, 2))
    total = len(pairs)
    logger.info("Total pairs: %d (from %d buses)", total, len(candidate_buses))

    # Run sweep
    rows: list[dict] = []

    for idx, (bus_A, bus_B) in enumerate(pairs, start=1):
        logger.info("[%d/%d] DC1(%s)@%s + DC2(%s)@%s", idx, total, site_A_id, bus_A, site_B_id, bus_B)

        try:
            dc_config_A = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=site_A.base_kw_per_phase)
            dc_config_B = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=site_B.base_kw_per_phase)

            schedules_A = {md.spec.model_label: sched for md, sched in site_A.models}
            schedules_B = {md.spec.model_label: sched for md, sched in site_B.models}

            wl_kwargs_A: dict = {"inference_data": inference_A, "replica_schedules": schedules_A}
            if training is not None:
                wl_kwargs_A["training"] = training

            wl_kwargs_B: dict = {"inference_data": inference_B, "replica_schedules": schedules_B}
            if training is not None:
                wl_kwargs_B["training"] = training

            dc_A = OfflineDatacenter(
                dc_config_A,
                OfflineWorkload(**wl_kwargs_A),
                name=site_A_id,
                dt_s=dt_dc,
                seed=site_A.seed,
                power_augmentation=power_augmentation,
                total_gpu_capacity=site_A.total_gpu_capacity,
            )
            dc_B = OfflineDatacenter(
                dc_config_B,
                OfflineWorkload(**wl_kwargs_B),
                name=site_B_id,
                dt_s=dt_dc,
                seed=site_B.seed,
                power_augmentation=power_augmentation,
                total_gpu_capacity=site_B.total_gpu_capacity,
            )

            datacenters = {site_A_id: dc_A, site_B_id: dc_B}

            grid = ScenarioOpenDSSGrid(
                pv_systems=pv_systems or [],
                time_varying_loads=time_varying_loads or [],
                source_pu=sys["source_pu"],
                dss_case_dir=sys["dss_case_dir"],
                dss_master_file=sys["dss_master_file"],
                dt_s=dt_grid,
                initial_tap_position=initial_taps,
                exclude_buses=exclude_buses,
            )
            grid.attach_dc(
                dc_A, bus=bus_A, connection_type=site_A.connection_type, power_factor=dc_config_A.power_factor
            )
            grid.attach_dc(
                dc_B, bus=bus_B, connection_type=site_B.connection_type, power_factor=dc_config_B.power_factor
            )

            ofo_A = OFOBatchSizeController(
                specs_A,
                datacenter=dc_A,
                grid=grid,
                models=logistic_models,
                config=ofo_config,
                dt_s=dt_ctrl,
            )
            ofo_B = OFOBatchSizeController(
                specs_B,
                datacenter=dc_B,
                grid=grid,
                models=logistic_models,
                config=ofo_config,
                dt_s=dt_ctrl,
            )

            coord = Coordinator(
                datacenters=list(datacenters.values()),
                grid=grid,
                controllers=[ofo_A, ofo_B],
                total_duration_s=total_duration_s,
            )

            t0 = time.monotonic()
            log = coord.run()
            wall_s = time.monotonic() - t0

            vstats = compute_allbus_voltage_stats(
                log.grid_states,
                v_min=v_min,
                v_max=v_max,
                exclude_buses=exclude_buses,
            )

            row = {
                "bus_A": bus_A,
                "bus_B": bus_B,
                "violation_time_s": vstats.violation_time_s,
                "integral_violation_pu_s": vstats.integral_violation_pu_s,
                "worst_vmin": vstats.worst_vmin,
                "worst_vmax": vstats.worst_vmax,
                "wall_time_s": wall_s,
            }
            rows.append(row)
            logger.info(
                "  -> viol=%.1fs  integral=%.4f pu·s  vmin=%.4f  wall=%.1fs",
                vstats.violation_time_s,
                vstats.integral_violation_pu_s,
                vstats.worst_vmin,
                wall_s,
            )

        except Exception:
            logger.exception("Pair (%s, %s) failed; skipping.", bus_A, bus_B)

    # Save CSV
    if not rows:
        logger.warning("No successful runs; nothing to save.")
        return

    df = pd.DataFrame(rows)
    csv_path = save_dir / f"results_{system}_sweep_dc_locations_2d.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %d rows to %s", len(df), csv_path)

    _plot_heatmaps(df, candidate_buses, save_dir)
    logger.info("All outputs in: %s", save_dir)


# ══════════════════════════════════════════════════════════════════════════════
# ZONE-CONSTRAINED SWEEP (N DC sites, each in its own zone)
# ══════════════════════════════════════════════════════════════════════════════


def _run_multi_dc_case(
    *,
    bus_map: dict[str, str],
    site_ids: list[str],
    dc_sites: dict[str, DCSite],
    site_inference_data: dict[str, InferenceData],
    sys: dict,
    training_trace: TrainingTrace,
    training: TrainingRun | None,
    logistic_models: LogisticModelStore,
    ofo_config: OFOConfig,
    target_kv: float,
    initial_taps: TapPosition | None,
    exclude_buses: tuple[str, ...],
    pv_systems: list | None = None,
    time_varying_loads: list | None = None,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    dt_dc: Fraction = DT_DC,
    dt_grid: Fraction = DT_GRID,
    dt_ctrl: Fraction = DT_CTRL,
    total_duration_s: int | None = None,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    stress_test: bool = False,
) -> dict:
    """Run a single multi-DC simulation and return metrics dict.

    When stress_test=True, PV runs at constant peak power and time-varying
    loads are disabled.  Each datacenter uses an immediate inference ramp to
    fill its total_gpu_capacity, simulating the extreme case.
    """
    duration = total_duration_s if total_duration_s is not None else TOTAL_DURATION_S

    # Build datacenters
    datacenters = {}
    for sid in site_ids:
        site = dc_sites[sid]
        dc_config = DatacenterConfig(
            gpus_per_server=8,
            base_kw_per_phase=site.base_kw_per_phase,
        )
        # Schedules combine the (initial, ramps) pair on each (md, sched).
        replica_schedules: dict[str, ReplicaSchedule] = {md.spec.model_label: sched for md, sched in site.models}
        if stress_test:
            # Immediate per-model ramps to fill total GPU capacity
            current_gpus = sum(sched.initial * md.spec.gpus_per_replica for md, sched in site.models)
            total_cap = site.total_gpu_capacity or current_gpus
            scale = total_cap / current_gpus if current_gpus > 0 else 1.0
            if scale > 1.0:
                replica_schedules = {
                    md.spec.model_label: ReplicaSchedule(initial=sched.initial).ramp_to(
                        round(scale * sched.initial),
                        t_start=0,
                        t_end=1,
                    )
                    for md, sched in site.models
                }
        elif training is not None:
            pass  # training handled below
        wl_kwargs: dict = {"inference_data": site_inference_data[sid], "replica_schedules": replica_schedules}
        if not stress_test and training is not None:
            wl_kwargs["training"] = training

        dc = OfflineDatacenter(
            dc_config,
            OfflineWorkload(**wl_kwargs),
            name=sid,
            dt_s=dt_dc,
            seed=site.seed,
            power_augmentation=power_augmentation,
            total_gpu_capacity=site.total_gpu_capacity,
        )
        datacenters[sid] = dc

    # Build grid then attach DCs (master's imperative pattern).
    grid = ScenarioOpenDSSGrid(
        pv_systems=pv_systems or [],
        time_varying_loads=[] if stress_test else (time_varying_loads or []),
        source_pu=sys["source_pu"],
        constant_pv=stress_test,
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        dt_s=dt_grid,
        initial_tap_position=initial_taps,
        exclude_buses=exclude_buses,
    )
    dc_pf = DatacenterConfig().power_factor
    for sid in site_ids:
        grid.attach_dc(
            datacenters[sid],
            bus=bus_map[sid],
            connection_type=dc_sites[sid].connection_type,
            power_factor=dc_pf,
        )

    # Build OFO controllers
    controllers = []
    for sid in site_ids:
        site_specs = tuple(md.spec for md, _ in dc_sites[sid].models)
        ofo_ctrl = OFOBatchSizeController(
            site_specs,
            datacenter=datacenters[sid],
            grid=grid,
            models=logistic_models,
            config=ofo_config,
            dt_s=dt_ctrl,
        )
        controllers.append(ofo_ctrl)

    coord = Coordinator(
        datacenters=list(datacenters.values()),
        grid=grid,
        controllers=controllers,
        total_duration_s=duration,
    )

    t0 = time.monotonic()
    log = coord.run()
    wall_s = time.monotonic() - t0

    vstats = compute_allbus_voltage_stats(
        log.grid_states,
        v_min=v_min,
        v_max=v_max,
        exclude_buses=exclude_buses,
    )

    row: dict = {}
    for sid in site_ids:
        row[f"bus_{sid}"] = bus_map[sid]
    row.update(
        {
            "violation_time_s": vstats.violation_time_s,
            "integral_violation_pu_s": vstats.integral_violation_pu_s,
            "worst_vmin": vstats.worst_vmin,
            "worst_vmax": vstats.worst_vmax,
            "wall_time_s": wall_s,
        }
    )
    return row


def _log_results_table(
    df: pd.DataFrame,
    site_ids: list[str],
    title: str,
) -> None:
    """Print a formatted results table to the logger."""
    logger.info("")
    logger.info("=" * 120)
    logger.info(title)
    logger.info("=" * 120)
    bus_cols = [f"bus_{sid}" for sid in site_ids]
    header = "  ".join(f"{col:>10s}" for col in bus_cols)
    header += f"  {'Viol(s)':>12s} {'Integral':>12s} {'Vmin':>10s} {'Vmax':>10s} {'Wall(s)':>10s}"
    logger.info(header)
    logger.info("-" * 120)
    for _, row in df.iterrows():
        line = "  ".join(f"{row[col]:>10s}" for col in bus_cols)
        line += "  {:12.1f} {:12.4f} {:10.4f} {:10.4f} {:10.1f}".format(
            row["violation_time_s"],
            row["integral_violation_pu_s"],
            row["worst_vmin"],
            row["worst_vmax"],
            row["wall_time_s"],
        )
        logger.info(line)


def _plot_zoned_summary(
    df: pd.DataFrame,
    site_ids: list[str],
    save_dir: Path,
    system: str,
    suffix: str = "",
) -> None:
    """Plot bar chart of zone-constrained sweep results."""
    import matplotlib.pyplot as plt

    n_show = min(20, len(df))
    df_top = df.head(n_show)

    bus_cols = [f"bus_{sid}" for sid in site_ids]
    labels = ["/".join(str(row[c]) for c in bus_cols) for _, row in df_top.iterrows()]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, n_show * 0.7), 10))

    x = np.arange(n_show)
    ax1.bar(x, df_top["violation_time_s"].values, color="steelblue")
    ax1.set_ylabel("Violation Time (s)")
    title_tag = f" ({suffix})" if suffix else ""
    ax1.set_title(f"{system.upper()} DC Location Sweep{title_tag} — Violation Time (top {n_show})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    ax2.bar(x, df_top["integral_violation_pu_s"].values, color="coral")
    ax2.set_ylabel("Integral Violation (pu·s)")
    ax2.set_title(f"{system.upper()} DC Location Sweep{title_tag} — Integral Violation (top {n_show})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    fname = (
        f"Phase_2_combination_results_{system}.png"
        if suffix == "phase2"
        else f"comparison_{system}_zoned{('_' + suffix) if suffix else ''}.png"
    )
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison plot: %s", fname)


def _plot_screening_bars(
    screening_results: dict[str, pd.DataFrame],
    save_dir: Path,
    system: str,
) -> None:
    """Plot per-zone screening results as grouped bar charts."""
    import matplotlib.pyplot as plt

    n_zones = len(screening_results)
    fig, axes = plt.subplots(1, n_zones, figsize=(5 * n_zones, 5), squeeze=False)

    for ax, (sid, df) in zip(axes[0], screening_results.items(), strict=False):
        buses = df[f"bus_{sid}"].values
        viol = df["violation_time_s"].values
        x = np.arange(len(buses))
        ax.bar(x, viol, color="steelblue")
        ax.set_xlabel("Bus")
        ax.set_ylabel("Violation Time (s)")
        ax.set_title(f"Zone '{sid}' Screening")
        ax.set_xticks(x)
        ax.set_xticklabels(buses, rotation=45, ha="right", fontsize=9)

    fig.tight_layout()
    fname = f"Phase_1_screening_results_{system}.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved screening plot: %s", fname)


def _plot_refinement_iteration(
    iteration_results: dict[str, list[dict]],
    site_ids: list[str],
    save_dir: Path,
    system: str,
    iteration: int,
    prev_best: dict[str, str] | None = None,
    new_best: dict[str, str] | None = None,
) -> None:
    """Plot Phase 3 refinement results: one subplot per zone, bar charts of
    violation time and integral violation for each candidate bus.

    Marks the previous best bus (from the prior iteration or Phase 2) with a
    diamond marker and the new best bus (from this iteration) with a star.
    """
    import matplotlib.pyplot as plt

    n_zones = len(site_ids)
    fig, axes = plt.subplots(2, n_zones, figsize=(5 * n_zones, 8), squeeze=False)

    for col, sid in enumerate(site_ids):
        zone_rows = iteration_results.get(sid, [])
        if not zone_rows:
            axes[0][col].set_title(f"Zone '{sid}' (no data)")
            axes[1][col].set_title(f"Zone '{sid}' (no data)")
            continue

        df = pd.DataFrame(zone_rows).sort_values(
            ["violation_time_s", "integral_violation_pu_s"],
        )
        buses = df[f"bus_{sid}"].values
        x = np.arange(len(buses))

        prev_bus = prev_best.get(sid) if prev_best else None
        new_bus = new_best.get(sid) if new_best else None

        for ax_row, metric, ylabel, color, title_suffix in [
            (0, "violation_time_s", "Violation Time (s)", "steelblue", "Violation Time"),
            (1, "integral_violation_pu_s", "Integral Violation (pu·s)", "coral", "Integral Violation"),
        ]:
            ax = axes[ax_row][col]
            vals = df[metric].values
            ax.bar(x, vals, color=color)

            # Mark previous best with diamond, new best with star
            for idx, bus in enumerate(buses):
                bus_str = str(bus)
                if prev_bus and bus_str == prev_bus and bus_str != (new_bus or ""):
                    ax.plot(idx, vals[idx], "Dk", markersize=10, zorder=5)
                if new_bus and bus_str == new_bus:
                    ax.plot(
                        idx,
                        vals[idx],
                        "*",
                        color="gold",
                        markersize=16,
                        markeredgecolor="black",
                        markeredgewidth=0.8,
                        zorder=5,
                    )

            ax.set_ylabel(ylabel)
            ax.set_title(f"Zone '{sid}' — {title_suffix}")
            ax.set_xticks(x)
            ax.set_xticklabels(buses, rotation=45, ha="right", fontsize=9)

    # Add legend for markers
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="black", markersize=8, label="Previous best"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gold",
            markersize=14,
            markeredgecolor="black",
            label="New best",
        ),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.9)

    fig.suptitle(
        f"{system.upper()} Phase 3 Refinement — Iteration {iteration}",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fname = f"Phase_3_refinement_results_{system}_iter{iteration}.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved refinement plot: %s", fname)


def main_zoned(
    *,
    sys: dict,
    system: str,
    dc_sites: dict[str, DCSite],
    all_models: tuple[InferenceModelSpec, ...],
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    training: TrainingRun | None,
    logistic_models: LogisticModelStore,
    ofo_config: OFOConfig,
    zones: dict[str, list[str]],
    pv_systems: list | None = None,
    time_varying_loads: list | None = None,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    total_duration_s: int = TOTAL_DURATION_S,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    dt_override: str | None = None,
    dt_screening: str | None = None,
    output_dir: Path | None = None,
    top_k: int = 4,
    refine: bool = False,
) -> None:
    """Zone-constrained sweep with screening + combination + optional refinement.

    Three-phase design with different resolution/duration per phase:

    Phase 1 (Screening): Sweep each zone independently while holding other zones
        at their default config buses.  Uses coarse resolution (``dt_screening``,
        default 60 s) over the full simulation duration (typically 3600 s) for
        fast ranking.  Keep top-K per zone.
    Phase 2 (Combination): Cartesian product of top-K per zone.  Uses native
        config resolution (typically dt=0.1 s) but only 60 s duration as a
        stress test with full DC capacity and constant PV.
    Phase 3 (Refinement, optional): Starting from the Phase 2 winner, re-sweep
        each zone one at a time (holding others at best).  Uses native config
        resolution over the full simulation duration for accurate evaluation.
    """
    # Full-resolution dt (for Phase 2 & 3)
    dt_dc = DT_DC
    dt_grid = DT_GRID
    dt_ctrl = DT_CTRL
    logger.info("Phase 2/3 resolution: dt_dc=%s, dt_grid=%s, dt_ctrl=%s", dt_dc, dt_grid, dt_ctrl)

    # Screening dt (Phase 1): coarse resolution for fast ranking
    if dt_screening is not None:
        frac = Fraction(dt_screening) if "/" in dt_screening else Fraction(int(dt_screening))
        dt_dc_screen = dt_grid_screen = dt_ctrl_screen = frac
    elif dt_override is not None:
        frac = Fraction(dt_override) if "/" in dt_override else Fraction(int(dt_override))
        dt_dc_screen = dt_grid_screen = dt_ctrl_screen = frac
    else:
        # Default: 60 s for screening (coarse but fast)
        dt_dc_screen = dt_grid_screen = dt_ctrl_screen = Fraction(60)
    logger.info("Phase 1 screening resolution: dt = %s s", dt_dc_screen)

    save_dir = (
        output_dir
        if output_dir
        else (Path(__file__).resolve().parent / "outputs" / system / "sweep_dc_locations_zoned")
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    site_ids = list(dc_sites.keys())

    # Discover all 3-phase candidate buses
    ref_site = next(iter(dc_sites.values()))
    target_kv = ref_site.bus_kv

    logger.info("Discovering candidate 3-phase buses at %.1f kV...", target_kv)
    all_candidate_buses = set(
        discover_candidate_buses(
            sys["dss_case_dir"],
            sys["dss_master_file"],
            target_kv,
            exclude=set(sys["exclude_buses"]),
        )
    )
    logger.info("Found %d candidate 3-phase buses total", len(all_candidate_buses))

    # Build per-zone candidate lists (intersect zone bus lists with 3-phase candidates)
    zone_candidates: dict[str, list[str]] = {}
    for site_id in site_ids:
        zone_buses = zones.get(site_id, [])
        if not zone_buses:
            logger.warning("Site '%s' has no zone defined; using all candidate buses", site_id)
            zone_candidates[site_id] = sorted(all_candidate_buses)
        else:
            zone_set = {b.lower() for b in zone_buses}
            candidates = sorted(b for b in all_candidate_buses if b.lower() in zone_set)
            zone_candidates[site_id] = candidates
        logger.info(
            "  Zone '%s': %d candidate buses: %s",
            site_id,
            len(zone_candidates[site_id]),
            zone_candidates[site_id],
        )

    # Check for empty zones
    for site_id, cands in zone_candidates.items():
        if not cands:
            logger.error("Zone '%s' has no 3-phase candidate buses! Aborting.", site_id)
            return

    # Prepare per-site inference data
    site_inference_data: dict[str, InferenceData] = {}
    for sid in site_ids:
        site_specs = tuple(md.spec for md, _ in dc_sites[sid].models)
        site_inference_data[sid] = inference_data.filter_models(site_specs)

    initial_taps = sys["initial_taps"]
    exclude_buses = tuple(sys["exclude_buses"])

    # Shared kwargs for _run_multi_dc_case (without dt, added per-phase)
    _base_kwargs = dict(
        site_ids=site_ids,
        dc_sites=dc_sites,
        site_inference_data=site_inference_data,
        sys=sys,
        training_trace=training_trace,
        training=training,
        logistic_models=logistic_models,
        ofo_config=ofo_config,
        target_kv=target_kv,
        initial_taps=initial_taps,
        exclude_buses=exclude_buses,
        pv_systems=pv_systems,
        time_varying_loads=time_varying_loads,
        power_augmentation=power_augmentation,
        v_min=v_min,
        v_max=v_max,
    )
    screening_kwargs = dict(
        **_base_kwargs,
        dt_dc=dt_dc_screen,
        dt_grid=dt_grid_screen,
        dt_ctrl=dt_ctrl_screen,
    )
    full_res_kwargs = dict(
        **_base_kwargs,
        dt_dc=dt_dc,
        dt_grid=dt_grid,
        dt_ctrl=dt_ctrl,
    )

    # Default bus map from config
    default_bus_map = {sid: dc_sites[sid].bus for sid in site_ids}

    # ── Phase 1: Screening ───────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1: SCREENING — sweep each zone independently")
    logger.info("=" * 80)
    logger.info("Default bus map: %s", default_bus_map)

    screening_results: dict[str, pd.DataFrame] = {}
    top_k_per_zone: dict[str, list[str]] = {}

    total_screening = sum(len(zone_candidates[sid]) for sid in site_ids)
    screen_idx = 0

    for sweep_sid in site_ids:
        zone_rows: list[dict] = []
        cands = zone_candidates[sweep_sid]
        logger.info("")
        logger.info("── Screening zone '%s': %d candidates ──", sweep_sid, len(cands))

        for bus in cands:
            screen_idx += 1
            # Use default bus map but override the zone being swept
            bus_map = dict(default_bus_map)
            bus_map[sweep_sid] = bus
            combo_str = ", ".join(f"{sid}@{b}" for sid, b in bus_map.items())
            logger.info(
                "[%d/%d] Screening %s: %s",
                screen_idx,
                total_screening,
                sweep_sid,
                combo_str,
            )

            try:
                row = _run_multi_dc_case(bus_map=bus_map, **screening_kwargs)
                zone_rows.append(row)
                logger.info(
                    "  -> viol=%.1fs  integral=%.4f  vmin=%.4f  wall=%.1fs",
                    row["violation_time_s"],
                    row["integral_violation_pu_s"],
                    row["worst_vmin"],
                    row["wall_time_s"],
                )
            except Exception:
                logger.exception("  Screening %s@%s failed; skipping.", sweep_sid, bus)

        if not zone_rows:
            logger.error("Zone '%s' screening produced no results!", sweep_sid)
            return

        df_zone = pd.DataFrame(zone_rows).sort_values(
            ["violation_time_s", "integral_violation_pu_s"],
        )
        screening_results[sweep_sid] = df_zone

        # Select top-K (capped at available)
        k = min(top_k, len(df_zone))
        top_buses = df_zone[f"bus_{sweep_sid}"].head(k).tolist()
        top_k_per_zone[sweep_sid] = top_buses
        logger.info(
            "  Zone '%s' top-%d: %s",
            sweep_sid,
            k,
            top_buses,
        )

        # Save per-zone screening CSV
        csv_path = save_dir / f"Phase_1_screening_{system}_{sweep_sid}.csv"
        df_zone.to_csv(csv_path, index=False)

    # Save screening plots
    _plot_screening_bars(screening_results, save_dir, system)

    total_phase1 = screen_idx
    logger.info("")
    logger.info("Phase 1 complete: %d simulations", total_phase1)
    for sid, buses_list in top_k_per_zone.items():
        logger.info("  %s top-%d: %s", sid, len(buses_list), buses_list)

    # ── Phase 2: Combination (1-min stress test: constant PV + full DC) ──────
    # Duration must yield at least 2 grid steps; ensure >= 2 x dt_grid.
    min_duration = int(math.ceil(float(dt_grid) * 2))
    phase2_duration_s = max(60, min_duration)
    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "PHASE 2: COMBINATION — Cartesian product of top-%d per zone "
        "(%ds stress test: constant PV + full DC capacity, no time-varying loads)",
        top_k,
        phase2_duration_s,
    )
    logger.info("=" * 80)

    zone_bus_lists = [top_k_per_zone[sid] for sid in site_ids]
    combinations = list(itertools.product(*zone_bus_lists))
    total = len(combinations)
    logger.info(
        "Combinations: %d (%s)",
        total,
        " x ".join(str(len(z)) for z in zone_bus_lists),
    )

    phase2_rows: list[dict] = []

    for idx, combo in enumerate(combinations, start=1):
        bus_map = {sid: bus for sid, bus in zip(site_ids, combo, strict=False)}
        combo_str = ", ".join(f"{sid}@{bus}" for sid, bus in bus_map.items())
        logger.info("[%d/%d] %s", idx, total, combo_str)

        try:
            row = _run_multi_dc_case(
                bus_map=bus_map,
                total_duration_s=phase2_duration_s,
                stress_test=True,
                **full_res_kwargs,
            )
            phase2_rows.append(row)
            logger.info(
                "  -> viol=%.1fs  integral=%.4f  vmin=%.4f  wall=%.1fs",
                row["violation_time_s"],
                row["integral_violation_pu_s"],
                row["worst_vmin"],
                row["wall_time_s"],
            )
        except Exception:
            logger.exception("Combination %s failed; skipping.", combo_str)

    if not phase2_rows:
        logger.warning("Phase 2 produced no results!")
        return

    df_phase2 = pd.DataFrame(phase2_rows).sort_values(
        ["violation_time_s", "integral_violation_pu_s"],
    )
    csv_path = save_dir / f"Phase_2_combination_results_{system}.csv"
    df_phase2.to_csv(csv_path, index=False)
    logger.info("Saved Phase 2 results (%d rows) to %s", len(df_phase2), csv_path)

    _log_results_table(df_phase2, site_ids, "PHASE 2 RESULTS (sorted by violation time)")
    _plot_zoned_summary(df_phase2, site_ids, save_dir, system, suffix="phase2")

    # Best from Phase 2
    best_row = df_phase2.iloc[0]
    best_bus_map = {sid: str(best_row[f"bus_{sid}"]) for sid in site_ids}
    logger.info("")
    logger.info(
        "Phase 2 best: %s  (viol=%.1fs, integral=%.4f)",
        ", ".join(f"{sid}@{b}" for sid, b in best_bus_map.items()),
        best_row["violation_time_s"],
        best_row["integral_violation_pu_s"],
    )

    total_sims = total_phase1 + len(phase2_rows)

    # Save final summary with the best result from Phase 2
    final_csv_path = save_dir / f"sweep_dc_locations_final_results_{system}.csv"
    df_phase2.head(1).to_csv(final_csv_path, index=False)
    logger.info("Saved best result to: %s", final_csv_path)

    logger.info("")
    logger.info("=" * 80)
    logger.info("BEST RESULT (Phase 2)")
    logger.info("=" * 80)
    logger.info(
        "  %s  ->  viol=%.1fs  integral=%.4f  vmin=%.4f  vmax=%.4f",
        ", ".join(f"{sid}@{b}" for sid, b in best_bus_map.items()),
        best_row["violation_time_s"],
        best_row["integral_violation_pu_s"],
        best_row["worst_vmin"],
        best_row["worst_vmax"],
    )

    # ── Phase 3: Refinement (optional) ───────────────────────────────────────
    if not refine:
        logger.info("")
        logger.info("Phase 3 (refinement) skipped. Use --refine to enable.")
        logger.info("Total simulations: %d (Phase 1: %d, Phase 2: %d)", total_sims, total_phase1, len(phase2_rows))
        logger.info("All outputs in: %s", save_dir)
        return

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 3: REFINEMENT — re-sweep each zone from Phase 2 best")
    logger.info("=" * 80)

    current_best = dict(best_bus_map)
    improved = True
    iteration = 0

    # Collect per-iteration, per-zone results for CSV and plotting
    all_refinement_rows: dict[int, dict[str, list[dict]]] = {}

    while improved:
        iteration += 1
        improved = False
        prev_best = dict(current_best)  # snapshot before this iteration
        logger.info("")
        logger.info("── Refinement iteration %d ──", iteration)
        logger.info("Current best: %s", current_best)

        all_refinement_rows[iteration] = {}

        for sweep_sid in site_ids:
            cands = zone_candidates[sweep_sid]
            logger.info(
                "  Re-sweeping zone '%s': %d candidates",
                sweep_sid,
                len(cands),
            )

            best_for_zone = current_best[sweep_sid]
            best_viol = float("inf")
            best_integral = float("inf")
            zone_rows: list[dict] = []

            for bus in cands:
                bus_map = dict(current_best)
                bus_map[sweep_sid] = bus
                combo_str = ", ".join(f"{sid}@{b}" for sid, b in bus_map.items())

                try:
                    row = _run_multi_dc_case(bus_map=bus_map, **full_res_kwargs)
                    total_sims += 1
                    viol = row["violation_time_s"]
                    integ = row["integral_violation_pu_s"]
                    logger.info(
                        "    %s@%s -> viol=%.1fs  integral=%.4f",
                        sweep_sid,
                        bus,
                        viol,
                        integ,
                    )
                    zone_rows.append(row)
                    if (viol, integ) < (best_viol, best_integral):
                        best_viol = viol
                        best_integral = integ
                        best_for_zone = bus
                except Exception:
                    logger.exception("    %s@%s failed; skipping.", sweep_sid, bus)

            all_refinement_rows[iteration][sweep_sid] = zone_rows

            if best_for_zone != current_best[sweep_sid]:
                logger.info(
                    "  Zone '%s' improved: %s -> %s (viol=%.1fs)",
                    sweep_sid,
                    current_best[sweep_sid],
                    best_for_zone,
                    best_viol,
                )
                current_best[sweep_sid] = best_for_zone
                improved = True
            else:
                logger.info(
                    "  Zone '%s' unchanged: %s",
                    sweep_sid,
                    current_best[sweep_sid],
                )

        # Save per-iteration CSV and plot
        for sweep_sid in site_ids:
            zone_rows = all_refinement_rows[iteration].get(sweep_sid, [])
            if zone_rows:
                df_iter_zone = pd.DataFrame(zone_rows).sort_values(
                    ["violation_time_s", "integral_violation_pu_s"],
                )
                csv_path = save_dir / (f"Phase_3_refinement_results_{system}_iter{iteration}_{sweep_sid}.csv")
                df_iter_zone.to_csv(csv_path, index=False)

        # Plot this iteration: one subplot per zone
        _plot_refinement_iteration(
            all_refinement_rows[iteration],
            site_ids,
            save_dir,
            system,
            iteration,
            prev_best=prev_best,
            new_best=dict(current_best),
        )

    # Final result
    logger.info("")
    logger.info("── Refinement converged after %d iteration(s) ──", iteration)
    final_str = ", ".join(f"{sid}@{b}" for sid, b in current_best.items())

    # Run one final sim at full config resolution to get exact metrics
    final_row = _run_multi_dc_case(bus_map=current_best, **full_res_kwargs)
    total_sims += 1

    df_final = pd.DataFrame([final_row])
    # Also overwrite the final summary with the refined result
    df_final.to_csv(
        save_dir / f"sweep_dc_locations_final_results_{system}.csv",
        index=False,
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL RESULT (after refinement)")
    logger.info("=" * 80)
    logger.info(
        "  %s  ->  viol=%.1fs  integral=%.4f  vmin=%.4f  vmax=%.4f",
        final_str,
        final_row["violation_time_s"],
        final_row["integral_violation_pu_s"],
        final_row["worst_vmin"],
        final_row["worst_vmax"],
    )
    logger.info("")
    logger.info(
        "Total simulations: %d (Phase 1: %d, Phase 2: %d, Phase 3: %d)",
        total_sims,
        total_phase1,
        len(phase2_rows),
        total_sims - total_phase1 - len(phase2_rows),
    )
    logger.info("All outputs in: %s", save_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Per-system experiment definitions
# ══════════════════════════════════════════════════════════════════════════════


def _experiment_ieee13(sys_const, training_trace):
    """IEEE 13-bus: single DC at bus 671 with training overlay."""
    models = (
        deploy("Llama-3.1-8B", 720),
        deploy("Llama-3.1-70B", 180),
        deploy("Llama-3.1-405B", 90),
        deploy("Qwen3-30B-A3B", 480),
        deploy("Qwen3-235B-A22B", 210),
    )

    # Per-model ramps: target = round(fraction * initial) with fraction=0.2.
    # Inject each ramp into its model's ReplicaSchedule via with_ramp() —
    # replaces the old single OR-chained InferenceRampSchedule.
    ramp_targets = {
        "Llama-3.1-8B": 144,
        "Llama-3.1-70B": 36,
        "Llama-3.1-405B": 18,
        "Qwen3-30B-A3B": 96,
        "Qwen3-235B-A22B": 42,
    }
    models = tuple(
        with_ramp((md, sched), ramp_targets[md.spec.model_label], t_start=2500, t_end=3000) for md, sched in models
    )

    dc_sites = {
        "default": DCSite(
            bus="671",
            bus_kv=sys_const["bus_kv"],
            base_kw_per_phase=500.0,
            models=models,
            seed=0,
            total_gpu_capacity=7200,
        ),
    }

    training = (
        TrainingRun(
            n_gpus=2400,
            trace=training_trace,
            target_peak_W_per_gpu=400.0,
        ).at(t_start=1000.0, t_end=2000.0)
        if training_trace is not None
        else None
    )

    ofo_config = OFOConfig(
        primal_step_size=0.1,
        w_throughput=0.001,
        w_switch=1.0,
        voltage_gradient_scale=1e6,
        v_min=V_MIN,
        v_max=V_MAX,
        voltage_dual_step_size=1.0,
        latency_dual_step_size=1.0,
        sensitivity_update_interval=3600,
        sensitivity_perturbation_kw=100.0,
    )

    tap_schedule_entries = [
        (1500, TapPosition(regulators={"creg1a": tap(16), "creg1b": tap(6), "creg1c": tap(17)})),
        (3300, TapPosition(regulators={"creg1a": tap(10), "creg1b": tap(6), "creg1c": tap(10)})),
    ]

    pv_systems = [PVSystemSpec(bus="675", bus_kv=4.16, peak_kw=10.0)]
    time_varying_loads = [TimeVaryingLoadSpec(bus="680", bus_kv=4.16, peak_kw=10.0)]

    return dict(
        dc_sites=dc_sites,
        training=training,
        ofo_config=ofo_config,
        tap_schedule_entries=tap_schedule_entries,
        training_t_start=1000.0,
        training_t_end=2000.0,
        ramp_t_end=3000.0,
        pv_systems=pv_systems,
        time_varying_loads=time_varying_loads,
    )


def _experiment_ieee34(sys_const, training_trace):
    """IEEE 34-bus: two DC sites (upstream + downstream)."""
    upstream_models = (
        deploy("Llama-3.1-8B", 720),
        deploy("Llama-3.1-70B", 180),
        deploy("Llama-3.1-405B", 90),
    )
    downstream_models = (
        deploy("Qwen3-30B-A3B", 480),
        deploy("Qwen3-235B-A22B", 210),
    )

    dc_sites = {
        "upstream": DCSite(
            bus="850",
            bus_kv=sys_const["bus_kv"],
            base_kw_per_phase=120.0,
            models=upstream_models,
            seed=0,
            total_gpu_capacity=520,
        ),
        "downstream": DCSite(
            bus="834",
            bus_kv=sys_const["bus_kv"],
            base_kw_per_phase=80.0,
            models=downstream_models,
            seed=42,
            total_gpu_capacity=600,
        ),
    }

    ofo_config = OFOConfig(
        primal_step_size=0.05,
        w_throughput=0.001,
        w_switch=1.0,
        voltage_gradient_scale=1e6,
        v_min=V_MIN,
        v_max=V_MAX,
        voltage_dual_step_size=20.0,
        latency_dual_step_size=1.0,
        sensitivity_update_interval=3600,
        sensitivity_perturbation_kw=10.0,
    )

    pv_systems = [
        PVSystemSpec(bus="848", bus_kv=24.9, peak_kw=130.0),
        PVSystemSpec(bus="830", bus_kv=24.9, peak_kw=65.0),
    ]
    time_varying_loads = [
        TimeVaryingLoadSpec(bus="860", bus_kv=24.9, peak_kw=80.0),
        TimeVaryingLoadSpec(bus="844", bus_kv=24.9, peak_kw=120.0),
        TimeVaryingLoadSpec(bus="840", bus_kv=24.9, peak_kw=60.0),
        TimeVaryingLoadSpec(bus="858", bus_kv=24.9, peak_kw=50.0),
        TimeVaryingLoadSpec(bus="854", bus_kv=24.9, peak_kw=40.0),
    ]

    return dict(
        dc_sites=dc_sites,
        training=None,
        ofo_config=ofo_config,
        pv_systems=pv_systems,
        time_varying_loads=time_varying_loads,
    )


def _experiment_ieee123(sys_const, training_trace):
    """IEEE 123-bus: four DC zones with per-site ramps."""
    dc_sites = {
        "z1_sw": DCSite(
            bus="8",
            bus_kv=sys_const["bus_kv"],
            base_kw_per_phase=310.0,
            # 8B: round(1.5 * 120) = 180
            models=(with_ramp(deploy("Llama-3.1-8B", 120), 180, t_start=500, t_end=1000),),
            seed=0,
            total_gpu_capacity=120,
        ),
        "z2_nw": DCSite(
            bus="23",
            bus_kv=sys_const["bus_kv"],
            base_kw_per_phase=265.0,
            # 30B: round(1.3 * 80) = 104
            models=(with_ramp(deploy("Qwen3-30B-A3B", 80), 104, t_start=1500, t_end=2500),),
            seed=17,
            total_gpu_capacity=160,
        ),
        "z3_se": DCSite(
            bus="60",
            bus_kv=sys_const["bus_kv"],
            base_kw_per_phase=295.0,
            # 70B: round(1.5 * 30) = 45, 405B: round(1.5 * 35) = 52
            models=(
                with_ramp(deploy("Llama-3.1-70B", 30), 45, t_start=700, t_end=1100),
                with_ramp(deploy("Llama-3.1-405B", 35), 52, t_start=700, t_end=1100),
            ),
            seed=34,
            total_gpu_capacity=400,
        ),
        "z4_ne": DCSite(
            bus="105",
            bus_kv=sys_const["bus_kv"],
            base_kw_per_phase=325.0,
            # 235B: round(0.5 * 55) = 27
            models=(with_ramp(deploy("Qwen3-235B-A22B", 55), 27, t_start=2000, t_end=2500),),
            seed=51,
            total_gpu_capacity=440,
        ),
    }

    ofo_config = OFOConfig(
        primal_step_size=0.05,
        w_throughput=0.001,
        w_switch=1.0,
        voltage_gradient_scale=1e6,
        v_min=V_MIN,
        v_max=V_MAX,
        voltage_dual_step_size=0.3,
        latency_dual_step_size=1.0,
        sensitivity_update_interval=3600,
        sensitivity_perturbation_kw=10.0,
    )

    pv_systems = [
        PVSystemSpec(bus="1", bus_kv=4.16, peak_kw=333.3),
        PVSystemSpec(bus="48", bus_kv=4.16, peak_kw=333.3),
        PVSystemSpec(bus="99", bus_kv=4.16, peak_kw=333.3),
    ]

    return dict(
        dc_sites=dc_sites,
        training=None,
        ofo_config=ofo_config,
        pv_systems=pv_systems,
        time_varying_loads=[],
        zones=sys_const.get("zones"),
    )


_EXPERIMENTS = {
    "ieee13": _experiment_ieee13,
    "ieee34": _experiment_ieee34,
    "ieee123": _experiment_ieee123,
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def main(
    *,
    system: str,
    buses: list[str] | None = None,
    dt_override: str | None = None,
    dt_screening: str | None = None,
    output_dir: Path | None = None,
    top_k: int = 4,
    refine: bool = False,
) -> None:
    sys_const = SYSTEMS[system]()

    # Pre-load data using all model specs (per-spec content-addressed cache).
    all_models = all_model_specs()

    logger.info("Loading data for %s...", system)
    inference_data = InferenceData.ensure(
        SPECS_CACHE_DIR,
        all_models,
        plot=False,
        dt_s=float(DT_DC),
    )
    training_trace = TrainingTrace.ensure(TRAINING_TRACE_PATH)
    logistic_models = LogisticModelStore.ensure(
        SPECS_CACHE_DIR,
        all_models,
        plot=False,
    )

    # Build experiment config
    experiment_fn = _EXPERIMENTS[system]
    experiment = experiment_fn(sys_const, training_trace)

    dc_sites = experiment["dc_sites"]
    n_sites = len(dc_sites)
    has_zones = experiment.get("zones") is not None and len(experiment.get("zones", {})) > 0

    # Collect all model specs from DC sites for the experiment
    exp_all_models: list[InferenceModelSpec] = []
    seen_labels: set[str] = set()
    for site in dc_sites.values():
        for md, _ in site.models:
            if md.spec.model_label not in seen_labels:
                exp_all_models.append(md.spec)
                seen_labels.add(md.spec.model_label)
    exp_all_models_tuple = tuple(exp_all_models)

    if has_zones:
        logger.info(
            "Config has %d DC site(s) with %d zone(s) -> zone-constrained sweep",
            n_sites,
            len(experiment["zones"]),
        )
        main_zoned(
            sys=sys_const,
            system=system,
            dc_sites=dc_sites,
            all_models=exp_all_models_tuple,
            inference_data=inference_data,
            training_trace=training_trace,
            training=experiment.get("training"),
            logistic_models=logistic_models,
            ofo_config=experiment["ofo_config"],
            zones=experiment["zones"],
            pv_systems=experiment.get("pv_systems"),
            time_varying_loads=experiment.get("time_varying_loads"),
            dt_override=dt_override,
            dt_screening=dt_screening,
            output_dir=output_dir,
            top_k=top_k,
            refine=refine,
        )
    elif n_sites <= 1:
        logger.info("Config has %d DC site(s) -> 1-D sweep", n_sites)
        dc_site = next(iter(dc_sites.values()))
        main_1d(
            sys=sys_const,
            system=system,
            dc_site=dc_site,
            all_models=exp_all_models_tuple,
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            ofo_config=experiment["ofo_config"],
            training=experiment.get("training"),
            replica_schedules={md.spec.model_label: sched for md, sched in dc_site.models},
            tap_schedule_entries=experiment.get("tap_schedule_entries"),
            training_t_start=experiment.get("training_t_start", 0),
            training_t_end=experiment.get("training_t_end", 0),
            ramp_t_end=experiment.get("ramp_t_end", 0),
            pv_systems=experiment.get("pv_systems"),
            time_varying_loads=experiment.get("time_varying_loads"),
            buses=buses,
            dt_override=dt_override,
            output_dir=output_dir,
        )
    else:
        logger.info("Config has %d DC site(s) -> 2-D sweep", n_sites)
        main_2d(
            sys=sys_const,
            system=system,
            dc_sites=dc_sites,
            all_models=exp_all_models_tuple,
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            ofo_config=experiment["ofo_config"],
            training=experiment.get("training"),
            pv_systems=experiment.get("pv_systems"),
            time_varying_loads=experiment.get("time_varying_loads"),
            dt_override=dt_override,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        system: str = "ieee13"
        """System name (ieee13, ieee34, ieee123)."""
        buses: str | None = None
        """Comma-separated list of buses to test (overrides auto-discovery, 1-D only)."""
        dt: str | None = None
        """Override time step for all components (e.g. '60' for 60s resolution)."""
        dt_screening: str | None = None
        """Coarser time step for Phase 1 screening only (e.g. '60'). Phases 2/3 use --dt or config."""
        top_k: int = 4
        """Number of top candidates per zone to keep after screening (zone-constrained mode)."""
        refine: bool = False
        """Enable Phase 3 iterative refinement (zone-constrained mode)."""
        output_dir: str | None = None
        """Override output directory path."""
        log_level: str = "INFO"
        """Logging verbosity (DEBUG, INFO, WARNING)."""

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    bus_list = [b.strip() for b in args.buses.split(",")] if args.buses else None
    out_dir = Path(args.output_dir) if args.output_dir else None
    main(
        system=args.system,
        buses=bus_list,
        dt_override=args.dt,
        dt_screening=args.dt_screening,
        output_dir=out_dir,
        top_k=args.top_k,
        refine=args.refine,
    )
