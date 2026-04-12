"""PV Expansion Planning (and optional joint DC siting) via Sensitivity-Based MILP.

Supports two modes via `--mode`:

- `pv-only` (default): PV placement, capacity, and regulator tap co-optimisation
  with fixed datacenter locations.
- `pv-and-dc`: Joint PV + DC location co-optimisation where DC bus assignments
  are also decision variables.

Uses OpenDSS-derived voltage sensitivity matrices and representative
operating scenarios.  A Gurobi MILP minimises annualised investment cost
minus operational savings (PV electricity bill reduction) plus a
voltage-violation penalty, subject to linearised voltage constraints.

PV-only formulation
-------------------
Sets:
    B   candidate buses for PV placement
    R   voltage regulators (each phase independently)
    I   bus-phase pairs for voltage monitoring (excludes source/regulator)
    T   time steps within each scenario
    S   scenarios (solar + load profiles, with weights)

Decision variables:
    x[b]  in {0,1}       PV placement at bus b
    s[b]  >= 0           installed PV capacity (kW 3-phase) at bus b
    c[b,t,s] >= 0        curtailed PV power at bus b, time t, scenario s (kW)
    dt[r,s,h] in Z       tap change for regulator r, scenario s, hour h
    slack_o[i,t,s] >= 0  overvoltage slack
    slack_u[i,t,s] >= 0  undervoltage slack

PV+DC formulation (extends PV-only)
------------------------------------
Additional sets:
    B_dc  candidate buses for DC placement
    K     DC sites (to be assigned to buses)

Additional decision variables:
    y[k,b] in {0,1}  assign DC site k to bus b (within its zone)

Additional constraints:
    sum_b y[k,b] == 1  for each DC site k (assignment)
    sum_k y[k,b] <= 1  for each bus b   (mutual exclusion)
    v includes DC load effect via y-weighted sensitivities

Usage:
    python optimize_pv_locations_and_capacities.py --system ieee34
    python optimize_pv_locations_and_capacities.py --system ieee34 --n-pv 5 --s-max-kw 1000 --s-total-max-kw 2500
    python optimize_pv_locations_and_capacities.py --system ieee123
    python optimize_pv_locations_and_capacities.py --system ieee34 --mode pv-and-dc
    python optimize_pv_locations_and_capacities.py --system ieee123 --mode pv-and-dc --n-pv 5
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from opendssdirect import dss
from utils import discover_candidate_buses

from openg2g.controller.ofo import LogisticModelStore, OFOBatchSizeController, OFOConfig
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.config import (
    DatacenterConfig,
    InferenceModelSpec,
    ModelDeployment,
    PowerAugmentationConfig,
    ReplicaSchedule,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData, MLEnergySource
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.config import TapPosition, TapSchedule
from openg2g.grid.generator import SyntheticPV
from openg2g.grid.load import SyntheticLoad
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.performance import PerformanceStats, compute_performance_stats
from openg2g.metrics.voltage import VoltageStats, compute_allbus_voltage_stats

from systems import SYSTEMS, tap

matplotlib.use("Agg")

logger = logging.getLogger("optimize_pv_locations_and_capacities")


# Scenario generation


@dataclass
class Scenario:
    """A representative operating scenario (e.g. one day)."""

    name: str
    hours: np.ndarray  # shape (T,)  -- hour of day
    pv_profile: np.ndarray  # shape (T,)  -- normalised PV output [0,1]
    price_per_kwh: np.ndarray  # shape (T,)  -- electricity price ($/kWh)
    weight: float = 1.0  # scenario weight in objective
    hours_per_step: float = 1.0  # duration of each time step (hours)
    # Per-bus load profiles: bus -> kW per phase array (T,)
    load_kw_per_bus: dict[str, np.ndarray] = field(default_factory=dict)


def _solar_profile(hour: float, peak_hour: float = 12.0, daylight: float = 14.0) -> float:
    """Normalised solar output at a given hour (bell-curve)."""
    sunrise = peak_hour - daylight / 2
    sunset = peak_hour + daylight / 2
    if hour < sunrise or hour > sunset:
        return 0.0
    return math.sin(math.pi * (hour - sunrise) / daylight) ** 2


def _tou_price(hour: float) -> float:
    """Time-of-use electricity price ($/kWh).

    Off-peak  (0:00-8:00, 21:00-24:00):  $0.05/kWh
    Mid-peak  (8:00-12:00, 18:00-21:00): $0.10/kWh
    On-peak   (12:00-18:00):             $0.20/kWh
    """
    if hour < 8 or hour >= 21:
        return 0.05
    if hour < 12 or hour >= 18:
        return 0.10
    return 0.20


# Load profile archetypes


def _industrial_load(hour: float) -> float:
    """Flat during work hours, low nights."""
    if 6 <= hour <= 18:
        return 0.8 + 0.2 * math.sin(math.pi * (hour - 6) / 12)
    return 0.15


def _commercial_load(hour: float) -> float:
    """Ramp up morning, plateau midday, ramp down evening."""
    if hour < 7:
        return 0.1
    if hour < 9:
        return 0.1 + 0.9 * (hour - 7) / 2
    if hour < 17:
        return 1.0
    if hour < 19:
        return 1.0 - 0.9 * (hour - 17) / 2
    return 0.1


def _residential_load(hour: float) -> float:
    """Dual peak morning and evening."""
    morning = 0.6 * math.exp(-((hour - 8) ** 2) / 2)
    evening = 1.0 * math.exp(-((hour - 19) ** 2) / 3)
    return min(1.0, 0.2 + morning + evening)


def _ev_charging_load(hour: float) -> float:
    """Evening ramp, overnight trickle."""
    if hour < 16:
        return 0.05
    if hour < 22:
        return 0.05 + 0.95 * ((hour - 16) / 6) ** 2
    return 0.3


def _constant_load(hour: float) -> float:
    """Always-on (street lighting, base infrastructure)."""
    return 0.8


# Assign archetypes to load bus indices
_LOAD_ARCHETYPES = [
    _industrial_load,  # load bus 0 (e.g. 860)
    _commercial_load,  # load bus 1 (e.g. 844)
    _residential_load,  # load bus 2 (e.g. 840)
    _residential_load,  # load bus 3 (e.g. 858)
    _constant_load,  # load bus 4 (e.g. 854)
    _ev_charging_load,  # load bus 5 (e.g. 848)
]


def _dc_demand_upstream(hour: float) -> float:
    """Normalised demand for the upstream DC site [0,1].

    Pattern: LLM serving traffic -- clear diurnal pattern with pronounced
    peak/off-peak.  Off-peak still has ~40% load from international users
    and automated agents.  Peak midday from user-facing chat traffic.
    """
    base = 0.40  # always-on: cooling, networking, overnight intl traffic
    # Broad daytime demand from user-facing chat
    daytime = 0.50 * math.exp(-((hour - 13.0) ** 2) / 8.0)
    # Morning ramp as US users come online
    morning = 0.15 * math.exp(-((hour - 9.0) ** 2) / 1.5)
    # Evening batch jobs (fine-tuning, evals)
    evening_batch = 0.12 * math.exp(-((hour - 21.5) ** 2) / 1.0)
    # Bursty ripple during business hours
    ripple = 0.04 * math.sin(2 * math.pi * hour / 1.5) if 7 <= hour <= 22 else 0
    return max(0.0, min(1.0, base + daytime + morning + evening_batch + ripple))


def _dc_demand_downstream(hour: float) -> float:
    """Normalised demand for the downstream DC site [0,1].

    Pattern: reasoning-heavy workloads (Qwen thinking models) -- high and
    relatively flat since long-running reasoning jobs run continuously.
    Irregular fluctuations from heterogeneous batch job arrivals (not periodic).
    """
    base = 0.65  # high base: always-on reasoning pipelines, agents, batch jobs
    # Mild daytime uplift from interactive reasoning queries
    daytime = 0.12 * math.exp(-((hour - 13.5) ** 2) / 12.0)
    # Irregular fluctuations: superposition of incommensurate frequencies
    # so the pattern never repeats cleanly within 24h
    f1 = 0.05 * math.sin(2 * math.pi * hour / 2.3)  # ~2.3h cycle
    f2 = 0.04 * math.sin(2 * math.pi * hour / 3.7 + 1.0)  # ~3.7h cycle
    f3 = 0.03 * math.sin(2 * math.pi * hour / 1.1 + 2.5)  # ~1.1h fast jitter
    fluctuation = (f1 + f2 + f3) if 5 <= hour <= 23.5 else 0
    # Small end-of-day bump (report generation, summarisation)
    eod = 0.08 * math.exp(-((hour - 17.5) ** 2) / 0.5)
    # Brief overnight maintenance dip
    maintenance = -0.06 * math.exp(-((hour - 3.5) ** 2) / 0.5)
    return max(0.0, min(1.0, base + daytime + fluctuation + eod + maintenance))


def _dc_demand_batch_heavy(hour: float) -> float:
    """Normalised demand for a batch-heavy DC site [0,1].

    Pattern: large batch training/eval jobs with overnight pre-emption.
    High sustained load during business hours, drops at night for
    maintenance windows, with sharp ramp-up in the morning.
    """
    base = 0.35
    # Morning ramp-up as batch jobs resume after maintenance
    morning_ramp = 0.55 * (1 / (1 + math.exp(-(hour - 7.0) * 2.0)))
    # Evening wind-down
    evening_down = -0.50 * (1 / (1 + math.exp(-(hour - 20.0) * 2.5)))
    # Mid-afternoon peak from parallel job submissions
    afternoon = 0.10 * math.exp(-((hour - 14.5) ** 2) / 2.0)
    return max(0.0, min(1.0, base + morning_ramp + evening_down + afternoon))


def _dc_demand_mixed(hour: float) -> float:
    """Normalised demand for a mixed-workload DC site [0,1].

    Pattern: combination of user-facing and batch -- moderate diurnal
    variation with a broad plateau and gentle transitions.
    """
    base = 0.50
    # Broad daytime plateau
    daytime = 0.35 * math.exp(-((hour - 13.0) ** 2) / 18.0)
    # Small early-morning dip
    night_dip = -0.15 * math.exp(-((hour - 3.0) ** 2) / 2.0)
    # Slight evening secondary peak (async API traffic)
    evening = 0.08 * math.exp(-((hour - 20.0) ** 2) / 1.5)
    return max(0.0, min(1.0, base + daytime + night_dip + evening))


# Ordered list of DC demand archetypes for multi-site assignment
_DC_DEMAND_ARCHETYPES = [
    _dc_demand_upstream,
    _dc_demand_downstream,
    _dc_demand_batch_heavy,
    _dc_demand_mixed,
]


def _make_load_profiles(
    hours: np.ndarray,
    load_configs: list[tuple[str, float]],
    load_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Create per-bus load profiles (kW per phase) for one scenario.

    Args:
        load_configs: [(bus, peak_kw_per_phase), ...]
        load_scale: scenario-specific multiplier (e.g. 1.3 for heat wave)
    """
    profiles = {}
    for idx, (bus, peak_kw) in enumerate(load_configs):
        archetype = _LOAD_ARCHETYPES[idx % len(_LOAD_ARCHETYPES)]
        profile = np.array([archetype(h) for h in hours])
        profiles[bus] = peak_kw * profile * load_scale
    return profiles


def generate_scenarios(
    hours: np.ndarray | None = None,
    load_configs: list[tuple[str, float]] | None = None,
    dc_sites_info: list[tuple[str, float, float, str]] | None = None,
) -> list[Scenario]:
    """Create representative scenarios for a 24-hour day.

    Returns 5 scenarios covering clear/cloudy/winter/spring/extreme conditions.
    Each includes TOU prices and per-bus load profiles with distinct archetypes.

    Scenario weights sum to 1.0 and represent the fraction of the year each
    scenario represents.  Annual cost = 365 * sum(weight_s * daily_cost_s).

    Args:
        load_configs: [(bus, peak_kw_per_phase), ...] for time-varying loads.
        dc_sites_info: [(bus, peak_kw_per_phase, base_kw_per_phase, site_id), ...]
            for DC sites.  peak_kw is the load at full utilization; base_kw is
            the sensitivity linearisation point (mean demand).  The deviation
            entered into the voltage model is: peak_kw * profile(t) - base_kw.
    """
    if hours is None:
        hours = np.arange(0, 24, 15 / 60)  # 15-minute resolution (96 steps/day)

    hours_per_step = float(hours[1] - hours[0]) if len(hours) > 1 else 1.0
    price = np.array([_tou_price(h) for h in hours])
    lc = load_configs or []

    # Per-site DC demand profiles (normalised 0-1)
    # Cycle through archetypes for arbitrary number of DC sites
    dc_profiles: dict[str, np.ndarray] = {}
    if dc_sites_info:
        for idx, (_bus, _pk, _bs, site_id) in enumerate(dc_sites_info):
            archetype = _DC_DEMAND_ARCHETYPES[idx % len(_DC_DEMAND_ARCHETYPES)]
            dc_profiles[site_id] = np.array([archetype(h) for h in hours])

    def _sc(name, pv, weight, load_mult=1.0):
        loads = _make_load_profiles(hours, lc, load_scale=load_mult) if lc else {}
        # DC load deviation from sensitivity base case:
        # actual load = peak_kw * profile(t), sensitivity base = base_kw (constant)
        # deviation = peak_kw * profile(t) - base_kw
        if dc_sites_info:
            for dc_bus, dc_peak, dc_base, site_id in dc_sites_info:
                loads[dc_bus] = dc_peak * dc_profiles[site_id] - dc_base
        return Scenario(name, hours, pv, price, weight=weight, hours_per_step=hours_per_step, load_kw_per_bus=loads)

    scenarios = []

    # Weights sum to 1.0: fraction of year each scenario represents
    # summer       29% (~106 days) -- merged clear + heat, weighted avg load
    # summer_cloudy 8% (~29 days) -- extreme cloud event with sharp PV drops
    # spring       21% (~77 days) -- moderate PV, moderate load
    # autumn_cloudy 12% (~44 days) -- variable clouds with oscillating PV
    # winter       30% (~110 days) -- merged winter + winter peak, high load

    # 1. Summer (merged clear + heat) -- high PV, slightly elevated load
    #    Original: clear w=0.25 load=1.0, heat w=0.04 load=1.3
    #    Weighted avg load mult: (0.25*1.0 + 0.04*1.3) / 0.29 ≈ 1.04
    pv_summer = np.array([_solar_profile(h, peak_hour=13.0, daylight=14.0) for h in hours])
    scenarios.append(_sc("summer", pv_summer, weight=0.29, load_mult=1.20))

    # 2. Summer cloudy -- heavy overcast with intermittent breaks
    #    Base output ~30% of clear sky, with two brief sunny breaks at ~10h
    #    and ~14h where output spikes to ~70%.  Very different shape from
    #    clear summer: low sustained output with sharp transients.
    pv_summer_cloudy = np.array(
        [
            _solar_profile(h, peak_hour=13.0, daylight=14.0)
            * (
                0.3  # heavy overcast base
                + 0.4 * math.exp(-((h - 10.0) ** 2) / 0.5)  # brief break ~10h
                + 0.4 * math.exp(-((h - 14.5) ** 2) / 0.3)  # brief break ~14.5h
            )
            for h in hours
        ]
    )
    pv_summer_cloudy = np.minimum(pv_summer_cloudy, pv_summer)  # cap at clear sky
    scenarios.append(_sc("summer_cloudy", pv_summer_cloudy, weight=0.08, load_mult=1.05))

    # 3. Spring shoulder -- moderate PV, moderate load
    pv_spring = np.array([_solar_profile(h, peak_hour=12.5, daylight=12.0) * 0.85 for h in hours])
    scenarios.append(_sc("spring", pv_spring, weight=0.21))

    # 4. Autumn variable clouds -- PV with periodic cloud modulation
    cloud_mod = 0.5 + 0.4 * np.array([math.sin(math.pi * h / 3) ** 2 for h in hours])
    pv_autumn = pv_spring * cloud_mod
    scenarios.append(_sc("autumn_cloudy", pv_autumn, weight=0.12, load_mult=0.90))

    # 5. Winter (merged winter + winter peak) -- short day, high load
    #    Original: winter w=0.22 load=1.1 pv=1.0×, peak w=0.08 load=1.2 pv=0.7×
    #    Weighted avg load mult: (0.22*1.1 + 0.08*1.2) / 0.30 ≈ 1.13
    #    Weighted avg PV scale: (0.22*1.0 + 0.08*0.7) / 0.30 ≈ 0.92
    pv_winter = np.array([_solar_profile(h, peak_hour=12.0, daylight=9.0) for h in hours])
    scenarios.append(_sc("winter", pv_winter * 0.92, weight=0.30, load_mult=1.15))

    return scenarios


# Sensitivity computation


def _add_dc_loads(
    dc_sites: dict[str, dict] | None,
    bus_kv: float,
) -> None:
    """Add DC datacenter loads to the active OpenDSS circuit.

    Each site gets three single-phase loads (one per phase) at the specified bus.
    This ensures the sensitivity base case includes the datacenter operating point.
    """
    if not dc_sites:
        return

    ln_kv = bus_kv / math.sqrt(3.0)
    for site_id, site_cfg in dc_sites.items():
        bus = site_cfg["bus"]
        kw_per_phase = site_cfg.get("base_kw_per_phase", 0.0)
        if kw_per_phase <= 0:
            continue
        for ph, suffix in [(1, "a"), (2, "b"), (3, "c")]:
            dss.Text.Command(
                f"New Load.DC_{site_id}_{suffix} bus1={bus}.{ph} phases=1 "
                f"conn=wye kV={ln_kv:.6f} kW={kw_per_phase:.2f} kvar=0 model=1 vminpu=0.85"
            )
    logger.info(
        "Added DC loads: %s",
        ", ".join(f"{sid}@{s['bus']}={s.get('base_kw_per_phase', 0):.0f}kW/ph" for sid, s in dc_sites.items()),
    )


def compute_pv_sensitivities(
    case_dir: Path,
    master_file: str,
    candidate_buses: list[str],
    exclude_buses: set[str],
    source_pu: float = 1.05,
    initial_taps: dict[str, float] | None = None,
    perturbation_kw: float = 100.0,
    bus_kv: float = 24.9,
    dc_sites: dict[str, dict] | None = None,
    existing_pv: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Compute voltage sensitivity to PV injection at each candidate bus.

    For each candidate bus b, injects +perturbation_kw per phase as negative
    load (PV generation) and measures the voltage change at all monitored
    bus-phase pairs.

    Args:
        dc_sites: Optional dict of DC site configs (bus, base_kw_per_phase)
            to include in the base case operating point.
        existing_pv: Optional dict {bus: kw_3phase} of already-installed PV
            to include in the base case.  Used during iterative re-linearisation
            so sensitivities are computed around the actual operating point.

    Returns:
        H_pv: shape (n_monitored, n_candidates) -- dv/dp_pv per phase (pu/kW)
        v0:   shape (n_monitored,) -- base voltage (no PV)
        v_index: list of (bus, phase) pairs corresponding to rows
    """
    # Compile circuit
    dss.Basic.ClearAll()
    dss.Text.Command(f"Redirect [{case_dir / master_file}]")

    if source_pu is not None:
        dss.Text.Command(f"Edit Vsource.source pu={source_pu}")

    # Set initial taps (disable controls)
    dss.Text.Command("Set controlmode=off")
    if initial_taps:
        for reg_name, tap_pu in initial_taps.items():
            dss.RegControls.Name(reg_name)
            xfmr = dss.RegControls.Transformer()
            dss.Transformers.Name(xfmr)
            dss.Transformers.Wdg(2)
            dss.Transformers.Tap(tap_pu)

    # Add DC loads to establish realistic operating point
    _add_dc_loads(dc_sites, bus_kv)

    # Add existing PV installations to the base case (for re-linearisation)
    if existing_pv:
        pv_ln_kv = bus_kv / math.sqrt(3.0)
        for bus, kw_3ph in existing_pv.items():
            kw_per_phase = kw_3ph / 3.0
            for ph, suffix in [(1, "a"), (2, "b"), (3, "c")]:
                name = f"ExPV_{bus}_{suffix}"
                dss.Text.Command(
                    f"New Load.{name} bus1={bus}.{ph} phases=1 "
                    f"conn=wye kV={pv_ln_kv:.6f} kW={-kw_per_phase:.4f} kvar=0 model=1 vminpu=0.85"
                )

    # Add PV load elements (initially 0) for each candidate bus
    pv_ln_kv = bus_kv / math.sqrt(3.0)
    pv_load_names = {}  # bus -> [load_a, load_b, load_c]
    for bus in candidate_buses:
        names = []
        for ph, suffix in [(1, "a"), (2, "b"), (3, "c")]:
            name = f"PV_{bus}_{suffix}"
            dss.Text.Command(
                f"New Load.{name} bus1={bus}.{ph} phases=1 conn=wye kV={pv_ln_kv:.6f} kW=0 kvar=0 model=1 vminpu=0.85"
            )
            names.append(name)
        pv_load_names[bus] = names

    # Solve base case
    dss.Solution.SolveNoControl()

    # Build v_index (monitored bus-phase pairs)
    exclude_lower = {b.lower() for b in exclude_buses}
    v_index: list[tuple[str, str]] = []
    all_bus_names = dss.Circuit.AllBusNames()
    for bus_name in all_bus_names:
        if bus_name.lower() in exclude_lower:
            continue
        dss.Circuit.SetActiveBus(bus_name)
        nodes = dss.Bus.Nodes()
        bus_kv_actual = dss.Bus.kVBase()
        # Only monitor 3-phase buses at target voltage (24.9 kV -> kVBase ~14.376)
        target_kv_base = bus_kv / math.sqrt(3.0)
        if abs(bus_kv_actual - target_kv_base) > 0.5:
            continue
        for node in nodes:
            if node in (1, 2, 3):
                phase = {1: "a", 2: "b", 3: "c"}[node]
                v_index.append((bus_name, phase))

    # Get base voltages, filter out bus-phases with 0 voltage (non-existent)
    v0_raw = np.zeros(len(v_index))
    for idx, (bus, phase) in enumerate(v_index):
        dss.Circuit.SetActiveBus(bus)
        v_mag = dss.Bus.puVmagAngle()
        node = {"a": 0, "b": 1, "c": 2}[phase]
        if node < len(v_mag) // 2:
            v0_raw[idx] = v_mag[2 * node]

    # Remove entries with zero or near-zero voltage (phase doesn't exist)
    valid = v0_raw > 0.5
    v_index = [v_index[i] for i in range(len(v_index)) if valid[i]]
    v0 = v0_raw[valid]

    # Compute sensitivity for each candidate bus (using filtered v_index)
    n_mon = len(v_index)
    n_cand = len(candidate_buses)
    H_pv = np.zeros((n_mon, n_cand))

    for j, bus in enumerate(candidate_buses):
        # Inject PV (negative load = generation) on all 3 phases
        for name in pv_load_names[bus]:
            dss.Loads.Name(name)
            dss.Loads.kW(-perturbation_kw)  # negative = generation

        dss.Solution.SolveNoControl()

        # Measure voltage change at monitored bus-phases
        for idx, (mon_bus, phase) in enumerate(v_index):
            dss.Circuit.SetActiveBus(mon_bus)
            v_mag = dss.Bus.puVmagAngle()
            node = {"a": 0, "b": 1, "c": 2}[phase]
            if node < len(v_mag) // 2:
                v_new = v_mag[2 * node]
                H_pv[idx, j] = (v_new - v0[idx]) / perturbation_kw

        # Reset PV load to 0
        for name in pv_load_names[bus]:
            dss.Loads.Name(name)
            dss.Loads.kW(0.0)

        dss.Solution.SolveNoControl()  # restore base state

    logger.info("Computed PV sensitivities: %d monitored bus-phases x %d candidates", n_mon, n_cand)
    return H_pv, v0, v_index


# Tap sensitivity computation

TAP_STEP = 0.00625  # per-unit change per integer tap step


def compute_tap_sensitivities(
    case_dir: Path,
    master_file: str,
    regulator_names: list[str],
    v_index: list[tuple[str, str]],
    *,
    source_pu: float = 1.05,
    initial_taps: dict[str, float] | None = None,
    bus_kv: float = 24.9,
    dc_sites: dict[str, dict] | None = None,
    existing_pv: dict[str, float] | None = None,
) -> np.ndarray:
    """Compute voltage sensitivity to regulator tap changes.

    For each regulator, perturbs the tap by +1 integer step and measures
    the voltage change at all monitored bus-phase pairs in *v_index*
    (which must match the ordering from `compute_pv_sensitivities`).

    Returns:
        H_tap: shape (n_monitored, n_regulators) -- dv per +1 tap step (pu)
    """
    # Compile circuit (fresh instance -- PV loads from previous call are gone)
    dss.Basic.ClearAll()
    dss.Text.Command(f"Redirect [{case_dir / master_file}]")

    if source_pu is not None:
        dss.Text.Command(f"Edit Vsource.source pu={source_pu}")

    dss.Text.Command("Set controlmode=off")
    if initial_taps:
        for reg_name, tap_pu in initial_taps.items():
            dss.RegControls.Name(reg_name)
            xfmr = dss.RegControls.Transformer()
            dss.Transformers.Name(xfmr)
            dss.Transformers.Wdg(2)
            dss.Transformers.Tap(tap_pu)

    # Add DC loads to establish realistic operating point
    _add_dc_loads(dc_sites, bus_kv)

    # Add existing PV installations to the base case (for re-linearisation)
    if existing_pv:
        pv_ln_kv = bus_kv / math.sqrt(3.0)
        for bus, kw_3ph in existing_pv.items():
            kw_per_phase = kw_3ph / 3.0
            for ph, suffix in [(1, "a"), (2, "b"), (3, "c")]:
                name = f"ExPV_{bus}_{suffix}"
                dss.Text.Command(
                    f"New Load.{name} bus1={bus}.{ph} phases=1 "
                    f"conn=wye kV={pv_ln_kv:.6f} kW={-kw_per_phase:.4f} kvar=0 model=1 vminpu=0.85"
                )

    # Solve base case
    dss.Solution.SolveNoControl()

    # Get base voltages at v_index entries
    n_mon = len(v_index)
    v0 = np.zeros(n_mon)
    for idx, (bus, phase) in enumerate(v_index):
        dss.Circuit.SetActiveBus(bus)
        v_mag = dss.Bus.puVmagAngle()
        node = {"a": 0, "b": 1, "c": 2}[phase]
        if node < len(v_mag) // 2:
            v0[idx] = v_mag[2 * node]

    # Compute sensitivity for each regulator
    n_regs = len(regulator_names)
    H_tap = np.zeros((n_mon, n_regs))

    for r, reg_name in enumerate(regulator_names):
        # Get current tap
        dss.RegControls.Name(reg_name)
        xfmr = dss.RegControls.Transformer()
        dss.Transformers.Name(xfmr)
        dss.Transformers.Wdg(2)
        current_tap = dss.Transformers.Tap()

        # Perturb: +1 integer step
        dss.Transformers.Tap(current_tap + TAP_STEP)
        dss.Solution.SolveNoControl()

        # Measure voltage change
        for idx, (bus, phase) in enumerate(v_index):
            dss.Circuit.SetActiveBus(bus)
            v_mag = dss.Bus.puVmagAngle()
            node = {"a": 0, "b": 1, "c": 2}[phase]
            if node < len(v_mag) // 2:
                v_new = v_mag[2 * node]
                H_tap[idx, r] = v_new - v0[idx]  # sensitivity per +1 tap step

        # Reset tap
        dss.Transformers.Tap(current_tap)
        dss.Solution.SolveNoControl()

    logger.info("Computed tap sensitivities: %d monitored x %d regulators", n_mon, n_regs)
    for r, reg_name in enumerate(regulator_names):
        logger.info("  %s: mean |dv/dtap| = %.6f pu/step", reg_name, np.mean(np.abs(H_tap[:, r])))
    return H_tap


# Load sensitivity computation


def compute_load_sensitivities(
    case_dir: Path,
    master_file: str,
    load_buses: list[str],
    v_index: list[tuple[str, str]],
    *,
    source_pu: float = 1.05,
    initial_taps: dict[str, float] | None = None,
    bus_kv: float = 24.9,
    dc_sites: dict[str, dict] | None = None,
    perturbation_kw: float = 100.0,
    existing_pv: dict[str, float] | None = None,
) -> np.ndarray:
    """Compute voltage sensitivity to load injection at specified buses.

    Returns:
        H_load: shape (n_monitored, n_load_buses) -- dv per kW per phase (pu/kW)
    """
    dss.Basic.ClearAll()
    dss.Text.Command(f"Redirect [{case_dir / master_file}]")

    if source_pu is not None:
        dss.Text.Command(f"Edit Vsource.source pu={source_pu}")

    dss.Text.Command("Set controlmode=off")
    if initial_taps:
        for reg_name, tap_pu in initial_taps.items():
            dss.RegControls.Name(reg_name)
            xfmr = dss.RegControls.Transformer()
            dss.Transformers.Name(xfmr)
            dss.Transformers.Wdg(2)
            dss.Transformers.Tap(tap_pu)

    _add_dc_loads(dc_sites, bus_kv)

    # Add existing PV installations to the base case (for re-linearisation)
    if existing_pv:
        pv_ln_kv = bus_kv / math.sqrt(3.0)
        for bus, kw_3ph in existing_pv.items():
            kw_per_phase = kw_3ph / 3.0
            for ph, suffix in [(1, "a"), (2, "b"), (3, "c")]:
                name = f"ExPV_{bus}_{suffix}"
                dss.Text.Command(
                    f"New Load.{name} bus1={bus}.{ph} phases=1 "
                    f"conn=wye kV={pv_ln_kv:.6f} kW={-kw_per_phase:.4f} kvar=0 model=1 vminpu=0.85"
                )

    # Add perturbation load elements (initially 0)
    ln_kv = bus_kv / math.sqrt(3.0)
    load_names: dict[str, list[str]] = {}
    for bus in load_buses:
        names = []
        for ph, suffix in [(1, "a"), (2, "b"), (3, "c")]:
            name = f"LPERT_{bus}_{suffix}"
            dss.Text.Command(
                f"New Load.{name} bus1={bus}.{ph} phases=1 conn=wye kV={ln_kv:.6f} kW=0 kvar=0 model=1 vminpu=0.85"
            )
            names.append(name)
        load_names[bus] = names

    dss.Solution.SolveNoControl()

    # Get base voltages
    n_mon = len(v_index)
    v0 = np.zeros(n_mon)
    for idx, (bus, phase) in enumerate(v_index):
        dss.Circuit.SetActiveBus(bus)
        v_mag = dss.Bus.puVmagAngle()
        node = {"a": 0, "b": 1, "c": 2}[phase]
        if node < len(v_mag) // 2:
            v0[idx] = v_mag[2 * node]

    # Perturb each load bus
    n_loads = len(load_buses)
    H_load = np.zeros((n_mon, n_loads))

    for j, bus in enumerate(load_buses):
        for name in load_names[bus]:
            dss.Loads.Name(name)
            dss.Loads.kW(perturbation_kw)

        dss.Solution.SolveNoControl()

        for idx, (mon_bus, phase) in enumerate(v_index):
            dss.Circuit.SetActiveBus(mon_bus)
            v_mag = dss.Bus.puVmagAngle()
            node = {"a": 0, "b": 1, "c": 2}[phase]
            if node < len(v_mag) // 2:
                v_new = v_mag[2 * node]
                H_load[idx, j] = (v_new - v0[idx]) / perturbation_kw

        for name in load_names[bus]:
            dss.Loads.Name(name)
            dss.Loads.kW(0.0)
        dss.Solution.SolveNoControl()

    logger.info("Computed load sensitivities: %d monitored x %d load buses", n_mon, n_loads)
    return H_load


def precompute_load_v_shift(
    H_load: np.ndarray,
    load_buses: list[str],
    scenarios: list[Scenario],
) -> list[np.ndarray]:
    """Precompute voltage shift from time-varying loads for each scenario.

    Returns:
        List of arrays, one per scenario, each shape (T, n_monitored).
        v_shift[sc_idx][t, i] = voltage shift at bus-phase i, time step t.
    """
    shifts = []
    for sc in scenarios:
        T = len(sc.hours)
        n_mon = H_load.shape[0]
        shift = np.zeros((T, n_mon))
        for j, bus in enumerate(load_buses):
            if bus in sc.load_kw_per_bus:
                load_kw = sc.load_kw_per_bus[bus]  # shape (T,) -- kW per phase
                # H_load is per-phase sensitivity (pu/kW), load_kw is per-phase
                shift += np.outer(load_kw, H_load[:, j])
        shifts.append(shift)
    return shifts


# MILP formulation


@dataclass
class MILPResult:
    """Result of the PV expansion MILP."""

    status: str
    objective: float
    pv_locations: list[str]  # selected buses
    pv_capacities_kw: list[float]  # installed capacity (kW 3-phase)
    investment_cost: float
    operational_savings: float  # annual PV electricity savings ($)
    violation_penalty: float
    switching_cost: float  # annual tap switching cost ($)
    curtailment_mwh: float  # annual PV curtailment (MWh/year)
    solve_time_s: float
    all_x: dict[str, float]  # x[bus] values
    all_s: dict[str, float]  # s[bus] values (kW)
    # Time-varying tap schedules: tap_schedules[scenario_name][hour] = {reg: absolute_tap}
    tap_schedules: dict[str, dict[int, dict[str, int]]] | None = None


# DC co-optimisation data structures


@dataclass
class _DCSiteInfo:
    """Minimal DC site descriptor for the co-optimisation MILP."""

    bus: str
    base_kw_per_phase: float
    models: list[str] | None = None
    total_gpu_capacity: int | None = None


@dataclass
class CoOptMILPResult:
    """Result of the joint PV + DC co-optimisation MILP."""

    status: str
    objective: float
    pv_locations: list[str]
    pv_capacities_kw: list[float]
    dc_assignments: dict[str, str]  # site_id -> assigned bus
    investment_cost: float
    operational_savings: float
    violation_penalty: float
    switching_cost: float
    curtailment_mwh: float
    solve_time_s: float
    all_x: dict[str, float]
    all_s: dict[str, float]
    all_y: dict[tuple[str, str], float]  # (site_id, bus) -> y value
    tap_schedules: dict[str, dict[int, dict[str, int]]] | None = None


# DC demand profile computation


def compute_dc_demand_profiles(
    dc_site_configs: dict[str, object],
    model_gpu_map: dict[str, int],
    scenarios: list[Scenario],
    typical_gpu_w: float = 300.0,
) -> tuple[np.ndarray, list[str]]:
    """Compute time-varying DC demand for each site.

    Cycles through `_DC_DEMAND_ARCHETYPES` for each site.  For each site,
    computes the peak demand (base infrastructure + GPU inference) and scales
    the archetype profile accordingly.

    Returns:
        dc_demand_kw: shape (n_sites, T, n_scenarios), kW per phase
        site_ids: ordered list of site IDs
    """
    site_ids = list(dc_site_configs.keys())
    n_sites = len(site_ids)
    if n_sites == 0:
        raise ValueError("No DC sites provided")

    # Use the first scenario to determine T (all scenarios should have same T)
    T = len(scenarios[0].hours)
    n_sc = len(scenarios)
    hours = scenarios[0].hours

    dc_demand_kw = np.zeros((n_sites, T, n_sc))

    for k, site_id in enumerate(site_ids):
        site_cfg = dc_site_configs[site_id]

        # Compute total GPU count for this site
        if site_cfg.total_gpu_capacity is not None:
            total_gpus = site_cfg.total_gpu_capacity
        else:
            total_gpus = sum(model_gpu_map.get(m, 0) for m in (site_cfg.models or []))

        gpu_kw_per_phase = total_gpus * typical_gpu_w / 1000.0 / 3.0
        peak_kw_per_phase = site_cfg.base_kw_per_phase + gpu_kw_per_phase

        # Select archetype by cycling
        archetype = _DC_DEMAND_ARCHETYPES[k % len(_DC_DEMAND_ARCHETYPES)]

        # Compute normalised profile (same across scenarios, but demand
        # level can vary by scenario via load multiplier -- we keep it
        # simple and use the same profile for all scenarios)
        profile = np.array([archetype(h) for h in hours])

        for sc_idx in range(n_sc):
            dc_demand_kw[k, :, sc_idx] = peak_kw_per_phase * profile

        logger.info(
            "DC site %s: %d GPUs, peak %.0f kW/ph (base %.0f + GPU %.0f), archetype %d (%s)",
            site_id,
            total_gpus,
            peak_kw_per_phase,
            site_cfg.base_kw_per_phase,
            gpu_kw_per_phase,
            k % len(_DC_DEMAND_ARCHETYPES),
            archetype.__name__,
        )

    return dc_demand_kw, site_ids


# DC sensitivity computation


def compute_dc_sensitivities(
    case_dir: Path,
    master_file: str,
    dc_candidate_buses: list[str],
    v_index: list[tuple[str, str]],
    *,
    source_pu: float = 1.05,
    initial_taps: dict[str, float] | None = None,
    bus_kv: float = 24.9,
    perturbation_kw: float = 100.0,
) -> np.ndarray:
    """Compute voltage sensitivity to DC load at each candidate bus.

    This is a wrapper around `compute_load_sensitivities` -- DC loads
    are positive loads (consuming power), so the sensitivity is dv/dP_load.

    Returns:
        H_dc: shape (n_monitored, n_dc_candidates) -- dv per kW per phase (pu/kW)
    """
    return compute_load_sensitivities(
        case_dir=case_dir,
        master_file=master_file,
        load_buses=dc_candidate_buses,
        v_index=v_index,
        source_pu=source_pu,
        initial_taps=initial_taps,
        bus_kv=bus_kv,
        perturbation_kw=perturbation_kw,
    )


def solve_pv_milp(
    candidate_buses: list[str],
    H_pv: np.ndarray,
    v0: np.ndarray,
    v_index: list[tuple[str, str]],
    scenarios: list[Scenario],
    *,
    n_pv: int = 5,
    s_max_kw: float = 2000.0,
    s_total_max_kw: float | None = None,
    v_min: float = 0.95,
    v_max: float = 1.05,
    c_inv: float = 1.0,
    c_viol: float = 1000.0,
    time_limit_s: float = 300.0,
    mip_gap: float = 0.02,
    # Tap co-optimisation
    H_tap: np.ndarray | None = None,
    regulator_names: list[str] | None = None,
    initial_tap_ints: dict[str, int] | None = None,
    tap_range: tuple[int, int] = (-16, 16),
    max_tap_delta: int = 8,
    # Operational cost
    days_per_year: float = 365.0,
    # Tap switching cost ($/switch/regulator)
    c_switch: float = 0.0,
    # Load voltage shift (precomputed)
    load_v_shift: list[np.ndarray] | None = None,
    # Zone constraints: at most max_pv_per_zone PVs in each zone
    zones: dict[str, list[str]] | None = None,
    max_pv_per_zone: int = 1,
) -> MILPResult:
    """Formulate and solve the PV + tap co-optimisation MILP.

    Args:
        candidate_buses: bus names (columns of H_pv)
        H_pv: sensitivity matrix (n_monitored x n_candidates), pu/kW per phase
        v0: base voltages (n_monitored,)
        v_index: (bus, phase) pairs for rows of H_pv/v0
        scenarios: list of operating scenarios (with TOU prices and weights)
        n_pv: max number of PVs to place
        s_max_kw: maximum PV capacity per site (kW, 3-phase total)
        s_total_max_kw: total PV capacity budget across all sites (kW).
            If None, no total budget constraint is applied.
        v_min, v_max: voltage limits (pu)
        c_inv: annualised investment cost per kW ($/kW/year)
        c_viol: violation penalty per pu slack
        time_limit_s: solver time limit
        mip_gap: acceptable MIP gap
        H_tap: tap sensitivity matrix (n_monitored x n_regulators), pu per +1 step
        regulator_names: regulator names (columns of H_tap)
        initial_tap_ints: initial tap positions as integers
        tap_range: absolute tap limits (min, max)
        days_per_year: for annualising scenario-based savings
    """
    import gurobipy as gp
    from gurobipy import GRB

    n_mon = len(v_index)
    n_cand = len(candidate_buses)
    n_regs = len(regulator_names) if regulator_names else 0

    model = gp.Model("PV_Tap_Expansion")
    model.Params.TimeLimit = time_limit_s
    model.Params.MIPGap = mip_gap
    model.Params.OutputFlag = 1
    model.Params.MIPFocus = 1  # focus on finding good feasible solutions

    # PV decision variables
    x = model.addVars(n_cand, vtype=GRB.BINARY, name="x")
    s = model.addVars(n_cand, lb=0.0, ub=s_max_kw, name="s")

    for j in range(n_cand):
        model.addConstr(s[j] <= s_max_kw * x[j], name=f"link_{j}")

    model.addConstr(gp.quicksum(x[j] for j in range(n_cand)) <= n_pv, name="n_pv")

    # Total capacity budget
    if s_total_max_kw is not None:
        model.addConstr(
            gp.quicksum(s[j] for j in range(n_cand)) <= s_total_max_kw,
            name="s_total_max",
        )

    # Per-zone PV count limit
    if zones is not None:
        cand_lower = [b.lower() for b in candidate_buses]
        for zid, zone_buses in zones.items():
            zone_set = {b.lower() for b in zone_buses}
            zone_indices = [j for j, b in enumerate(cand_lower) if b in zone_set]
            if zone_indices:
                model.addConstr(
                    gp.quicksum(x[j] for j in zone_indices) <= max_pv_per_zone,
                    name=f"zone_{zid}",
                )
                logger.info("  Zone %s: %d candidates, max %d PV", zid, len(zone_indices), max_pv_per_zone)

    # Tap decision variables (2-hour resolution regardless of time step)
    # delta_tap[(r, sc_idx, slot)] = integer tap change from initial position
    # Taps change at most once per 2 hours; finer timesteps share the
    # same tap variable within each 2-hour slot.
    TAP_INTERVAL_H = 2  # hours per tap slot
    N_TAP_SLOTS = 24 // TAP_INTERVAL_H  # 12 slots per day
    delta_tap = {}
    if n_regs > 0 and initial_tap_ints is not None:
        for sc_idx in range(len(scenarios)):
            for h in range(N_TAP_SLOTS):
                for r, reg_name in enumerate(regulator_names):
                    init = initial_tap_ints[reg_name]
                    abs_lb = tap_range[0] - init
                    abs_ub = tap_range[1] - init
                    lb = max(abs_lb, -max_tap_delta)
                    ub = min(abs_ub, max_tap_delta)
                    delta_tap[(r, sc_idx, h)] = model.addVar(
                        vtype=GRB.INTEGER,
                        lb=lb,
                        ub=ub,
                        name=f"dtap_{reg_name}_{sc_idx}_{h}",
                    )

    # Curtailment variables
    # curtail[(j, sc_idx, t)] = curtailed PV power at bus j (kW, 3-phase total)
    # Actual PV output = s[j] * pv_frac - curtail[j, sc_idx, t]
    # This makes savings location-dependent: buses with high voltage sensitivity
    # may need curtailment, reducing their economic value.
    curtail: dict[tuple[int, int, int], any] = {}

    # Voltage constraints per scenario × time step (minute-level)
    all_slack_o = []
    all_slack_u = []

    for sc_idx, sc in enumerate(scenarios):
        T = len(sc.hours)
        for t in range(T):
            pv_frac = sc.pv_profile[t]

            # Skip time steps with no PV AND no tap variables
            if pv_frac < 1e-6 and n_regs == 0:
                continue

            # Create curtailment variables for this timestep (only when PV active)
            if pv_frac >= 1e-6:
                for j in range(n_cand):
                    c_var = model.addVar(lb=0.0, name=f"curt_{j}_{sc_idx}_{t}")
                    curtail[(j, sc_idx, t)] = c_var
                    # Can't curtail more than available generation
                    model.addConstr(
                        c_var <= s[j] * pv_frac,
                        name=f"curt_ub_{j}_{sc_idx}_{t}",
                    )

            slack_o = model.addVars(n_mon, lb=0.0, name=f"so_{sc_idx}_{t}")
            slack_u = model.addVars(n_mon, lb=0.0, name=f"su_{sc_idx}_{t}")

            # Map minute-level timestep to 2-hour tap slot
            tap_hour = min(int(sc.hours[t]) // TAP_INTERVAL_H, N_TAP_SLOTS - 1)

            for i in range(n_mon):
                # PV voltage effect (using actual output after curtailment)
                pv_voltage_effect = (
                    gp.quicksum(H_pv[i, j] * (s[j] * pv_frac - curtail[(j, sc_idx, t)]) / 3.0 for j in range(n_cand))
                    if pv_frac >= 1e-6
                    else 0
                )

                # Tap voltage effect (hourly taps applied to minute-level constraint)
                tap_voltage_effect = (
                    gp.quicksum(H_tap[i, r] * delta_tap[(r, sc_idx, tap_hour)] for r in range(n_regs))
                    if n_regs > 0
                    else 0
                )

                # Load voltage shift (known constant, not a decision variable)
                load_shift = load_v_shift[sc_idx][t, i] if load_v_shift is not None else 0.0

                v_expr = v0[i] + pv_voltage_effect + tap_voltage_effect + load_shift

                model.addConstr(v_expr >= v_min - slack_u[i], name=f"vmin_{sc_idx}_{t}_{i}")
                model.addConstr(v_expr <= v_max + slack_o[i], name=f"vmax_{sc_idx}_{t}_{i}")

            all_slack_o.extend([(slack_o[i], sc.weight) for i in range(n_mon)])
            all_slack_u.extend([(slack_u[i], sc.weight) for i in range(n_mon)])

    # Objective

    # 1. Annualised investment cost ($/year)
    investment = c_inv * gp.quicksum(s[j] for j in range(n_cand))

    # 2. Annual operational savings from PV generation ($/year)
    #    Savings are location-dependent: actual output = s[b]*pv_frac - curtail[b,t,sc]
    #    Curtailment reduces savings, making high-sensitivity buses less attractive.
    savings_expr = 0
    for sc_idx, sc in enumerate(scenarios):
        for t in range(len(sc.hours)):
            pv_frac = sc.pv_profile[t]
            if pv_frac < 1e-6:
                continue
            price = sc.price_per_kwh[t]
            actual_gen = gp.quicksum(s[j] * pv_frac - curtail[(j, sc_idx, t)] for j in range(n_cand))
            savings_expr += sc.weight * price * actual_gen * sc.hours_per_step

    annual_savings = savings_expr * days_per_year

    # 3. Voltage violation penalty (annualised, same weighting as savings)
    violation = (
        c_viol
        * days_per_year
        * (gp.quicksum(w * sv for sv, w in all_slack_o) + gp.quicksum(w * sv for sv, w in all_slack_u))
    )

    # 4. Tap switching cost: penalise changes between consecutive hours
    #    |delta_tap[r,sc,h+1] - delta_tap[r,sc,h]| via auxiliary variable
    switching_cost_expr = 0
    if n_regs > 0 and c_switch > 0 and delta_tap:
        all_switch_vars = []
        for sc_idx, sc in enumerate(scenarios):
            for h in range(N_TAP_SLOTS - 1):
                for r in range(n_regs):
                    d = model.addVar(lb=0, name=f"dswitch_{sc_idx}_{h}_{r}")
                    diff = delta_tap[(r, sc_idx, h + 1)] - delta_tap[(r, sc_idx, h)]
                    model.addConstr(d >= diff, name=f"sw_pos_{sc_idx}_{h}_{r}")
                    model.addConstr(d >= -diff, name=f"sw_neg_{sc_idx}_{h}_{r}")
                    all_switch_vars.append((d, sc.weight))
        # Annualise: c_switch * days_per_year * weighted_sum_of_switches
        switching_cost_expr = c_switch * days_per_year * gp.quicksum(w * d for d, w in all_switch_vars)

    model.setObjective(
        investment - annual_savings + violation + switching_cost_expr,
        GRB.MINIMIZE,
    )

    model.update()
    logger.info(
        "MILP: %d candidates, %d monitored, %d scenarios, %d PVs, %d regulators",
        n_cand,
        n_mon,
        len(scenarios),
        n_pv,
        n_regs,
    )
    logger.info(
        "MILP: s_max=%.0f kW, s_total_max=%s kW, c_inv=%.2f $/kW/yr, c_viol=%.0f, c_switch=%.0f, days/yr=%.0f",
        s_max_kw,
        f"{s_total_max_kw:.0f}" if s_total_max_kw else "None",
        c_inv,
        c_viol,
        c_switch,
        days_per_year,
    )
    logger.info("MILP: %d variables, %d constraints", model.NumVars, model.NumConstrs)

    t0 = time.time()
    model.optimize()
    solve_time = time.time() - t0

    if model.SolCount > 0:
        status = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        pv_locs = []
        pv_caps = []
        all_x_vals = {}
        all_s_vals = {}
        for j in range(n_cand):
            all_x_vals[candidate_buses[j]] = x[j].X
            all_s_vals[candidate_buses[j]] = s[j].X
            if x[j].X > 0.5:
                pv_locs.append(candidate_buses[j])
                pv_caps.append(s[j].X)

        # Tap schedules: scenario_name -> {hour -> {reg_name: absolute_tap}}
        # Expand 2-hour slots back to hourly schedule for downstream use
        tap_schedules_val = None
        if n_regs > 0 and initial_tap_ints is not None:
            tap_schedules_val = {}
            for sc_idx, sc in enumerate(scenarios):
                schedule = {}
                for h_slot in range(N_TAP_SLOTS):
                    hour_taps = {}
                    for r, reg_name in enumerate(regulator_names):
                        delta = int(round(delta_tap[(r, sc_idx, h_slot)].X))
                        hour_taps[reg_name] = initial_tap_ints[reg_name] + delta
                    # Expand slot to individual hours
                    for hh in range(h_slot * TAP_INTERVAL_H, (h_slot + 1) * TAP_INTERVAL_H):
                        if hh < 24:
                            schedule[hh] = hour_taps
                tap_schedules_val[sc.name] = schedule

        # Compute cost components
        inv_val = sum(all_s_vals.values()) * c_inv
        savings_val = 0.0
        curtail_kwh = 0.0  # total curtailed energy per weighted day (kWh)
        for sc_idx, sc in enumerate(scenarios):
            for t in range(len(sc.hours)):
                pv_frac = sc.pv_profile[t]
                if pv_frac < 1e-6:
                    continue
                price = sc.price_per_kwh[t]
                actual_gen = 0.0
                for j, b in enumerate(candidate_buses):
                    c_val = curtail[(j, sc_idx, t)].X if (j, sc_idx, t) in curtail else 0.0
                    gen_j = all_s_vals[b] * pv_frac - c_val
                    actual_gen += gen_j
                    curtail_kwh += sc.weight * c_val * sc.hours_per_step
                savings_val += sc.weight * price * actual_gen * sc.hours_per_step
        savings_val *= days_per_year
        curtail_mwh_annual = curtail_kwh * days_per_year / 1000.0

        # Compute switching cost from solution
        switch_val = 0.0
        if n_regs > 0 and c_switch > 0 and tap_schedules_val:
            for sc_idx, sc in enumerate(scenarios):
                for h in range(N_TAP_SLOTS - 1):
                    for r in range(n_regs):
                        d0 = int(round(delta_tap[(r, sc_idx, h)].X))
                        d1 = int(round(delta_tap[(r, sc_idx, h + 1)].X))
                        switch_val += sc.weight * abs(d1 - d0)
            switch_val *= c_switch * days_per_year

        viol_val = model.ObjVal - inv_val + savings_val - switch_val

        return MILPResult(
            status=status,
            objective=model.ObjVal,
            pv_locations=pv_locs,
            pv_capacities_kw=pv_caps,
            investment_cost=inv_val,
            operational_savings=savings_val,
            violation_penalty=viol_val,
            switching_cost=switch_val,
            curtailment_mwh=curtail_mwh_annual,
            solve_time_s=solve_time,
            all_x=all_x_vals,
            all_s=all_s_vals,
            tap_schedules=tap_schedules_val,
        )
    else:
        return MILPResult(
            status="infeasible" if model.Status == GRB.INFEASIBLE else f"status_{model.Status}",
            objective=float("inf"),
            pv_locations=[],
            pv_capacities_kw=[],
            investment_cost=0.0,
            operational_savings=0.0,
            violation_penalty=0.0,
            switching_cost=0.0,
            curtailment_mwh=0.0,
            solve_time_s=solve_time,
            all_x={},
            all_s={},
        )


# Joint PV + DC co-optimisation MILP


def solve_pv_dc_milp(
    pv_candidate_buses: list[str],
    dc_candidate_buses: list[str],
    dc_site_ids: list[str],
    dc_zone_indices: dict[str, list[int]],  # site_id -> indices into dc_candidate_buses
    dc_demand_kw: np.ndarray,  # (n_sites, T, n_scenarios) kW per phase
    H_pv: np.ndarray,
    H_dc: np.ndarray,  # (n_mon, n_dc_cand) pu/kW per phase
    v0: np.ndarray,
    v_index: list[tuple[str, str]],
    scenarios: list[Scenario],
    *,
    # PV params
    n_pv: int = 5,
    s_max_kw: float = 2000.0,
    s_total_max_kw: float | None = None,
    # Voltage limits
    v_min: float = 0.95,
    v_max: float = 1.05,
    # Cost params
    c_inv: float = 1.0,
    c_viol: float = 1000.0,
    time_limit_s: float = 600.0,
    mip_gap: float = 0.02,
    # Tap params
    H_tap: np.ndarray | None = None,
    regulator_names: list[str] | None = None,
    initial_tap_ints: dict[str, int] | None = None,
    tap_range: tuple[int, int] = (-16, 16),
    max_tap_delta: int = 8,
    days_per_year: float = 365.0,
    c_switch: float = 0.0,
    load_v_shift: list[np.ndarray] | None = None,
    # Zone constraints
    pv_zones: dict[str, list[str]] | None = None,
    max_pv_per_zone: int = 1,
) -> CoOptMILPResult:
    """Formulate and solve the joint PV + DC co-optimisation MILP.

    Args:
        pv_candidate_buses: bus names for PV placement (columns of H_pv)
        dc_candidate_buses: union of all DC candidate buses (columns of H_dc)
        dc_site_ids: ordered list of DC site identifiers
        dc_zone_indices: site_id -> list of column indices into dc_candidate_buses
        dc_demand_kw: time-varying DC demand (n_sites, T, n_scenarios), kW/phase
        H_pv: PV sensitivity matrix (n_mon x n_pv_cand), pu/kW per phase
        H_dc: DC load sensitivity matrix (n_mon x n_dc_cand), pu/kW per phase
        v0: base voltages (n_mon,)
        v_index: (bus, phase) pairs for rows
        scenarios: list of operating scenarios
        n_pv: max number of PV sites
        s_max_kw: max PV capacity per site (kW, 3-phase)
        s_total_max_kw: total PV capacity budget (kW)
        v_min, v_max: voltage limits (pu)
        c_inv: annualised investment cost per kW ($/kW/year)
        c_viol: violation penalty per pu slack
        time_limit_s: solver time limit
        mip_gap: acceptable MIP gap
        H_tap: tap sensitivity matrix
        regulator_names: regulator names
        initial_tap_ints: initial tap positions as integers
        tap_range: absolute tap limits (min, max)
        max_tap_delta: max tap change from initial
        days_per_year: for annualising savings
        c_switch: tap switching cost ($/switch)
        load_v_shift: precomputed load voltage shifts
        pv_zones: zone constraints for PV
        max_pv_per_zone: max PV per zone
    """
    import gurobipy as gp
    from gurobipy import GRB

    n_mon = len(v_index)
    n_pv_cand = len(pv_candidate_buses)
    n_dc_cand = len(dc_candidate_buses)
    n_sites = len(dc_site_ids)
    n_regs = len(regulator_names) if regulator_names else 0

    model = gp.Model("PV_DC_CoOpt")
    model.Params.TimeLimit = time_limit_s
    model.Params.MIPGap = mip_gap
    model.Params.OutputFlag = 1
    model.Params.MIPFocus = 1  # focus on finding good feasible solutions

    # PV decision variables
    x = model.addVars(n_pv_cand, vtype=GRB.BINARY, name="x")
    s = model.addVars(n_pv_cand, lb=0.0, ub=s_max_kw, name="s")

    for j in range(n_pv_cand):
        model.addConstr(s[j] <= s_max_kw * x[j], name=f"pv_link_{j}")

    model.addConstr(gp.quicksum(x[j] for j in range(n_pv_cand)) <= n_pv, name="n_pv")

    # Total PV capacity budget
    if s_total_max_kw is not None:
        model.addConstr(
            gp.quicksum(s[j] for j in range(n_pv_cand)) <= s_total_max_kw,
            name="s_total_max",
        )

    # Per-zone PV count limit
    if pv_zones is not None:
        cand_lower = [b.lower() for b in pv_candidate_buses]
        for zid, zone_buses in pv_zones.items():
            zone_set = {b.lower() for b in zone_buses}
            zone_indices = [j for j, b in enumerate(cand_lower) if b in zone_set]
            if zone_indices:
                model.addConstr(
                    gp.quicksum(x[j] for j in zone_indices) <= max_pv_per_zone,
                    name=f"pv_zone_{zid}",
                )
                logger.info(
                    "  PV zone %s: %d candidates, max %d PV",
                    zid,
                    len(zone_indices),
                    max_pv_per_zone,
                )

    # DC placement decision variables
    y: dict[tuple[int, int], gp.Var] = {}
    for k, site_id in enumerate(dc_site_ids):
        for b in dc_zone_indices[site_id]:
            y[(k, b)] = model.addVar(
                vtype=GRB.BINARY,
                name=f"y_{site_id}_{dc_candidate_buses[b]}",
            )

    # DC assignment: each site to exactly one bus
    for k, site_id in enumerate(dc_site_ids):
        model.addConstr(
            gp.quicksum(y[(k, b)] for b in dc_zone_indices[site_id]) == 1,
            name=f"dc_assign_{site_id}",
        )

    # DC mutual exclusion: at most one DC per bus
    for b in range(n_dc_cand):
        sites_at_b = [k for k in range(n_sites) if b in dc_zone_indices[dc_site_ids[k]]]
        if len(sites_at_b) > 1:
            model.addConstr(
                gp.quicksum(y[(k, b)] for k in sites_at_b) <= 1,
                name=f"dc_excl_{dc_candidate_buses[b]}",
            )

    # Tap decision variables (2-hour resolution)
    TAP_INTERVAL_H = 2
    N_TAP_SLOTS = 24 // TAP_INTERVAL_H  # 12 slots per day
    delta_tap: dict[tuple[int, int, int], gp.Var] = {}
    if n_regs > 0 and initial_tap_ints is not None:
        for sc_idx in range(len(scenarios)):
            for h in range(N_TAP_SLOTS):
                for r, reg_name in enumerate(regulator_names):
                    init = initial_tap_ints[reg_name]
                    abs_lb = tap_range[0] - init
                    abs_ub = tap_range[1] - init
                    lb = max(abs_lb, -max_tap_delta)
                    ub = min(abs_ub, max_tap_delta)
                    delta_tap[(r, sc_idx, h)] = model.addVar(
                        vtype=GRB.INTEGER,
                        lb=lb,
                        ub=ub,
                        name=f"dtap_{reg_name}_{sc_idx}_{h}",
                    )

    # Precompute DC voltage sensitivity expressions
    # dc_sens_expr[(k, i)] = sum_b H_dc[i, b] * y[k, b]
    # This is the effective sensitivity for site k at monitored bus-phase i,
    # weighted by the placement decision.
    dc_sens_expr: dict[tuple[int, int], gp.LinExpr] = {}
    for k, site_id in enumerate(dc_site_ids):
        for i in range(n_mon):
            dc_sens_expr[(k, i)] = gp.quicksum(H_dc[i, b] * y[(k, b)] for b in dc_zone_indices[site_id])

    # Curtailment variables and voltage constraints
    curtail: dict[tuple[int, int, int], gp.Var] = {}
    all_slack_o: list[tuple[gp.Var, float]] = []
    all_slack_u: list[tuple[gp.Var, float]] = []

    for sc_idx, sc in enumerate(scenarios):
        T = len(sc.hours)
        for t in range(T):
            pv_frac = sc.pv_profile[t]

            # Skip timesteps with no PV AND no tap variables AND no DC effect
            # DC always has an effect, so we only skip if truly nothing changes.
            if pv_frac < 1e-6 and n_regs == 0 and n_sites == 0:
                continue

            # Create curtailment variables (only when PV active)
            if pv_frac >= 1e-6:
                for j in range(n_pv_cand):
                    c_var = model.addVar(lb=0.0, name=f"curt_{j}_{sc_idx}_{t}")
                    curtail[(j, sc_idx, t)] = c_var
                    model.addConstr(
                        c_var <= s[j] * pv_frac,
                        name=f"curt_ub_{j}_{sc_idx}_{t}",
                    )

            slack_o = model.addVars(n_mon, lb=0.0, name=f"so_{sc_idx}_{t}")
            slack_u = model.addVars(n_mon, lb=0.0, name=f"su_{sc_idx}_{t}")

            # Map timestep to 2-hour tap slot
            tap_hour = min(int(sc.hours[t]) // TAP_INTERVAL_H, N_TAP_SLOTS - 1)

            for i in range(n_mon):
                # PV voltage effect (using actual output after curtailment)
                if pv_frac >= 1e-6:
                    pv_effect = gp.quicksum(
                        H_pv[i, j] * (s[j] * pv_frac - curtail[(j, sc_idx, t)]) / 3.0 for j in range(n_pv_cand)
                    )
                else:
                    pv_effect = 0

                # DC effect: sum over sites of demand * sensitivity expression
                # dc_demand_kw[k, t, sc_idx] is kW per phase (scalar for this t, sc)
                # dc_sens_expr[(k, i)] is the effective sensitivity (LinExpr of y vars)
                dc_effect = gp.quicksum(dc_demand_kw[k, t, sc_idx] * dc_sens_expr[(k, i)] for k in range(n_sites))

                # Tap voltage effect
                if n_regs > 0:
                    tap_effect = gp.quicksum(H_tap[i, r] * delta_tap[(r, sc_idx, tap_hour)] for r in range(n_regs))
                else:
                    tap_effect = 0

                # Load voltage shift (known constant)
                load_shift = load_v_shift[sc_idx][t, i] if load_v_shift is not None else 0.0

                v_expr = v0[i] + pv_effect + dc_effect + tap_effect + load_shift

                model.addConstr(
                    v_expr >= v_min - slack_u[i],
                    name=f"vmin_{sc_idx}_{t}_{i}",
                )
                model.addConstr(
                    v_expr <= v_max + slack_o[i],
                    name=f"vmax_{sc_idx}_{t}_{i}",
                )

            all_slack_o.extend([(slack_o[i], sc.weight) for i in range(n_mon)])
            all_slack_u.extend([(slack_u[i], sc.weight) for i in range(n_mon)])

    # Objective

    # 1. Annualised investment cost ($/year) -- PV only, no DC cost
    investment = c_inv * gp.quicksum(s[j] for j in range(n_pv_cand))

    # 2. Annual operational savings from PV generation ($/year)
    savings_expr = 0
    for sc_idx, sc in enumerate(scenarios):
        for t in range(len(sc.hours)):
            pv_frac = sc.pv_profile[t]
            if pv_frac < 1e-6:
                continue
            price = sc.price_per_kwh[t]
            actual_gen = gp.quicksum(s[j] * pv_frac - curtail[(j, sc_idx, t)] for j in range(n_pv_cand))
            savings_expr += sc.weight * price * actual_gen * sc.hours_per_step

    annual_savings = savings_expr * days_per_year

    # 3. Voltage violation penalty
    violation = (
        c_viol
        * days_per_year
        * (gp.quicksum(w * sv for sv, w in all_slack_o) + gp.quicksum(w * sv for sv, w in all_slack_u))
    )

    # 4. Tap switching cost
    switching_cost_expr = 0
    if n_regs > 0 and c_switch > 0 and delta_tap:
        all_switch_vars: list[tuple[gp.Var, float]] = []
        for sc_idx, sc in enumerate(scenarios):
            for h in range(N_TAP_SLOTS - 1):
                for r in range(n_regs):
                    d = model.addVar(lb=0, name=f"dswitch_{sc_idx}_{h}_{r}")
                    diff = delta_tap[(r, sc_idx, h + 1)] - delta_tap[(r, sc_idx, h)]
                    model.addConstr(d >= diff, name=f"sw_pos_{sc_idx}_{h}_{r}")
                    model.addConstr(d >= -diff, name=f"sw_neg_{sc_idx}_{h}_{r}")
                    all_switch_vars.append((d, sc.weight))
        switching_cost_expr = c_switch * days_per_year * gp.quicksum(w * d for d, w in all_switch_vars)

    model.setObjective(
        investment - annual_savings + violation + switching_cost_expr,
        GRB.MINIMIZE,
    )

    model.update()
    logger.info(
        "MILP: %d PV cand, %d DC cand, %d DC sites, %d monitored, %d scenarios, %d PVs, %d regulators",
        n_pv_cand,
        n_dc_cand,
        n_sites,
        n_mon,
        len(scenarios),
        n_pv,
        n_regs,
    )
    logger.info(
        "MILP: s_max=%.0f kW, s_total_max=%s kW, c_inv=%.2f $/kW/yr, c_viol=%.0f, c_switch=%.0f, days/yr=%.0f",
        s_max_kw,
        f"{s_total_max_kw:.0f}" if s_total_max_kw else "None",
        c_inv,
        c_viol,
        c_switch,
        days_per_year,
    )
    logger.info("MILP: %d variables, %d constraints", model.NumVars, model.NumConstrs)

    t0 = time.time()
    model.optimize()
    solve_time = time.time() - t0

    if model.SolCount > 0:
        status = "optimal" if model.Status == GRB.OPTIMAL else "feasible"

        # Extract PV results
        pv_locs: list[str] = []
        pv_caps: list[float] = []
        all_x_vals: dict[str, float] = {}
        all_s_vals: dict[str, float] = {}
        for j in range(n_pv_cand):
            all_x_vals[pv_candidate_buses[j]] = x[j].X
            all_s_vals[pv_candidate_buses[j]] = s[j].X
            if x[j].X > 0.5:
                pv_locs.append(pv_candidate_buses[j])
                pv_caps.append(s[j].X)

        # Extract DC assignments
        dc_assignments: dict[str, str] = {}
        all_y_vals: dict[tuple[str, str], float] = {}
        for k, site_id in enumerate(dc_site_ids):
            for b in dc_zone_indices[site_id]:
                val = y[(k, b)].X
                all_y_vals[(site_id, dc_candidate_buses[b])] = val
                if val > 0.5:
                    dc_assignments[site_id] = dc_candidate_buses[b]

        # Extract tap schedules
        tap_schedules_val = None
        if n_regs > 0 and initial_tap_ints is not None:
            tap_schedules_val = {}
            for sc_idx, sc in enumerate(scenarios):
                schedule: dict[int, dict[str, int]] = {}
                for h_slot in range(N_TAP_SLOTS):
                    hour_taps = {}
                    for r, reg_name in enumerate(regulator_names):
                        delta = int(round(delta_tap[(r, sc_idx, h_slot)].X))
                        hour_taps[reg_name] = initial_tap_ints[reg_name] + delta
                    for hh in range(h_slot * TAP_INTERVAL_H, (h_slot + 1) * TAP_INTERVAL_H):
                        if hh < 24:
                            schedule[hh] = hour_taps
                tap_schedules_val[sc.name] = schedule

        # Compute cost components
        inv_val = sum(all_s_vals.values()) * c_inv
        savings_val = 0.0
        curtail_kwh = 0.0
        for sc_idx, sc in enumerate(scenarios):
            for t in range(len(sc.hours)):
                pv_frac = sc.pv_profile[t]
                if pv_frac < 1e-6:
                    continue
                price = sc.price_per_kwh[t]
                actual_gen = 0.0
                for j, b in enumerate(pv_candidate_buses):
                    c_val = curtail[(j, sc_idx, t)].X if (j, sc_idx, t) in curtail else 0.0
                    gen_j = all_s_vals[b] * pv_frac - c_val
                    actual_gen += gen_j
                    curtail_kwh += sc.weight * c_val * sc.hours_per_step
                savings_val += sc.weight * price * actual_gen * sc.hours_per_step
        savings_val *= days_per_year
        curtail_mwh_annual = curtail_kwh * days_per_year / 1000.0

        # Compute switching cost from solution
        switch_val = 0.0
        if n_regs > 0 and c_switch > 0 and tap_schedules_val:
            for sc_idx, sc in enumerate(scenarios):
                for h in range(N_TAP_SLOTS - 1):
                    for r in range(n_regs):
                        d0 = int(round(delta_tap[(r, sc_idx, h)].X))
                        d1 = int(round(delta_tap[(r, sc_idx, h + 1)].X))
                        switch_val += sc.weight * abs(d1 - d0)
            switch_val *= c_switch * days_per_year

        viol_val = model.ObjVal - inv_val + savings_val - switch_val

        return CoOptMILPResult(
            status=status,
            objective=model.ObjVal,
            pv_locations=pv_locs,
            pv_capacities_kw=pv_caps,
            dc_assignments=dc_assignments,
            investment_cost=inv_val,
            operational_savings=savings_val,
            violation_penalty=viol_val,
            switching_cost=switch_val,
            curtailment_mwh=curtail_mwh_annual,
            solve_time_s=solve_time,
            all_x=all_x_vals,
            all_s=all_s_vals,
            all_y=all_y_vals,
            tap_schedules=tap_schedules_val,
        )
    else:
        return CoOptMILPResult(
            status=("infeasible" if model.Status == GRB.INFEASIBLE else f"status_{model.Status}"),
            objective=float("inf"),
            pv_locations=[],
            pv_capacities_kw=[],
            dc_assignments={},
            investment_cost=0.0,
            operational_savings=0.0,
            violation_penalty=0.0,
            switching_cost=0.0,
            curtailment_mwh=0.0,
            solve_time_s=solve_time,
            all_x={},
            all_s={},
            all_y={},
        )


# Validation via direct OpenDSS power flow


def validate_with_opendss(
    case_dir: Path,
    master_file: str,
    pv_locations: list[str],
    pv_capacities_kw: list[float],
    scenario: Scenario,
    v_index: list[tuple[str, str]],
    *,
    source_pu: float = 1.05,
    initial_taps: dict[str, float] | None = None,
    bus_kv: float = 24.9,
    dc_sites: dict[str, dict] | None = None,
    tap_schedule: dict[int, dict[str, int]] | None = None,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> dict:
    """Validate MILP result using direct OpenDSS static power flow.

    Sets up the same circuit as the sensitivity computation (including DC loads),
    then solves power flow at each scenario time step with optimised PV and
    time-varying taps.

    Args:
        tap_schedule: {hour: {reg_name: absolute_tap_int}} -- per-hour tap positions.
    """
    # Compile circuit (same setup as sensitivity computation)
    dss.Basic.ClearAll()
    dss.Text.Command(f"Redirect [{case_dir / master_file}]")

    if source_pu is not None:
        dss.Text.Command(f"Edit Vsource.source pu={source_pu}")

    dss.Text.Command("Set controlmode=off")

    # Set initial taps (will be overridden by tap_positions if provided)
    base_taps: dict[str, float] = {}
    if initial_taps:
        for reg_name, tap_pu in initial_taps.items():
            dss.RegControls.Name(reg_name)
            xfmr = dss.RegControls.Transformer()
            dss.Transformers.Name(xfmr)
            dss.Transformers.Wdg(2)
            dss.Transformers.Tap(tap_pu)
            base_taps[reg_name] = tap_pu

    # Add DC loads (same as sensitivity)
    _add_dc_loads(dc_sites, bus_kv)

    # Add PV load elements (initially 0)
    pv_ln_kv = bus_kv / math.sqrt(3.0)
    for bus, _cap_kw in zip(pv_locations, pv_capacities_kw, strict=False):
        for ph, suffix in [(1, "a"), (2, "b"), (3, "c")]:
            dss.Text.Command(
                f"New Load.PV_{bus}_{suffix} bus1={bus}.{ph} phases=1 "
                f"conn=wye kV={pv_ln_kv:.6f} kW=0 kvar=0 model=1 vminpu=0.85"
            )

    # Add time-varying load elements (initially 0)
    load_buses_in_scenario = list(scenario.load_kw_per_bus.keys())
    ln_kv = bus_kv / math.sqrt(3.0)
    for bus in load_buses_in_scenario:
        for ph, suffix in [(1, "a"), (2, "b"), (3, "c")]:
            dss.Text.Command(
                f"New Load.TVL_{bus}_{suffix} bus1={bus}.{ph} phases=1 "
                f"conn=wye kV={ln_kv:.6f} kW=0 kvar=0 model=1 vminpu=0.85"
            )

    # Sweep time steps and record voltages
    worst_vmin = 1.5
    worst_vmax = 0.0
    violation_steps = 0
    total_steps = len(scenario.hours)

    for t, hour in enumerate(scenario.hours):
        pv_frac = scenario.pv_profile[t]

        # Set PV generation
        for bus, cap_kw in zip(pv_locations, pv_capacities_kw, strict=False):
            pv_per_phase = cap_kw / 3.0 * pv_frac
            for suffix in ("a", "b", "c"):
                dss.Loads.Name(f"PV_{bus}_{suffix}")
                dss.Loads.kW(-pv_per_phase)

        # Set time-varying loads
        for bus in load_buses_in_scenario:
            load_kw = scenario.load_kw_per_bus[bus][t]  # kW per phase
            for suffix in ("a", "b", "c"):
                dss.Loads.Name(f"TVL_{bus}_{suffix}")
                dss.Loads.kW(load_kw)

        # Set taps for this time step
        hour_int = int(hour)
        if tap_schedule is not None and hour_int in tap_schedule:
            for reg_name, tap_int in tap_schedule[hour_int].items():
                dss.RegControls.Name(reg_name)
                xfmr = dss.RegControls.Transformer()
                dss.Transformers.Name(xfmr)
                dss.Transformers.Wdg(2)
                dss.Transformers.Tap(1.0 + tap_int * TAP_STEP)

        dss.Solution.SolveNoControl()

        # Measure voltages at monitored bus-phases
        step_vmin = 1.5
        step_vmax = 0.0
        for bus, phase in v_index:
            dss.Circuit.SetActiveBus(bus)
            v_mag = dss.Bus.puVmagAngle()
            node = {"a": 0, "b": 1, "c": 2}[phase]
            if node < len(v_mag) // 2:
                v = v_mag[2 * node]
                step_vmin = min(step_vmin, v)
                step_vmax = max(step_vmax, v)

        worst_vmin = min(worst_vmin, step_vmin)
        worst_vmax = max(worst_vmax, step_vmax)
        if step_vmin < v_min or step_vmax > v_max:
            violation_steps += 1

    result = {
        "scenario": scenario.name,
        "pv_locations": pv_locations,
        "pv_capacities_kw": pv_capacities_kw,
        "total_pv_kw": sum(pv_capacities_kw),
        "violation_steps": violation_steps,
        "total_steps": total_steps,
        "worst_vmin": worst_vmin,
        "worst_vmax": worst_vmax,
    }

    logger.info(
        "Validation [%s]: %d/%d steps with violations, vmin=%.4f, vmax=%.4f",
        scenario.name,
        violation_steps,
        total_steps,
        worst_vmin,
        worst_vmax,
    )
    return result


# Co-opt validation wrapper


def validate_coopt(
    *,
    case_dir: Path,
    master_file: str,
    pv_locations: list[str],
    pv_capacities_kw: list[float],
    dc_assignments: dict[str, str],
    dc_demand_kw: np.ndarray,
    dc_site_ids: list[str],
    scenario: Scenario,
    sc_idx: int,
    v_index: list[tuple[str, str]],
    source_pu: float = 1.05,
    initial_taps: dict[str, float] | None = None,
    bus_kv: float = 24.9,
    tap_schedule: dict[int, dict[str, int]] | None = None,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> dict:
    """Validate co-optimisation result using OpenDSS power flow.

    Embeds DC demand into the scenario's `load_kw_per_bus` and calls
    `validate_with_opendss` with `dc_sites=None` (no fixed DC loads).
    """
    # Create a modified scenario with DC demands embedded as time-varying loads
    mod_sc = copy.deepcopy(scenario)

    for k, site_id in enumerate(dc_site_ids):
        assigned_bus = dc_assignments.get(site_id)
        if assigned_bus is None:
            continue
        # dc_demand_kw[k, :, sc_idx] is the total demand in kW per phase
        dc_profile = dc_demand_kw[k, :, sc_idx]
        if assigned_bus in mod_sc.load_kw_per_bus:
            mod_sc.load_kw_per_bus[assigned_bus] = mod_sc.load_kw_per_bus[assigned_bus] + dc_profile
        else:
            mod_sc.load_kw_per_bus[assigned_bus] = dc_profile.copy()

    return validate_with_opendss(
        case_dir=case_dir,
        master_file=master_file,
        pv_locations=pv_locations,
        pv_capacities_kw=pv_capacities_kw,
        scenario=mod_sc,
        v_index=v_index,
        source_pu=source_pu,
        initial_taps=initial_taps,
        bus_kv=bus_kv,
        dc_sites=None,  # DC loads are in load_kw_per_bus
        tap_schedule=tap_schedule,
        v_min=v_min,
        v_max=v_max,
    )


# Plotting


def plot_results(
    candidate_buses: list[str],
    result: MILPResult,
    save_dir: Path,
    system: str,
) -> None:
    """Bar chart of PV capacity by bus + highlight selected sites."""
    fig, ax = plt.subplots(figsize=(max(10, len(candidate_buses) * 0.8), 6))

    caps = [result.all_s.get(b, 0.0) for b in candidate_buses]
    colors = ["#DD8452" if result.all_x.get(b, 0) > 0.5 else "#4C72B0" for b in candidate_buses]

    x = np.arange(len(candidate_buses))
    ax.bar(x, caps, color=colors)
    ax.set_xlabel("Bus")
    ax.set_ylabel("Installed PV Capacity (kW)")
    ax.set_title(f"PV Expansion Planning -- {system}")
    ax.set_xticks(x)
    ax.set_xticklabels(candidate_buses, rotation=45, ha="right")

    # Legend
    ax.legend(
        handles=[
            Patch(color="#DD8452", label="Selected"),
            Patch(color="#4C72B0", label="Not selected"),
        ]
    )

    fig.tight_layout()
    fig.savefig(save_dir / f"pv_expansion_{system}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", save_dir / f"pv_expansion_{system}.png")


def plot_coopt_results(
    pv_candidate_buses: list[str],
    dc_candidate_buses: list[str],
    result: CoOptMILPResult,
    save_dir: Path,
    system: str,
    dc_demand_kw: np.ndarray | None = None,
    dc_site_ids: list[str] | None = None,
) -> None:
    """Bar chart of PV capacities + DC load bar chart with assignment labels.

    Args:
        dc_demand_kw: shape (n_sites, T, n_scenarios), kW per phase.
            If provided, shows peak DC load at each assigned bus.
        dc_site_ids: ordered site IDs matching dc_demand_kw axis 0.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PV placements
    ax = axes[0]
    caps = [result.all_s.get(b, 0.0) for b in pv_candidate_buses]
    colors = ["#DD8452" if result.all_x.get(b, 0) > 0.5 else "#4C72B0" for b in pv_candidate_buses]
    xp = np.arange(len(pv_candidate_buses))
    ax.bar(xp, caps, color=colors)
    ax.set_xlabel("Bus")
    ax.set_ylabel("Installed PV Capacity (kW)")
    ax.set_title(f"PV Placements -- {system}")
    ax.set_xticks(xp)
    ax.set_xticklabels(pv_candidate_buses, rotation=45, ha="right", fontsize=8)
    ax.legend(
        handles=[
            Patch(color="#DD8452", label="Selected"),
            Patch(color="#4C72B0", label="Not selected"),
        ]
    )

    # DC assignments -- show peak load (kW per phase) at each candidate bus
    ax = axes[1]
    assigned_buses = set(result.dc_assignments.values())

    # Build per-bus peak demand from dc_demand_kw
    bus_peak_kw: dict[str, float] = {}
    bus_site_label: dict[str, str] = {}
    if dc_demand_kw is not None and dc_site_ids is not None:
        for site_id, bus in result.dc_assignments.items():
            if site_id in dc_site_ids:
                k = dc_site_ids.index(site_id)
                peak = float(dc_demand_kw[k].max())  # max over T and scenarios
                bus_peak_kw[bus] = bus_peak_kw.get(bus, 0.0) + peak
                bus_site_label[bus] = site_id

    bar_vals = [bus_peak_kw.get(b, 0.0) for b in dc_candidate_buses]
    dc_colors = ["#55A868" if b in assigned_buses else "#CCCCCC" for b in dc_candidate_buses]
    xd = np.arange(len(dc_candidate_buses))
    ax.bar(xd, bar_vals, color=dc_colors)
    ax.set_xlabel("Bus")
    ax.set_ylabel("DC Peak Load (kW per phase)")
    ax.set_title(f"DC Assignments -- {system}")
    ax.set_xticks(xd)
    ax.set_xticklabels(dc_candidate_buses, rotation=45, ha="right", fontsize=8)

    # Annotate with site IDs above assigned bars
    for site_id, bus in result.dc_assignments.items():
        if bus in dc_candidate_buses:
            idx = dc_candidate_buses.index(bus)
            y_val = bus_peak_kw.get(bus, 0.0)
            ax.annotate(
                site_id,
                (idx, y_val),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.legend(
        handles=[
            Patch(color="#55A868", label="DC assigned"),
            Patch(color="#CCCCCC", label="Not assigned"),
        ]
    )

    fig.suptitle(f"PV + DC Co-Optimisation -- {system}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_dir / f"pv_dc_placements_{system}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", save_dir / f"pv_dc_placements_{system}.png")


def plot_sensitivity_heatmap(
    H_pv: np.ndarray,
    v_index: list[tuple[str, str]],
    candidate_buses: list[str],
    save_dir: Path,
    system: str,
) -> None:
    """Heatmap of voltage sensitivity to PV injection."""
    # Aggregate: mean sensitivity per monitored bus (average over phases)
    bus_set = sorted(set(b for b, _ in v_index))
    H_agg = np.zeros((len(bus_set), len(candidate_buses)))
    for i, (bus, _phase) in enumerate(v_index):
        row = bus_set.index(bus)
        H_agg[row, :] += H_pv[i, :] / 3.0  # average over 3 phases

    fig, ax = plt.subplots(figsize=(max(10, len(candidate_buses) * 0.6), max(6, len(bus_set) * 0.3)))
    im = ax.imshow(H_agg * 1000, aspect="auto", cmap="RdYlGn")  # scale to mpu/kW
    ax.set_xticks(range(len(candidate_buses)))
    ax.set_xticklabels(candidate_buses, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(bus_set)))
    ax.set_yticklabels(bus_set, fontsize=8)
    ax.set_xlabel("PV Injection Bus")
    ax.set_ylabel("Monitored Bus")
    ax.set_title(f"Voltage Sensitivity to PV (mpu/kW per phase) -- {system}")
    fig.colorbar(im, ax=ax, label="mpu/kW")
    fig.tight_layout()
    fig.savefig(save_dir / f"pv_sensitivity_{system}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sensitivity heatmap saved: %s", save_dir / f"pv_sensitivity_{system}.png")


def plot_scenario_profiles(
    scenarios: list[Scenario],
    save_dir: Path,
    system: str,
    dc_sites_info: list[tuple[str, float, float, str]] | None = None,
    dc_demand_kw: np.ndarray | None = None,
    dc_site_ids: list[str] | None = None,
) -> None:
    """Plot PV profiles, load profiles, DC demand profiles, and TOU prices.

    Generates a single figure with four subplots showing all scenario
    curves, so the user can visually inspect the inputs to the optimisation.

    DC demand can be supplied in two ways (checked in order):
      1. `dc_demand_kw` + `dc_site_ids` -- precomputed array of shape
         (n_sites, T, n_scenarios) in kW per phase.
      2. `dc_sites_info` -- list of (bus, peak_kw, base_kw, site_id) tuples;
         demand is reconstructed from the scenario load profiles.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(scenarios), 1))

    # PV profiles
    ax = axes[0, 0]
    for i, sc in enumerate(scenarios):
        ax.plot(sc.hours, sc.pv_profile, label=sc.name, color=cmap(i), linewidth=1.5)
    ax.set_ylabel("Normalised PV Output")
    ax.set_title("PV Generation Profiles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # TOU prices
    ax = axes[0, 1]
    # All scenarios share the same price profile
    ax.step(scenarios[0].hours, scenarios[0].price_per_kwh, where="post", color="#4C72B0", linewidth=2)
    ax.set_ylabel("Price ($/kWh)")
    ax.set_title("Time-of-Use Electricity Price")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Time-varying load profiles (non-DC buses)
    ax = axes[1, 0]
    dc_buses: set[str] = set()
    if dc_sites_info:
        dc_buses = {info[0] for info in dc_sites_info}
    sc0 = scenarios[0]  # Use first scenario as representative
    load_buses_plotted = []
    for bus in sc0.load_kw_per_bus:
        if bus not in dc_buses:
            load_buses_plotted.append(bus)
    if load_buses_plotted:
        load_cmap = plt.colormaps.get_cmap("Set2").resampled(max(len(load_buses_plotted), 1))
        for j, bus in enumerate(load_buses_plotted):
            ax.plot(sc0.hours, sc0.load_kw_per_bus[bus], label=bus, color=load_cmap(j), linewidth=1.5)
        ax.legend(fontsize=8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Load (kW per phase)")
    ax.set_title(f"Time-Varying Load Profiles ({sc0.name})")
    ax.grid(True, alpha=0.3)

    # DC demand profiles
    ax = axes[1, 1]
    if dc_demand_kw is not None and dc_site_ids is not None:
        # Precomputed demand arrays (from co-opt path)
        dc_cmap = plt.colormaps.get_cmap("Dark2").resampled(max(len(dc_site_ids), 1))
        for k, site_id in enumerate(dc_site_ids):
            # Plot first scenario (sc_idx=0); shape is (T,)
            ax.plot(sc0.hours, dc_demand_kw[k, :, 0], label=site_id, color=dc_cmap(k), linewidth=1.5)
        ax.legend(fontsize=8)
    elif dc_sites_info:
        # Reconstruct from scenario load profiles
        dc_cmap = plt.colormaps.get_cmap("Dark2").resampled(max(len(dc_sites_info), 1))
        for k, (dc_bus, _dc_peak, dc_base, site_id) in enumerate(dc_sites_info):
            # Plot the actual demand (not the deviation from base)
            if dc_bus in sc0.load_kw_per_bus:
                actual = sc0.load_kw_per_bus[dc_bus] + dc_base
            else:
                actual = np.full_like(sc0.hours, dc_base)
            ax.plot(sc0.hours, actual, label=f"{site_id} ({dc_bus})", color=dc_cmap(k), linewidth=1.5)
        ax.legend(fontsize=8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("DC Load (kW per phase)")
    ax.set_title(f"DC Demand Profiles ({sc0.name})")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Scenario Profiles -- {system}", fontsize=14)
    fig.tight_layout()
    save_path = save_dir / f"scenario_profiles_{system}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Scenario profiles saved: %s", save_path)


# OFO comparison


def compare_with_ofo(
    *,
    sys: dict,
    dc_sites: dict,
    pv_locations: list[str],
    pv_capacities_kw: list[float],
    tap_schedule_by_scenario: dict[str, dict[int, dict[str, int]]] | None,
    save_dir: Path,
    system: str,
) -> None:
    """Run coordinator simulations with and without OFO using MILP-optimized PV.

    Compares three modes:
      1. tap_only   -- MILP tap schedule, no OFO (batch sizes stay fixed)
      2. ofo_only   -- OFO batch-size control, fixed initial taps (no MILP schedule)
      3. tap_plus_ofo -- MILP tap schedule + OFO batch-size control

    Uses the first scenario's tap schedule (summer_clear) for the 1-hour simulation.

    Args:
        sys: System constants dict from `systems.py`.
        dc_sites: `{site_id: DCSite}` dict describing the datacenter sites.
    """
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DT_DC = Fraction(1, 10)
    DT_GRID = Fraction(1, 10)
    DT_CTRL = Fraction(1)
    V_MIN, V_MAX = 0.95, 1.05
    TOTAL_DURATION_S = 3600
    POWER_AUG = PowerAugmentationConfig(amplitude_scale_range=(0.98, 1.02), noise_fraction=0.005)

    LLAMA_8B = InferenceModelSpec(
        model_label="Llama-3.1-8B",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpus_per_replica=1,
        itl_deadline_s=0.08,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    LLAMA_70B = InferenceModelSpec(
        model_label="Llama-3.1-70B",
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        gpus_per_replica=4,
        itl_deadline_s=0.10,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    LLAMA_405B = InferenceModelSpec(
        model_label="Llama-3.1-405B",
        model_id="meta-llama/Llama-3.1-405B-Instruct-FP8",
        gpus_per_replica=8,
        itl_deadline_s=0.12,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    QWEN_30B = InferenceModelSpec(
        model_label="Qwen3-30B-A3B",
        model_id="Qwen/Qwen3-30B-A3B-Thinking-2507",
        gpus_per_replica=2,
        itl_deadline_s=0.06,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    QWEN_235B = InferenceModelSpec(
        model_label="Qwen3-235B-A22B",
        model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
        gpus_per_replica=8,
        itl_deadline_s=0.14,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    ALL_MODEL_SPECS = (LLAMA_8B, LLAMA_70B, LLAMA_405B, QWEN_30B, QWEN_235B)
    _MODEL_SPECS = {s.model_label: s for s in ALL_MODEL_SPECS}

    def _deploy(
        label: str, num_replicas: int, initial_batch_size: int = 128
    ) -> tuple[ModelDeployment, ReplicaSchedule]:
        return (
            ModelDeployment(spec=_MODEL_SPECS[label], initial_batch_size=initial_batch_size),
            ReplicaSchedule(initial=num_replicas),
        )

    def _load_data_sources(config_path=None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "data_sources.json"
        with open(config_path) as f:
            cfg = json.load(f)
        sources_raw = cfg["data_sources"]
        data_sources = {s["model_label"]: MLEnergySource(**s) for s in sources_raw}
        blob = json.dumps(
            sorted(sources_raw, key=lambda s: s["model_label"]),
            sort_keys=True,
        ).encode()
        data_dir = _PROJECT_ROOT / "data" / "offline" / hashlib.sha256(blob).hexdigest()[:16]
        return data_sources, data_dir

    logger.info("")
    logger.info("=" * 70)
    logger.info("OFO COMPARISON: Running coordinator simulations")
    logger.info("=" * 70)

    # Load data pipeline
    data_sources, data_dir = _load_data_sources()
    all_models = ALL_MODEL_SPECS

    logger.info("  Loading inference data...")
    inference_data = InferenceData.ensure(
        data_dir,
        all_models,
        data_sources,
        plot=False,
        dt_s=float(DT_DC),
    )
    _ = TrainingTrace.ensure(data_dir / "training_trace.csv")
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv",
        all_models,
        data_sources,
        plot=False,
    )

    # Add MILP-optimized PV systems as (bus, Generator) tuples
    generators = [
        (bus, SyntheticPV(peak_kw=cap_kw / 3.0)) for bus, cap_kw in zip(pv_locations, pv_capacities_kw, strict=False)
    ]
    logger.info("  PV systems: %s", ", ".join(f"{bus}={gen._peak_kw:.0f}kW/ph" for bus, gen in generators))

    # Build MILP tap schedule for coordinator (use first scenario)
    milp_tap_schedule: TapSchedule | None = None
    if tap_schedule_by_scenario:
        sc_name = next(iter(tap_schedule_by_scenario))
        sc_schedule = tap_schedule_by_scenario[sc_name]
        logger.info("  Using tap schedule from scenario: %s", sc_name)
        entries: list[tuple[float, TapPosition]] = []
        for hour, taps in sorted(sc_schedule.items()):
            time_s = hour / 24.0 * TOTAL_DURATION_S
            tap_pu = {reg: 1.0 + tap_int * TAP_STEP for reg, tap_int in taps.items()}
            entries.append((time_s, TapPosition(regulators=tap_pu)))
        milp_tap_schedule = TapSchedule(tuple(entries))

    # OFO config
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

    ofo_save_dir = save_dir / "ofo_comparison"
    ofo_save_dir.mkdir(parents=True, exist_ok=True)

    exclude_buses = tuple(sys["exclude_buses"])

    # Build datacenters from DCSite descriptors
    datacenters: list[OfflineDatacenter] = []
    dc_specs_map: dict[str, tuple] = {}
    for site_id, site in dc_sites.items():
        specs = tuple(m.spec for m, _ in site.models)
        site_inference = inference_data.filter_models(specs)
        dc = OfflineDatacenter(
            DatacenterConfig(gpus_per_server=8, base_kw_per_phase=site.base_kw_per_phase),
            OfflineWorkload(
                inference_data=site_inference,
                replica_schedules={m.spec.model_label: s for m, s in site.models},
            ),
            name=site_id,
            dt_s=DT_DC,
            seed=site.seed,
            power_augmentation=POWER_AUG,
            total_gpu_capacity=site.total_gpu_capacity,
        )
        datacenters.append(dc)
        dc_specs_map[site_id] = specs

    # Run all three modes inline (construct grid + controllers per mode)
    results: dict[str, VoltageStats] = {}
    perf_results: dict[str, PerformanceStats] = {}
    for mode_name, ctrl_mode, sched in [
        ("tap_only", "baseline", milp_tap_schedule),
        ("ofo_only", "ofo", None),
        ("tap_plus_ofo", "ofo", milp_tap_schedule),
    ]:
        logger.info("  Running %s (ctrl=%s, tap_schedule=%s)...", mode_name, ctrl_mode, sched is not None)

        grid = OpenDSSGrid(
            dss_case_dir=sys["dss_case_dir"],
            dss_master_file=sys["dss_master_file"],
            source_pu=sys["source_pu"],
            dt_s=DT_GRID,
            initial_tap_position=sys["initial_taps"],
            exclude_buses=exclude_buses,
        )
        for dc, site in zip(datacenters, dc_sites.values(), strict=True):
            grid.attach_dc(dc, bus=site.bus, connection_type=site.connection_type)
        for bus, gen in generators:
            grid.attach_generator(gen, bus=bus)

        controllers: list = []
        controllers.append(
            TapScheduleController(schedule=sched if sched is not None else TapSchedule(()), dt_s=DT_CTRL)
        )

        if ctrl_mode == "ofo":
            for dc in datacenters:
                specs = dc_specs_map[dc.name]
                site = dc_sites[dc.name]
                controllers.append(
                    OFOBatchSizeController(
                        specs,
                        datacenter=dc,
                        models=logistic_models,
                        config=ofo_config,
                        dt_s=DT_CTRL,
                        initial_batch_sizes={m.spec.model_label: m.initial_batch_size for m, _ in site.models},
                        grid=grid,
                    )
                )

        coord = Coordinator(
            datacenters=datacenters,
            grid=grid,
            controllers=controllers,
            total_duration_s=TOTAL_DURATION_S,
        )
        log = coord.run()
        stats = compute_allbus_voltage_stats(log.grid_states, v_min=V_MIN, v_max=V_MAX, exclude_buses=exclude_buses)
        pstats = compute_performance_stats(
            log.dc_states, itl_deadline_s_by_model={ms.model_label: ms.itl_deadline_s for ms in all_models}
        )

        results[mode_name] = stats
        perf_results[mode_name] = pstats
        logger.info(
            "    Violation time: %.1f s, Vmin: %.4f, Vmax: %.4f, Integral: %.4f, Thpt: %.1f k tok/s, ITL miss: %.2f%%",
            stats.violation_time_s,
            stats.worst_vmin,
            stats.worst_vmax,
            stats.integral_violation_pu_s,
            pstats.mean_throughput_tps / 1e3,
            pstats.itl_deadline_fraction * 100.0,
        )

    # Comparison CSV
    csv_path = ofo_save_dir / f"ofo_comparison_{system}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "violation_time_s",
                "worst_vmin",
                "worst_vmax",
                "integral_violation_pu_s",
                "mean_throughput_tps",
                "integrated_throughput_tokens",
                "itl_deadline_fraction",
            ]
        )
        for mode_name, s in results.items():
            p = perf_results[mode_name]
            writer.writerow(
                [
                    mode_name,
                    s.violation_time_s,
                    s.worst_vmin,
                    s.worst_vmax,
                    s.integral_violation_pu_s,
                    p.mean_throughput_tps,
                    p.integrated_throughput_tokens,
                    p.itl_deadline_fraction,
                ]
            )
    logger.info("OFO comparison CSV: %s", csv_path)

    # Comparison bar chart
    modes = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    vals = [results[m].violation_time_s for m in modes]
    axes[0].bar(modes, vals, color=colors)
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Voltage Violation Time")
    axes[0].tick_params(axis="x", rotation=15)

    vals = [results[m].worst_vmin for m in modes]
    axes[1].bar(modes, vals, color=colors)
    axes[1].axhline(0.95, color="red", linestyle="--", alpha=0.7, label="Vmin")
    axes[1].set_ylabel("Per Unit")
    axes[1].set_title("Worst Vmin")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=15)

    vals = [results[m].integral_violation_pu_s for m in modes]
    axes[2].bar(modes, vals, color=colors)
    axes[2].set_ylabel("pu * s")
    axes[2].set_title("Integral Violation")
    axes[2].tick_params(axis="x", rotation=15)

    fig.suptitle(f"PV Expansion + OFO Comparison -- {system}", fontsize=14)
    fig.tight_layout()
    fig.savefig(ofo_save_dir / f"ofo_comparison_{system}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("OFO comparison plot: %s", ofo_save_dir / f"ofo_comparison_{system}.png")

    # Print summary table
    summary_lines = [
        "",
        "=" * 90,
        f"PV Expansion + OFO Comparison -- {system}",
        "=" * 90,
        f"  PV: {', '.join(f'{b}={c:.0f}kW' for b, c in zip(pv_locations, pv_capacities_kw, strict=False))}",
        f"{'Mode':<16s} {'Viol(s)':>10s} {'Vmin':>10s} {'Vmax':>10s} {'Integral':>14s}",
        "-" * 90,
    ]
    for mode_name, s in results.items():
        summary_lines.append(
            f"{mode_name:<16s} {s.violation_time_s:>10.1f} {s.worst_vmin:>10.4f} "
            f"{s.worst_vmax:>10.4f} {s.integral_violation_pu_s:>14.4f}"
        )
    summary_lines.append("-" * 90)
    for line in summary_lines:
        logger.info(line)
        print(line)


# PV+DC co-optimisation main


def main_pv_and_dc(
    *,
    system: str = "ieee34",
    n_pv: int = 3,
    s_max_kw: float = 500.0,
    s_total_max_kw: float | None = None,
    c_inv: float = 200.0,
    c_viol: float = 5_000.0,
    c_switch: float = 10.0,
    max_tap_delta: int = 8,
    time_limit_s: float = 600.0,
    validate: bool = True,
    v_min: float = 0.95,
    v_max: float = 1.05,
    days_per_year: float = 365.0,
    no_dc_zones: bool = False,
) -> None:
    """Run joint PV + DC location co-optimisation.

    All experiment parameters (system constants, DC sites, time-varying loads)
    are defined inline per system.  No external JSON config required.

    Steps:
    1. Build system config with inline parameters
    2. Discover PV candidate buses (all 3-phase, zone-filtered)
    3. Discover DC candidate buses (per-zone from system zones)
    4. Estimate DC peak demand per site
    5. Generate scenarios (WITHOUT dc_sites_info since DC locations are variable)
    6. Compute sensitivities (H_pv, H_dc, H_tap, H_load) on bare circuit
    7. Compute DC demand profiles
    8. Solve MILP
    9. Validate
    10. Save results
    """
    LLAMA_8B = InferenceModelSpec(
        model_label="Llama-3.1-8B",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpus_per_replica=1,
        itl_deadline_s=0.08,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    LLAMA_70B = InferenceModelSpec(
        model_label="Llama-3.1-70B",
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        gpus_per_replica=4,
        itl_deadline_s=0.10,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    LLAMA_405B = InferenceModelSpec(
        model_label="Llama-3.1-405B",
        model_id="meta-llama/Llama-3.1-405B-Instruct-FP8",
        gpus_per_replica=8,
        itl_deadline_s=0.12,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    QWEN_30B = InferenceModelSpec(
        model_label="Qwen3-30B-A3B",
        model_id="Qwen/Qwen3-30B-A3B-Thinking-2507",
        gpus_per_replica=2,
        itl_deadline_s=0.06,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    QWEN_235B = InferenceModelSpec(
        model_label="Qwen3-235B-A22B",
        model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
        gpus_per_replica=8,
        itl_deadline_s=0.14,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    _MODEL_SPECS = {s.model_label: s for s in (LLAMA_8B, LLAMA_70B, LLAMA_405B, QWEN_30B, QWEN_235B)}

    def deploy(label: str, num_replicas: int, initial_batch_size: int = 128) -> tuple[ModelDeployment, ReplicaSchedule]:
        return (
            ModelDeployment(spec=_MODEL_SPECS[label], initial_batch_size=initial_batch_size),
            ReplicaSchedule(initial=num_replicas),
        )

    # Build system config with experiment-specific overrides
    sys = SYSTEMS[system]()

    if system == "ieee34":
        # PV optimisation study: lower source voltage, uniform taps, no existing PV
        sys["source_pu"] = 1.05
        sys["initial_taps"] = TapPosition(
            regulators={
                "creg1a": tap(8),
                "creg1b": tap(8),
                "creg1c": tap(8),
                "creg2a": tap(8),
                "creg2b": tap(8),
                "creg2c": tap(8),
            }
        )

        # DC site descriptors (models referenced by label for GPU count estimation)
        all_deployments: tuple[tuple[ModelDeployment, ReplicaSchedule], ...] = (
            deploy("Llama-3.1-8B", 720),
            deploy("Llama-3.1-70B", 180),
            deploy("Llama-3.1-405B", 90),
            deploy("Qwen3-30B-A3B", 480),
            deploy("Qwen3-235B-A22B", 210),
        )
        dc_sites: dict[str, _DCSiteInfo] = {
            "upstream": _DCSiteInfo(
                bus="850",
                base_kw_per_phase=120.0,
                models=["Llama-3.1-8B", "Llama-3.1-70B", "Llama-3.1-405B"],
                total_gpu_capacity=520,
            ),
            "downstream": _DCSiteInfo(
                bus="834",
                base_kw_per_phase=80.0,
                models=["Qwen3-30B-A3B", "Qwen3-235B-A22B"],
                total_gpu_capacity=600,
            ),
        }

        time_varying_loads = [
            ("860", SyntheticLoad(peak_kw=50.0)),
            ("844", SyntheticLoad(peak_kw=80.0)),
            ("840", SyntheticLoad(peak_kw=15.0)),
            ("858", SyntheticLoad(peak_kw=30.0)),
            ("854", SyntheticLoad(peak_kw=10.0)),
            ("848", SyntheticLoad(peak_kw=60.0)),
        ]

    elif system == "ieee123":
        all_deployments = (
            deploy("Llama-3.1-8B", 120),
            deploy("Llama-3.1-70B", 30),
            deploy("Llama-3.1-405B", 35),
            deploy("Qwen3-30B-A3B", 80),
            deploy("Qwen3-235B-A22B", 55),
        )
        dc_sites = {
            "z1_sw": _DCSiteInfo(
                bus="8",
                base_kw_per_phase=310.0,
                models=["Llama-3.1-8B"],
                total_gpu_capacity=120,
            ),
            "z2_nw": _DCSiteInfo(
                bus="23",
                base_kw_per_phase=265.0,
                models=["Qwen3-30B-A3B"],
                total_gpu_capacity=160,
            ),
            "z3_se": _DCSiteInfo(
                bus="60",
                base_kw_per_phase=295.0,
                models=["Llama-3.1-70B", "Llama-3.1-405B"],
                total_gpu_capacity=400,
            ),
            "z4_ne": _DCSiteInfo(
                bus="105",
                base_kw_per_phase=325.0,
                models=["Qwen3-235B-A22B"],
                total_gpu_capacity=440,
            ),
        }
        time_varying_loads = []

    else:
        raise ValueError(f"PV+DC co-optimisation not configured for system '{system}'. Use ieee34 or ieee123.")

    bus_kv = sys["bus_kv"]
    source_pu = sys.get("source_pu", 1.05)
    exclude_buses_list = list(sys["exclude_buses"])
    exclude_buses_set = set(exclude_buses_list)
    zones = sys.get("regulator_zones") or sys.get("zones")

    # Build model GPU map for demand estimation
    model_gpu_map = {m.spec.model_label: s.initial * m.spec.gpus_per_replica for m, s in all_deployments}

    save_dir = Path(__file__).resolve().parent / "outputs" / system / "pv_dc_coopt"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Discover PV candidate buses
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 1: Discovering candidate buses")
    logger.info("=" * 70)

    all_candidate_buses = discover_candidate_buses(
        sys["dss_case_dir"],
        sys["dss_master_file"],
        bus_kv,
        exclude=exclude_buses_set,
    )

    # PV candidates: zone-filtered if zones are defined
    pv_candidate_buses = list(all_candidate_buses)
    if zones:
        zoned_buses = {b.lower() for buses_list in zones.values() for b in buses_list}
        before = len(pv_candidate_buses)
        pv_candidate_buses = [b for b in pv_candidate_buses if b.lower() in zoned_buses]
        logger.info("PV: Filtered to zoned buses: %d -> %d candidates", before, len(pv_candidate_buses))
    logger.info("PV candidate buses (%d): %s", len(pv_candidate_buses), pv_candidate_buses)

    # Step 2: Build DC candidate buses per zone
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: Building DC candidate buses per zone")
    logger.info("=" * 70)

    # DC candidate buses: union of all zone buses that are valid 3-phase buses
    all_cand_lower = {b.lower() for b in all_candidate_buses}
    dc_candidate_buses: list[str] = []
    dc_zone_indices: dict[str, list[int]] = {}

    if zones and not no_dc_zones:
        # Each DC site gets candidates from its zone
        zone_names = list(zones.keys())
        dc_site_ids = list(dc_sites.keys())

        # Collect all unique DC candidate buses across zones
        dc_bus_set: set[str] = set()
        for zone_buses in zones.values():
            for b in zone_buses:
                if b.lower() in all_cand_lower:
                    dc_bus_set.add(b)

        dc_candidate_buses = sorted(dc_bus_set)
        dc_cand_lower = [b.lower() for b in dc_candidate_buses]

        # Build zone indices: each site gets candidates from its assigned zone
        # If there are more sites than zones, cycle through zones
        for k, site_id in enumerate(dc_site_ids):
            zone_name = zone_names[k % len(zone_names)]
            zone_set = {b.lower() for b in zones[zone_name]}
            indices = [j for j, b in enumerate(dc_cand_lower) if b in zone_set]
            dc_zone_indices[site_id] = indices
            logger.info(
                "  DC site %s -> zone %s: %d candidate buses",
                site_id,
                zone_name,
                len(indices),
            )
    else:
        # No zones: all candidate buses are available for all DC sites
        dc_candidate_buses = list(all_candidate_buses)
        dc_site_ids = list(dc_sites.keys())
        all_indices = list(range(len(dc_candidate_buses)))
        for site_id in dc_site_ids:
            dc_zone_indices[site_id] = list(all_indices)

    logger.info("DC candidate buses (%d): %s", len(dc_candidate_buses), dc_candidate_buses)

    # Step 3: Parse taps and compute DC demand info
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: Parsing taps and DC demand")
    logger.info("=" * 70)

    # Parse initial taps from TapPosition object
    tap_pu: dict[str, float] = {}
    initial_tap_ints: dict[str, int] = {}
    regulator_names: list[str] = []
    initial_taps = sys.get("initial_taps")
    if initial_taps is not None:
        for name, val in initial_taps.regulators.items():
            tap_pu[name] = val
            initial_tap_ints[name] = round((val - 1.0) / TAP_STEP)
            regulator_names.append(name)

    # GPU power estimation
    TYPICAL_GPU_W = 300.0

    # Build load_configs from time_varying_loads
    load_configs: list[tuple[str, float]] = []
    load_buses: list[str] = []
    for bus, load in time_varying_loads:
        load_configs.append((bus, load._peak_kw))
        load_buses.append(bus)
    if load_configs:
        logger.info(
            "Load configs: %s",
            ", ".join(f"{b}={kw:.0f}kW" for b, kw in load_configs),
        )

    # Step 4: Generate scenarios (WITHOUT dc_sites_info)
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 4: Generating operating scenarios")
    logger.info("=" * 70)

    scenarios = generate_scenarios(
        load_configs=load_configs if load_configs else None,
        dc_sites_info=None,  # DC locations are variable -- no fixed DC in scenarios
    )
    total_weight = sum(sc.weight for sc in scenarios)
    logger.info("Total scenario weight: %.4f (should be 1.0)", total_weight)
    for sc in scenarios:
        logger.info(
            "  %s: %d steps, weight=%.2f (~%d days/yr), peak PV=%.2f, loads=%d buses",
            sc.name,
            len(sc.hours),
            sc.weight,
            int(sc.weight * 365),
            sc.pv_profile.max(),
            len(sc.load_kw_per_bus),
        )

    # (scenario profile plot deferred to after DC demand computation)

    # Step 5: Compute sensitivities on bare circuit (no DC)
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 5: Computing voltage sensitivities")
    logger.info("=" * 70)

    # 5a: PV sensitivities
    logger.info("  5a: PV sensitivities...")
    t0 = time.time()
    H_pv, v0, v_index = compute_pv_sensitivities(
        case_dir=sys["dss_case_dir"],
        master_file=sys["dss_master_file"],
        candidate_buses=pv_candidate_buses,
        exclude_buses=exclude_buses_set,
        source_pu=source_pu,
        initial_taps=tap_pu if tap_pu else None,
        bus_kv=bus_kv,
        dc_sites=None,  # bare circuit
    )
    logger.info(
        "  PV sensitivity: %.1f s, %d monitored x %d candidates", time.time() - t0, H_pv.shape[0], H_pv.shape[1]
    )
    logger.info("  Base voltage range: %.4f to %.4f pu", v0.min(), v0.max())

    plot_sensitivity_heatmap(H_pv, v_index, pv_candidate_buses, save_dir, system)
    # Rename the default output
    default_heatmap = save_dir / f"pv_sensitivity_{system}.png"
    pv_heatmap = save_dir / "sensitivity_heatmap_pv.png"
    if default_heatmap.exists():
        default_heatmap.replace(pv_heatmap)

    # 5b: DC sensitivities
    logger.info("  5b: DC sensitivities...")
    t0 = time.time()
    H_dc = compute_dc_sensitivities(
        case_dir=sys["dss_case_dir"],
        master_file=sys["dss_master_file"],
        dc_candidate_buses=dc_candidate_buses,
        v_index=v_index,
        source_pu=source_pu,
        initial_taps=tap_pu if tap_pu else None,
        bus_kv=bus_kv,
    )
    logger.info(
        "  DC sensitivity: %.1f s, %d monitored x %d candidates", time.time() - t0, H_dc.shape[0], H_dc.shape[1]
    )

    # Plot DC sensitivity heatmap
    plot_sensitivity_heatmap(H_dc, v_index, dc_candidate_buses, save_dir, system)
    default_heatmap = save_dir / f"pv_sensitivity_{system}.png"
    dc_heatmap = save_dir / "sensitivity_heatmap_dc.png"
    if default_heatmap.exists():
        default_heatmap.replace(dc_heatmap)

    # 5c: Tap sensitivities
    H_tap = None
    if regulator_names:
        logger.info("  5c: Tap sensitivities...")
        logger.info("    Regulators: %s", regulator_names)
        t0 = time.time()
        H_tap = compute_tap_sensitivities(
            case_dir=sys["dss_case_dir"],
            master_file=sys["dss_master_file"],
            regulator_names=regulator_names,
            v_index=v_index,
            source_pu=source_pu,
            initial_taps=tap_pu if tap_pu else None,
            bus_kv=bus_kv,
            dc_sites=None,
        )
        logger.info("  Tap sensitivity: %.1f s", time.time() - t0)

    # 5d: Load sensitivities
    load_v_shift = None
    if load_buses:
        logger.info("  5d: Load sensitivities...")
        t0 = time.time()
        H_load = compute_load_sensitivities(
            case_dir=sys["dss_case_dir"],
            master_file=sys["dss_master_file"],
            load_buses=load_buses,
            v_index=v_index,
            source_pu=source_pu,
            initial_taps=tap_pu if tap_pu else None,
            bus_kv=bus_kv,
            dc_sites=None,
        )
        logger.info("  Load sensitivity: %.1f s", time.time() - t0)
        load_v_shift = precompute_load_v_shift(H_load, load_buses, scenarios)
        logger.info("  Precomputed load voltage shifts for %d scenarios", len(load_v_shift))

    # Step 6: Compute DC demand profiles
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 6: Computing DC demand profiles")
    logger.info("=" * 70)

    dc_demand_kw, site_ids_ordered = compute_dc_demand_profiles(
        dc_sites,
        model_gpu_map,
        scenarios,
        typical_gpu_w=TYPICAL_GPU_W,
    )
    logger.info("DC demand shape: %s (sites x timesteps x scenarios)", dc_demand_kw.shape)
    for k, sid in enumerate(site_ids_ordered):
        logger.info(
            "  %s: mean=%.0f, min=%.0f, max=%.0f kW/ph",
            sid,
            dc_demand_kw[k].mean(),
            dc_demand_kw[k].min(),
            dc_demand_kw[k].max(),
        )

    plot_scenario_profiles(
        scenarios,
        save_dir,
        system,
        dc_demand_kw=dc_demand_kw,
        dc_site_ids=site_ids_ordered,
    )

    # Step 7: Solve MILP
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 7: Solving PV + DC co-optimisation MILP")
    logger.info("=" * 70)
    logger.info(
        "  n_pv=%d, s_max=%.0f kW, s_total_max=%s kW, c_inv=%.1f $/kW/yr, c_viol=%.0f, c_switch=%.0f",
        n_pv,
        s_max_kw,
        f"{s_total_max_kw:.0f}" if s_total_max_kw else "None",
        c_inv,
        c_viol,
        c_switch,
    )
    logger.info("  v_min=%.2f, v_max=%.2f, days_per_year=%.0f", v_min, v_max, days_per_year)

    result = solve_pv_dc_milp(
        pv_candidate_buses=pv_candidate_buses,
        dc_candidate_buses=dc_candidate_buses,
        dc_site_ids=site_ids_ordered,
        dc_zone_indices=dc_zone_indices,
        dc_demand_kw=dc_demand_kw,
        H_pv=H_pv,
        H_dc=H_dc,
        v0=v0,
        v_index=v_index,
        scenarios=scenarios,
        n_pv=n_pv,
        s_max_kw=s_max_kw,
        s_total_max_kw=s_total_max_kw,
        v_min=v_min,
        v_max=v_max,
        c_inv=c_inv,
        c_viol=c_viol,
        time_limit_s=time_limit_s,
        mip_gap=0.02,
        H_tap=H_tap,
        regulator_names=regulator_names if H_tap is not None else None,
        initial_tap_ints=initial_tap_ints if H_tap is not None else None,
        max_tap_delta=max_tap_delta,
        days_per_year=days_per_year,
        c_switch=c_switch,
        load_v_shift=load_v_shift,
        pv_zones=zones,
    )

    # Report results
    logger.info("")
    logger.info("=" * 70)
    logger.info("CO-OPT MILP RESULT")
    logger.info("=" * 70)
    logger.info("  Status: %s", result.status)
    logger.info("  Objective: $%.2f/year", result.objective)
    logger.info("  Investment cost: $%.2f/year", result.investment_cost)
    logger.info("  Operational savings: $%.2f/year", result.operational_savings)
    logger.info("  Violation penalty: $%.2f/year", result.violation_penalty)
    logger.info("  Switching cost: $%.2f/year", result.switching_cost)
    logger.info("  Curtailment: %.1f MWh/year", result.curtailment_mwh)
    logger.info("  Solve time: %.1f s", result.solve_time_s)
    logger.info("")
    logger.info("  PV Placements:")
    if result.pv_locations:
        for bus, cap in zip(result.pv_locations, result.pv_capacities_kw, strict=False):
            logger.info("    Bus %-10s  Capacity: %8.1f kW (%.1f kW/phase)", bus, cap, cap / 3)
    else:
        logger.info("    (none)")

    logger.info("")
    logger.info("  DC Assignments:")
    for site_id, bus in result.dc_assignments.items():
        logger.info("    Site %-10s -> Bus %s", site_id, bus)

    if result.tap_schedules:
        logger.info("")
        logger.info("  Tap Schedules (time-varying):")
        for sc_name, schedule in result.tap_schedules.items():
            unique_settings: dict[tuple, list[int]] = {}
            for hour, taps in sorted(schedule.items()):
                key = tuple(taps[r] for r in regulator_names)
                if key not in unique_settings:
                    unique_settings[key] = []
                unique_settings[key].append(hour)
            logger.info(
                "    Scenario: %s (%d distinct tap settings)",
                sc_name,
                len(unique_settings),
            )
            for taps_tuple, hours_list in unique_settings.items():
                hour_range = f"h{hours_list[0]}-{hours_list[-1]}" if len(hours_list) > 1 else f"h{hours_list[0]}"
                taps_str = ", ".join(f"{r}={t:+d}" for r, t in zip(regulator_names, taps_tuple, strict=False))
                logger.info("      %s: %s", hour_range, taps_str)

    # Save outputs
    logger.info("")
    logger.info("=" * 70)
    logger.info("Saving outputs")
    logger.info("=" * 70)

    # PV placements CSV
    pv_rows = []
    for bus in pv_candidate_buses:
        pv_rows.append(
            {
                "bus": bus,
                "selected": int(result.all_x.get(bus, 0) > 0.5),
                "capacity_kw": result.all_s.get(bus, 0.0),
            }
        )
    pv_csv_path = save_dir / f"pv_placements_{system}.csv"
    with open(pv_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pv_rows)
    logger.info("  PV placements: %s", pv_csv_path)

    # DC assignments CSV
    dc_rows = []
    for site_id in site_ids_ordered:
        assigned_bus = result.dc_assignments.get(site_id, "")
        # Determine zone name
        zone_name = ""
        if zones:
            zone_names = list(zones.keys())
            k = site_ids_ordered.index(site_id)
            zone_name = zone_names[k % len(zone_names)]
        # Peak demand
        peak_kw = float(dc_demand_kw[site_ids_ordered.index(site_id)].max())
        dc_rows.append(
            {
                "site_id": site_id,
                "assigned_bus": assigned_bus,
                "zone": zone_name,
                "peak_kw_per_phase": f"{peak_kw:.1f}",
            }
        )
    dc_csv_path = save_dir / f"dc_assignments_{system}.csv"
    with open(dc_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dc_rows)
    logger.info("  DC assignments: %s", dc_csv_path)

    # Tap schedule CSV
    if result.tap_schedules:
        tap_csv_path = save_dir / f"tap_schedule_{system}.csv"
        with open(tap_csv_path, "w", newline="") as f:
            fields = ["scenario", "hour"] + [f"{r}_tap" for r in regulator_names] + [f"{r}_pu" for r in regulator_names]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for sc_name, schedule in result.tap_schedules.items():
                for hour in sorted(schedule.keys()):
                    taps = schedule[hour]
                    row: dict = {"scenario": sc_name, "hour": hour}
                    for r in regulator_names:
                        row[f"{r}_tap"] = taps[r]
                        row[f"{r}_pu"] = f"{1.0 + taps[r] * TAP_STEP:.5f}"
                    writer.writerow(row)
        logger.info("  Tap schedule: %s", tap_csv_path)

    # Summary JSON
    summary = {
        "status": result.status,
        "objective_dollar_per_year": result.objective,
        "investment_cost_dollar_per_year": result.investment_cost,
        "operational_savings_dollar_per_year": result.operational_savings,
        "violation_penalty": result.violation_penalty,
        "switching_cost_dollar_per_year": result.switching_cost,
        "curtailment_mwh_per_year": result.curtailment_mwh,
        "solve_time_s": result.solve_time_s,
        "pv_locations": result.pv_locations,
        "pv_capacities_kw": result.pv_capacities_kw,
        "total_pv_kw": sum(result.pv_capacities_kw),
        "dc_assignments": result.dc_assignments,
        "tap_schedules": result.tap_schedules,
        "parameters": {
            "n_pv": n_pv,
            "s_max_kw": s_max_kw,
            "s_total_max_kw": s_total_max_kw,
            "c_inv": c_inv,
            "c_viol": c_viol,
            "c_switch": c_switch,
            "v_min": v_min,
            "v_max": v_max,
            "max_tap_delta": max_tap_delta,
            "days_per_year": days_per_year,
        },
    }
    summary_path = save_dir / f"milp_summary_{system}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("  Summary: %s", summary_path)

    # Plots
    plot_coopt_results(
        pv_candidate_buses,
        dc_candidate_buses,
        result,
        save_dir,
        system,
        dc_demand_kw=dc_demand_kw,
        dc_site_ids=site_ids_ordered,
    )

    # Step 8: Validate with full OpenDSS simulation
    if validate and (result.pv_locations or result.dc_assignments):
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 8: Validating with full OpenDSS simulation")
        logger.info("=" * 70)

        val_results = []
        for sc_idx, sc in enumerate(scenarios):
            val_schedule = None
            if result.tap_schedules and sc.name in result.tap_schedules:
                val_schedule = result.tap_schedules[sc.name]
                unique = len(set(tuple(sorted(taps.items())) for taps in val_schedule.values()))
                logger.info(
                    "  Validating %s with %d distinct tap settings across %d hours",
                    sc.name,
                    unique,
                    len(val_schedule),
                )

            vr = validate_coopt(
                case_dir=sys["dss_case_dir"],
                master_file=sys["dss_master_file"],
                pv_locations=result.pv_locations,
                pv_capacities_kw=result.pv_capacities_kw,
                dc_assignments=result.dc_assignments,
                dc_demand_kw=dc_demand_kw,
                dc_site_ids=site_ids_ordered,
                scenario=sc,
                sc_idx=sc_idx,
                v_index=v_index,
                source_pu=source_pu,
                initial_taps=tap_pu if tap_pu else None,
                bus_kv=bus_kv,
                tap_schedule=val_schedule,
                v_min=v_min,
                v_max=v_max,
            )
            val_results.append(vr)

        val_csv_path = save_dir / f"validation_{system}.csv"
        with open(val_csv_path, "w", newline="") as f:
            fields = list(val_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for vr in val_results:
                row = dict(vr)
                row["pv_locations"] = ";".join(row["pv_locations"])
                row["pv_capacities_kw"] = ";".join(f"{c:.1f}" for c in row["pv_capacities_kw"])
                writer.writerow(row)
        logger.info("  Validation CSV: %s", val_csv_path)

    logger.info("")
    logger.info("All outputs saved to: %s", save_dir)


# PV-only main


def _representative_taps(result: MILPResult, regulator_names: list[str]) -> dict[str, int]:
    """Extract the most common tap setting across all scenarios and hours.

    Used as the new linearization center for iterative re-linearization.
    """
    counters: dict[str, Counter] = {r: Counter() for r in regulator_names}
    if result.tap_schedules:
        for schedule in result.tap_schedules.values():
            for taps in schedule.values():
                for r in regulator_names:
                    counters[r][taps[r]] += 1

    return {r: counters[r].most_common(1)[0][0] if counters[r] else 0 for r in regulator_names}


def main(
    *,
    system: str = "ieee34",
    n_pv: int = 5,
    s_max_kw: float = 2000.0,
    s_total_max_kw: float | None = 2500.0,
    c_inv: float = 200.0,
    c_viol: float = 5_000.0,
    c_switch: float = 10.0,
    v_min: float = 0.95,
    v_max: float = 1.05,
    buses: list[str] | None = None,
    validate: bool = True,
    time_limit_s: float = 300.0,
    co_optimise_taps: bool = True,
    max_tap_delta: int = 8,
    days_per_year: float = 365.0,
    compare_ofo: bool = False,
) -> None:
    """Run PV expansion planning with optional tap co-optimisation.

    All experiment parameters (system constants, DC sites, time-varying loads)
    are defined inline per system.  No external JSON config required.

    Args:
        system: IEEE test feeder name (`ieee34` or `ieee123`).
        c_inv: Annualised investment cost ($/kW/year).  Typical utility-scale
            PV is ~$800-1200/kW installed; at 20-year lifetime this is
            ~$40-60/kW/year.  Use higher values to account for O&M, land,
            and interconnection costs.
        c_viol: Penalty per pu of voltage violation slack per bus-phase per
            hour.  Annualised as c_viol * 365 * sum(weight * slack).
        c_switch: Cost per tap switch per regulator per hour ($/switch).
            Annualised the same way.  Penalises frequent tap changes.
        co_optimise_taps: If True, include regulator tap positions as integer
            decision variables (requires initial_taps in config).
        days_per_year: For annualising scenario-based savings.
        compare_ofo: If True, run coordinator simulations comparing tap-only,
            OFO-only, and tap+OFO modes after MILP solve.
    """
    LLAMA_8B = InferenceModelSpec(
        model_label="Llama-3.1-8B",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpus_per_replica=1,
        itl_deadline_s=0.08,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    LLAMA_70B = InferenceModelSpec(
        model_label="Llama-3.1-70B",
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        gpus_per_replica=4,
        itl_deadline_s=0.10,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    LLAMA_405B = InferenceModelSpec(
        model_label="Llama-3.1-405B",
        model_id="meta-llama/Llama-3.1-405B-Instruct-FP8",
        gpus_per_replica=8,
        itl_deadline_s=0.12,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    QWEN_30B = InferenceModelSpec(
        model_label="Qwen3-30B-A3B",
        model_id="Qwen/Qwen3-30B-A3B-Thinking-2507",
        gpus_per_replica=2,
        itl_deadline_s=0.06,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    QWEN_235B = InferenceModelSpec(
        model_label="Qwen3-235B-A22B",
        model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
        gpus_per_replica=8,
        itl_deadline_s=0.14,
        feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
    )
    _MODEL_SPECS = {s.model_label: s for s in (LLAMA_8B, LLAMA_70B, LLAMA_405B, QWEN_30B, QWEN_235B)}

    def deploy(label: str, num_replicas: int, initial_batch_size: int = 128) -> tuple[ModelDeployment, ReplicaSchedule]:
        return (
            ModelDeployment(spec=_MODEL_SPECS[label], initial_batch_size=initial_batch_size),
            ReplicaSchedule(initial=num_replicas),
        )

    @dataclass
    class DCSite:
        bus: str
        base_kw_per_phase: float
        total_gpu_capacity: int
        models: tuple[tuple[ModelDeployment, ReplicaSchedule], ...] = ()
        seed: int = 0
        connection_type: str = "wye"

    # Build system config with PV-optimisation overrides
    sys = SYSTEMS[system]()

    # System-specific experiment parameters (DC sites, time-varying loads).
    # These mirror the JSON configs that were previously used but are now inline.
    if system == "ieee34":
        # PV optimisation study: lower source voltage (1.05 vs base 1.09),
        # uniform initial taps (+8 all), no existing PV, different load set.
        sys["source_pu"] = 1.05  # override for PV optimization study
        sys["initial_taps"] = TapPosition(
            regulators={
                "creg1a": tap(8),
                "creg1b": tap(8),
                "creg1c": tap(8),
                "creg2a": tap(8),
                "creg2b": tap(8),
                "creg2c": tap(8),
            }
        )

        upstream_models = (deploy("Llama-3.1-8B", 720), deploy("Llama-3.1-70B", 180), deploy("Llama-3.1-405B", 90))
        downstream_models = (deploy("Qwen3-30B-A3B", 480), deploy("Qwen3-235B-A22B", 210))

        dc_sites = {
            "upstream": DCSite(
                bus="850",
                base_kw_per_phase=120.0,
                models=upstream_models,
                seed=0,
                total_gpu_capacity=520,
            ),
            "downstream": DCSite(
                bus="834",
                base_kw_per_phase=80.0,
                models=downstream_models,
                seed=42,
                total_gpu_capacity=600,
            ),
        }

        time_varying_loads = [
            ("860", SyntheticLoad(peak_kw=50.0)),
            ("844", SyntheticLoad(peak_kw=80.0)),
            ("840", SyntheticLoad(peak_kw=15.0)),
            ("858", SyntheticLoad(peak_kw=30.0)),
            ("854", SyntheticLoad(peak_kw=10.0)),
            ("848", SyntheticLoad(peak_kw=60.0)),
        ]

    elif system == "ieee123":
        dc_sites = {
            "z1_sw": DCSite(
                bus="8",
                base_kw_per_phase=310.0,
                models=(deploy("Llama-3.1-8B", 120),),
                seed=0,
                total_gpu_capacity=120,
            ),
            "z2_nw": DCSite(
                bus="23",
                base_kw_per_phase=265.0,
                models=(deploy("Qwen3-30B-A3B", 80),),
                seed=17,
                total_gpu_capacity=160,
            ),
            "z3_se": DCSite(
                bus="60",
                base_kw_per_phase=295.0,
                models=(deploy("Llama-3.1-70B", 30), deploy("Llama-3.1-405B", 35)),
                seed=34,
                total_gpu_capacity=400,
            ),
            "z4_ne": DCSite(
                bus="105",
                base_kw_per_phase=325.0,
                models=(deploy("Qwen3-235B-A22B", 55),),
                seed=51,
                total_gpu_capacity=440,
            ),
        }
        time_varying_loads = []

    else:
        raise ValueError(f"PV optimisation not configured for system '{system}'. Use ieee34 or ieee123.")

    bus_kv = sys["bus_kv"]
    source_pu = sys.get("source_pu", 1.05)
    exclude_buses = set(sys["exclude_buses"])
    zones = sys.get("regulator_zones") or sys.get("zones")

    save_dir = Path(__file__).resolve().parent / "outputs" / system / "pv_expansion"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Parse initial taps
    tap_pu: dict[str, float] = {}
    initial_tap_ints: dict[str, int] = {}
    regulator_names: list[str] = []
    initial_taps = sys.get("initial_taps")
    if initial_taps is not None:
        for name, val in initial_taps.regulators.items():
            tap_pu[name] = val
            initial_tap_ints[name] = round((val - 1.0) / TAP_STEP)
            regulator_names.append(name)

    # Discover candidate buses
    if buses:
        candidate_buses = buses
    else:
        candidate_buses = discover_candidate_buses(
            sys["dss_case_dir"],
            sys["dss_master_file"],
            bus_kv,
            exclude=exclude_buses,
        )

    # Filter to only zoned buses (exclude inter-zone connectors)
    if zones:
        zoned_buses = {b.lower() for buses_list in zones.values() for b in buses_list}
        before = len(candidate_buses)
        candidate_buses = [b for b in candidate_buses if b.lower() in zoned_buses]
        logger.info("Filtered to zoned buses: %d -> %d candidates", before, len(candidate_buses))

    logger.info("Candidate buses (%d): %s", len(candidate_buses), candidate_buses)

    # Build dc_sites dict for sensitivity base case.
    # Estimate realistic power per phase = base_kw + GPU workload power.
    TYPICAL_GPU_W = 300.0  # H100 inference average

    dc_sites_dict: dict[str, dict] = {}
    for sid, site in dc_sites.items():
        total_gpus = site.total_gpu_capacity or sum(m.spec.gpus_per_replica * s.initial for m, s in site.models)
        gpu_kw_per_phase = total_gpus * TYPICAL_GPU_W / 1000.0 / 3.0
        peak_kw_per_phase = site.base_kw_per_phase + gpu_kw_per_phase
        dc_sites_dict[sid] = {
            "bus": site.bus,
            "base_kw_per_phase": peak_kw_per_phase,
            "peak_kw_per_phase": peak_kw_per_phase,
        }
        logger.info(
            "DC site %s@%s: %d GPUs, peak/base %.0f kW/ph (base %.0f + GPU %.0f)",
            sid,
            site.bus,
            total_gpus,
            peak_kw_per_phase,
            site.base_kw_per_phase,
            gpu_kw_per_phase,
        )

    # Build load_configs from time_varying_loads
    load_configs: list[tuple[str, float]] = []
    load_buses: list[str] = []
    for bus, load in time_varying_loads:
        load_configs.append((bus, load._peak_kw))
        load_buses.append(bus)
    if load_configs:
        logger.info("Load configs: %s", ", ".join(f"{b}={kw:.0f}kW" for b, kw in load_configs))

    # DC sites info for time-varying inference demand in scenarios
    dc_sites_info: list[tuple[str, float, float, str]] = []
    for sid, sd in dc_sites_dict.items():
        dc_bus = sd["bus"]
        dc_peak = sd["peak_kw_per_phase"]
        dc_base = sd["base_kw_per_phase"]
        dc_sites_info.append((dc_bus, dc_peak, dc_base, sid))
        if dc_bus not in load_buses:
            load_buses.append(dc_bus)
    if dc_sites_info:
        logger.info(
            "DC sites for time-varying demand: %s",
            ", ".join(f"{b}: peak={pk:.0f}, base={bs:.0f} kW/ph ({sid})" for b, pk, bs, sid in dc_sites_info),
        )

    # Step 2: Generate scenarios (before loop -- invariant)
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: Generating operating scenarios")
    logger.info("=" * 70)
    scenarios = generate_scenarios(
        load_configs=load_configs if load_configs else None,
        dc_sites_info=dc_sites_info if dc_sites_info else None,
    )
    total_weight = sum(sc.weight for sc in scenarios)
    logger.info("Total scenario weight: %.4f (should be 1.0)", total_weight)
    for sc in scenarios:
        logger.info(
            "  %s: %d steps, weight=%.2f (~%d days/yr), peak PV=%.2f, loads=%d buses",
            sc.name,
            len(sc.hours),
            sc.weight,
            int(sc.weight * 365),
            sc.pv_profile.max(),
            len(sc.load_kw_per_bus),
        )

    plot_scenario_profiles(scenarios, save_dir, system, dc_sites_info=dc_sites_info or None)

    current_tap_pu = dict(tap_pu)  # pu values for sensitivity computation
    current_tap_ints = dict(initial_tap_ints)  # integer positions for MILP center

    # Step 1: Compute PV sensitivities
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 1: Computing PV voltage sensitivities")
    logger.info("=" * 70)
    t0 = time.time()

    H_pv, v0, v_index = compute_pv_sensitivities(
        case_dir=sys["dss_case_dir"],
        master_file=sys["dss_master_file"],
        candidate_buses=candidate_buses,
        exclude_buses=exclude_buses,
        source_pu=source_pu,
        initial_taps=current_tap_pu if current_tap_pu else None,
        bus_kv=bus_kv,
        dc_sites=dc_sites_dict,
    )
    logger.info("PV sensitivity computation: %.1f s", time.time() - t0)
    logger.info("Base voltage range: %.4f to %.4f pu", v0.min(), v0.max())

    for idx, (bus, phase) in enumerate(v_index):
        if v0[idx] > v_max - 0.01 or v0[idx] < v_min + 0.01:
            logger.info("  WARNING: %s.%s base voltage = %.4f pu (near limit)", bus, phase, v0[idx])

    plot_sensitivity_heatmap(H_pv, v_index, candidate_buses, save_dir, system)

    # Step 1b: Compute tap sensitivities
    H_tap = None
    if co_optimise_taps and regulator_names:
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 1b: Computing tap voltage sensitivities")
        logger.info("=" * 70)
        logger.info("  Regulators: %s", regulator_names)
        logger.info("  Linearization taps: %s", {k: f"{v:+d}" for k, v in current_tap_ints.items()})
        t0 = time.time()
        H_tap = compute_tap_sensitivities(
            case_dir=sys["dss_case_dir"],
            master_file=sys["dss_master_file"],
            regulator_names=regulator_names,
            v_index=v_index,
            source_pu=source_pu,
            initial_taps=current_tap_pu if current_tap_pu else None,
            bus_kv=bus_kv,
            dc_sites=dc_sites_dict,
        )
        logger.info("Tap sensitivity computation: %.1f s", time.time() - t0)
    elif co_optimise_taps:
        logger.info("No initial_taps -- skipping tap co-optimisation")

    # Step 1c: Compute load sensitivities
    load_v_shift = None
    if load_buses:
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 1c: Computing load voltage sensitivities")
        logger.info("=" * 70)
        logger.info("  Load buses: %s", load_buses)
        t0 = time.time()
        H_load = compute_load_sensitivities(
            case_dir=sys["dss_case_dir"],
            master_file=sys["dss_master_file"],
            load_buses=load_buses,
            v_index=v_index,
            source_pu=source_pu,
            initial_taps=current_tap_pu if current_tap_pu else None,
            bus_kv=bus_kv,
            dc_sites=dc_sites_dict,
        )
        logger.info("Load sensitivity computation: %.1f s", time.time() - t0)
        load_v_shift = precompute_load_v_shift(H_load, load_buses, scenarios)
        logger.info("Precomputed load voltage shifts for %d scenarios", len(load_v_shift))

    # Step 3: Solve MILP
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: Solving PV + tap co-optimisation MILP")
    logger.info("=" * 70)
    logger.info(
        "  n_pv=%d, s_max=%.0f kW, s_total_max=%s kW, c_inv=%.1f $/kW/yr, c_viol=%.0f",
        n_pv,
        s_max_kw,
        f"{s_total_max_kw:.0f}" if s_total_max_kw else "None",
        c_inv,
        c_viol,
    )
    logger.info("  v_min=%.2f, v_max=%.2f, days_per_year=%.0f", v_min, v_max, days_per_year)

    result = solve_pv_milp(
        candidate_buses,
        H_pv,
        v0,
        v_index,
        scenarios,
        n_pv=n_pv,
        s_max_kw=s_max_kw,
        s_total_max_kw=s_total_max_kw,
        v_min=v_min,
        v_max=v_max,
        c_inv=c_inv,
        c_viol=c_viol,
        time_limit_s=time_limit_s,
        H_tap=H_tap,
        regulator_names=regulator_names if H_tap is not None else None,
        initial_tap_ints=current_tap_ints if H_tap is not None else None,
        max_tap_delta=max_tap_delta,
        days_per_year=days_per_year,
        c_switch=c_switch,
        load_v_shift=load_v_shift,
        zones=zones,
    )

    logger.info(
        "  Objective: $%.2f/year  (inv=$%.0f, savings=$%.0f, viol=$%.0f, switch=$%.0f, curtail=%.0f MWh/yr)",
        result.objective,
        result.investment_cost,
        result.operational_savings,
        result.violation_penalty,
        result.switching_cost,
        result.curtailment_mwh,
    )
    logger.info(
        "  PV: %s",
        ", ".join(f"{b}={c:.0f}kW" for b, c in zip(result.pv_locations, result.pv_capacities_kw, strict=False))
        if result.pv_locations
        else "(none)",
    )

    # Report results
    logger.info("")
    logger.info("=" * 70)
    logger.info("MILP RESULT")
    logger.info("=" * 70)
    logger.info("  Status: %s", result.status)
    logger.info("  Objective: $%.2f/year", result.objective)
    logger.info("  Investment cost: $%.2f/year", result.investment_cost)
    logger.info("  Operational savings: $%.2f/year", result.operational_savings)
    logger.info("  Violation penalty: $%.2f/year", result.violation_penalty)
    logger.info("  Switching cost: $%.2f/year", result.switching_cost)
    logger.info("  Curtailment: %.1f MWh/year", result.curtailment_mwh)
    logger.info("  Solve time: %.1f s", result.solve_time_s)
    logger.info("")
    logger.info("  PV Placements:")
    if result.pv_locations:
        for bus, cap in zip(result.pv_locations, result.pv_capacities_kw, strict=False):
            logger.info("    Bus %-10s  Capacity: %8.1f kW (%.1f kW/phase)", bus, cap, cap / 3)
    else:
        logger.info("    (none)")

    if result.tap_schedules:
        logger.info("")
        logger.info("  Tap Schedules (time-varying):")
        for sc_name, schedule in result.tap_schedules.items():
            # Show unique tap settings across the day
            unique_settings = {}
            for hour, taps in sorted(schedule.items()):
                key = tuple(taps[r] for r in regulator_names)
                if key not in unique_settings:
                    unique_settings[key] = []
                unique_settings[key].append(hour)
            logger.info("    Scenario: %s (%d distinct tap settings)", sc_name, len(unique_settings))
            for taps_tuple, hours in unique_settings.items():
                hour_range = f"h{hours[0]}-{hours[-1]}" if len(hours) > 1 else f"h{hours[0]}"
                taps_str = ", ".join(f"{r}={t:+d}" for r, t in zip(regulator_names, taps_tuple, strict=False))
                logger.info("      %s: %s", hour_range, taps_str)

    # Save result CSV
    rows = []
    for bus in candidate_buses:
        rows.append(
            {
                "bus": bus,
                "selected": int(result.all_x.get(bus, 0) > 0.5),
                "capacity_kw": result.all_s.get(bus, 0.0),
            }
        )
    with open(save_dir / f"milp_result_{system}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Save tap schedule CSV (one row per scenario x hour x regulator)
    if result.tap_schedules:
        with open(save_dir / f"tap_schedule_{system}.csv", "w", newline="") as f:
            fields = ["scenario", "hour"] + [f"{r}_tap" for r in regulator_names] + [f"{r}_pu" for r in regulator_names]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for sc_name, schedule in result.tap_schedules.items():
                for hour in sorted(schedule.keys()):
                    taps = schedule[hour]
                    row = {"scenario": sc_name, "hour": hour}
                    for r in regulator_names:
                        row[f"{r}_tap"] = taps[r]
                        row[f"{r}_pu"] = f"{1.0 + taps[r] * TAP_STEP:.5f}"
                    writer.writerow(row)

    # Save summary JSON

    summary = {
        "status": result.status,
        "objective_dollar_per_year": result.objective,
        "investment_cost_dollar_per_year": result.investment_cost,
        "operational_savings_dollar_per_year": result.operational_savings,
        "violation_penalty": result.violation_penalty,
        "switching_cost_dollar_per_year": result.switching_cost,
        "curtailment_mwh_per_year": result.curtailment_mwh,
        "pv_locations": result.pv_locations,
        "pv_capacities_kw": result.pv_capacities_kw,
        "total_pv_kw": sum(result.pv_capacities_kw),
        "tap_schedules": result.tap_schedules,
        "parameters": {
            "system": system,
            "n_pv": n_pv,
            "s_max_kw": s_max_kw,
            "s_total_max_kw": s_total_max_kw,
            "c_inv": c_inv,
            "c_viol": c_viol,
            "c_switch": c_switch,
            "v_min": v_min,
            "v_max": v_max,
            "days_per_year": days_per_year,
            "co_optimise_taps": co_optimise_taps,
            "max_tap_delta": max_tap_delta,
        },
    }
    with open(save_dir / f"summary_{system}.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_results(candidate_buses, result, save_dir, system)

    # Step 4: Validate with full OpenDSS simulation
    if validate and (result.pv_locations or result.tap_schedules):
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 4: Validating with full OpenDSS simulation")
        logger.info("=" * 70)

        val_results = []
        for sc in scenarios:
            val_schedule = None
            if result.tap_schedules and sc.name in result.tap_schedules:
                val_schedule = result.tap_schedules[sc.name]
                unique = len(set(tuple(sorted(taps.items())) for taps in val_schedule.values()))
                logger.info(
                    "  Validating %s with %d distinct tap settings across %d hours", sc.name, unique, len(val_schedule)
                )

            vr = validate_with_opendss(
                case_dir=sys["dss_case_dir"],
                master_file=sys["dss_master_file"],
                pv_locations=result.pv_locations,
                pv_capacities_kw=result.pv_capacities_kw,
                scenario=sc,
                v_index=v_index,
                source_pu=source_pu,
                initial_taps=tap_pu if tap_pu else None,
                bus_kv=bus_kv,
                dc_sites=dc_sites_dict,
                tap_schedule=val_schedule,
                v_min=v_min,
                v_max=v_max,
            )
            val_results.append(vr)

        with open(save_dir / f"validation_{system}.csv", "w", newline="") as f:
            fields = list(val_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for vr in val_results:
                row = dict(vr)
                row["pv_locations"] = ";".join(row["pv_locations"])
                row["pv_capacities_kw"] = ";".join(f"{c:.1f}" for c in row["pv_capacities_kw"])
                writer.writerow(row)

    # Step 5: OFO comparison
    if compare_ofo and result.pv_locations:
        compare_with_ofo(
            sys=sys,
            dc_sites=dc_sites,
            pv_locations=result.pv_locations,
            pv_capacities_kw=result.pv_capacities_kw,
            tap_schedule_by_scenario=result.tap_schedules,
            save_dir=save_dir,
            system=system,
        )

    logger.info("")
    logger.info("All outputs saved to: %s", save_dir)


if __name__ == "__main__":
    from typing import Literal

    import tyro

    @dataclass
    class Args:
        """Command-line arguments.

        Attributes:
            mode: Optimisation mode: pv-only (default) or pv-and-dc
                (joint PV + DC siting).
            system: System name (ieee34 or ieee123).
            n_pv: Max number of PV sites to place.
            s_max_kw: Maximum PV capacity per site (kW, 3-phase total).
            s_total_max_kw: Total PV capacity budget across all sites (kW).
                None = no limit.
            c_inv: Annualised investment cost ($/kW/year).
            c_viol: Voltage violation penalty per pu slack per bus-phase per hour.
            c_switch: Tap switching cost per switch per regulator per hour ($/switch).
            v_min: Minimum voltage limit (pu).
            v_max: Maximum voltage limit (pu).
            buses: Comma-separated candidate buses (overrides auto-discovery).
                pv-only mode only.
            no_validate: Skip full OpenDSS validation.
            no_taps: Skip tap co-optimisation. pv-only mode only.
            max_tap_delta: Max tap change from initial (+-steps).
            time_limit: Solver time limit (seconds).
            days_per_year: Days per year for annualising savings.
            compare_ofo: Run OFO comparison after MILP solve. pv-only mode only.
            no_dc_zones: Allow DC sites at any bus (ignore zone constraints).
                pv-and-dc mode only.
            log_level: Logging verbosity.
        """

        mode: Literal["pv-only", "pv-and-dc"] = "pv-only"
        system: str = "ieee34"
        n_pv: int = 5
        s_max_kw: float = 2000.0
        s_total_max_kw: float | None = 2500.0
        c_inv: float = 200.0
        c_viol: float = 5_000.0
        c_switch: float = 10.0
        v_min: float = 0.95
        v_max: float = 1.05
        buses: str | None = None
        no_validate: bool = False
        no_taps: bool = False
        max_tap_delta: int = 8
        time_limit: float = 300.0
        days_per_year: float = 365.0
        compare_ofo: bool = False
        no_dc_zones: bool = False
        log_level: str = "INFO"

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("gurobipy").setLevel(logging.WARNING)
    logging.getLogger("openg2g.coordinator").setLevel(logging.WARNING)
    logging.getLogger("openg2g.datacenter").setLevel(logging.WARNING)
    logging.getLogger("openg2g.grid").setLevel(logging.WARNING)

    if args.mode == "pv-and-dc":
        main_pv_and_dc(
            system=args.system,
            n_pv=args.n_pv,
            s_max_kw=args.s_max_kw,
            s_total_max_kw=args.s_total_max_kw,
            c_inv=args.c_inv,
            c_viol=args.c_viol,
            c_switch=args.c_switch,
            max_tap_delta=args.max_tap_delta,
            time_limit_s=args.time_limit,
            validate=not args.no_validate,
            v_min=args.v_min,
            v_max=args.v_max,
            days_per_year=args.days_per_year,
            no_dc_zones=args.no_dc_zones,
        )
    else:
        bus_list = [b.strip() for b in args.buses.split(",")] if args.buses else None

        main(
            system=args.system,
            n_pv=args.n_pv,
            s_max_kw=args.s_max_kw,
            s_total_max_kw=args.s_total_max_kw,
            c_inv=args.c_inv,
            c_viol=args.c_viol,
            c_switch=args.c_switch,
            v_min=args.v_min,
            v_max=args.v_max,
            buses=bus_list,
            validate=not args.no_validate,
            time_limit_s=args.time_limit,
            co_optimise_taps=not args.no_taps,
            max_tap_delta=args.max_tap_delta,
            days_per_year=args.days_per_year,
            compare_ofo=args.compare_ofo,
        )
