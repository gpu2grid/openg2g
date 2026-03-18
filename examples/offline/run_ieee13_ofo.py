"""IEEE 13-bus OFO simulation with multi-DC, PV, and external load support.

Components: OfflineDatacenter(s) + ScenarioOpenDSSGrid + OFOBatchSizeController(s) + Coordinator.

Supports:
  - Multiple datacenter sites at different buses (via dc_sites config)
  - Per-site OFO controllers with independent batch-size optimization
  - Rooftop PV systems at arbitrary buses (via pv_systems config)
  - Time-varying external loads at arbitrary buses (via time_varying_loads config)
  - PV/load profiles from CSV files or auto-generated sinusoidal curves
  - Tap schedule support: if config has tap_schedule entries, runs 4 cases
    (baseline_no-tap, ofo, baseline_tap-change, ofo_tap-change);
    otherwise runs 2 cases (baseline_no-tap, ofo).

Usage:
    python examples/offline/run_ieee13_ofo.py --config examples/offline/config_ieee13.json
    python examples/offline/run_ieee13_ofo.py --config examples/offline/config.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, model_validator

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
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
    InferenceRamp,
    PowerAugmentationConfig,
    TrainingRun,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData, MLEnergySource
from openg2g.datacenter.workloads.training import TrainingTrace, TrainingTraceParams
from openg2g.events import EventEmitter
from openg2g.grid.base import GridState
from openg2g.grid.config import DCLoadSpec, TapPosition, TapSchedule
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import VoltageStats, compute_allbus_voltage_stats

from plot_all_figures import (
    extract_per_model_timeseries,
    plot_allbus_voltages_per_phase,
    plot_model_timeseries_4panel,
)

logger = logging.getLogger("run_ieee13_ofo")


# ── Profile generation helpers ──────────────────────────────────────────────


def _smooth_bump(t: float, t_center: float, half_width: float) -> float:
    """Smooth bump: 1 at center, 0 outside +/-half_width (quartic)."""
    dt = abs(t - t_center)
    if dt >= half_width:
        return 0.0
    x = dt / half_width
    return (1 - x * x) ** 2


def pv_profile_kw(t: float, peak_kw: float, t_total: float = 3600.0) -> float:
    """Solar PV output with irregular cloud events (kW per phase).

    Starts high (~80%), dips with two cloud events at ~10 min and ~35 min,
    trends downward toward end. Small 5-min and 1.5-min fluctuations.
    """
    T = t_total
    trend = 0.85 - 0.30 * (t / T)
    cloud = 1.0
    cloud -= 0.55 * _smooth_bump(t, 0.17 * T, 0.033 * T)
    cloud -= 0.40 * _smooth_bump(t, 0.58 * T, 0.050 * T)
    fluct = 1.0 + 0.10 * math.sin(2 * math.pi * t / 300.0 + 0.3)
    fluct += 0.05 * math.sin(2 * math.pi * t / 90.0 + 1.7)
    return max(0.0, peak_kw * trend * max(cloud, 0.05) * fluct)


def load_profile_kw(t: float, peak_kw: float, t_total: float = 3600.0) -> float:
    """Time-varying load with three irregular pulses (kW per phase).

    Low baseline (~10%) with three distinct demand pulses at ~12 min,
    ~30 min, and ~48 min of increasing intensity. Simulates irregular
    industrial or EV charging activity.
    """
    T = t_total
    base = 0.10
    base += 0.45 * _smooth_bump(t, 0.20 * T, 0.08 * T)
    base += 0.65 * _smooth_bump(t, 0.50 * T, 0.10 * T)
    base += 0.90 * _smooth_bump(t, 0.80 * T, 0.08 * T)
    fluct = 1.0 + 0.06 * math.sin(2 * math.pi * t / 130.0)
    fluct += 0.04 * math.sin(2 * math.pi * t / 55.0 + 0.8)
    return max(0.0, peak_kw * base * fluct)


def load_csv_profile(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a (time_s, kw) profile from CSV. Returns (time_s, kw) arrays."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def eval_profile(
    t: float,
    *,
    peak_kw: float,
    csv_data: tuple[np.ndarray, np.ndarray] | None,
    profile_fn,
    t_total: float,
) -> float:
    """Evaluate a profile at time t: CSV interpolation or generated curve."""
    if csv_data is not None:
        return float(np.interp(t, csv_data[0], csv_data[1]))
    return profile_fn(t, peak_kw, t_total)


def plot_profiles(
    pv_specs: list,
    load_specs: list,
    t_total: float,
    save_path: Path,
) -> None:
    """Plot generated PV and load profiles and save to file."""
    import matplotlib.pyplot as plt

    t = np.linspace(0, t_total, 1000)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=150, sharex=True)

    ax = axes[0]
    for i, spec in enumerate(pv_specs):
        y = np.array([pv_profile_kw(ti, spec.peak_kw, t_total) for ti in t])
        ax.plot(t / 60, y, lw=1.5, label=f"PV @ {spec.bus} ({spec.peak_kw:.0f} kW peak)")
    ax.set_ylabel("PV Output (kW/phase)")
    ax.set_title("PV Generation Profiles")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, spec in enumerate(load_specs):
        y = np.array([load_profile_kw(ti, spec.peak_kw, t_total) for ti in t])
        ax.plot(t / 60, y, lw=1.5, label=f"Load @ {spec.bus} ({spec.peak_kw:.0f} kW peak)")
    ax.set_ylabel("Load (kW/phase)")
    ax.set_xlabel("Time (min)")
    ax.set_title("Time-Varying Load Profiles")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ── Config models ───────────────────────────────────────────────────────────


class InferenceRampConfig(BaseModel):
    """Single inference ramp event configuration."""

    target: float = 0.2
    t_start: float = 2500.0
    t_end: float = 3000.0
    model: str | None = None


class DCSiteConfig(BaseModel):
    """Datacenter site at a specific grid bus."""

    bus: str
    bus_kv: float = 4.16
    base_kw_per_phase: float = 500.0
    models: list[str] | None = None
    """Model labels served at this site. None = all models."""
    connection_type: Literal["wye", "delta"] = "wye"
    seed: int = 0
    total_gpu_capacity: int | None = None
    inference_ramps: list[InferenceRampConfig] = []


class PVSystemConfig(BaseModel):
    """Rooftop PV system at a grid bus."""

    bus: str
    bus_kv: float = 4.16
    peak_kw: float = 1000.0
    csv_path: Path | None = None
    """CSV with (time_s, kw) columns. If None, uses generated bell curve."""
    power_factor: float = 1.0


class TimeVaryingLoadConfig(BaseModel):
    """Time-varying external load at a grid bus."""

    bus: str
    bus_kv: float = 4.16
    peak_kw: float = 500.0
    csv_path: Path | None = None
    """CSV with (time_s, kw) columns. If None, uses generated bell curve."""
    power_factor: float = 0.96


TAP_STEP = 0.00625


def _parse_tap(val: float | str | None) -> float | None:
    """Parse a tap value: float passthrough, or string like '+14' / '-3'."""
    if val is None:
        return None
    if isinstance(val, str):
        return 1.0 + int(val) * TAP_STEP
    return float(val)


def _taps_dict_to_position(d: dict[str, float | str] | None) -> TapPosition | None:
    """Convert a dict of regulator name -> tap value to a TapPosition.

    Tap values can be:
      - A float (e.g. 1.0875 = direct tap ratio)
      - A string like "+14" meaning 1.0 + 14 * 0.00625 = 1.0875
    """
    if not d:
        return None
    return TapPosition(regulators={k: _parse_tap(v) for k, v in d.items()})


class OFOParams(BaseModel):
    """OFO controller tuning parameters (exposed in config)."""

    primal_step_size: float = 0.1
    w_throughput: float = 1e-3
    w_switch: float = 1.0
    voltage_gradient_scale: float = 1e6
    voltage_dual_step_size: float = 1.0
    latency_dual_step_size: float = 1.0
    sensitivity_update_interval: int = 3600
    sensitivity_perturbation_kw: float = 100.0


class TrainingConfig(BaseModel):
    """Training run overlay configuration."""

    dc_site: str | None = None
    """DC site ID this training job runs on. ``None`` uses the first site."""
    n_gpus: int = 2400
    target_peak_W_per_gpu: float = 400.0
    t_start: float = 1000.0
    t_end: float = 2000.0


class SimulationParams(BaseModel):
    """Simulation timing and voltage parameters."""

    total_duration_s: int = 3600
    dt_dc: str = "1/10"
    dt_grid: str = "1/10"
    dt_ctrl: str = "1"
    v_min: float = 0.95
    v_max: float = 1.05


class IEEE13Config(BaseModel):
    """Full configuration for IEEE 13-bus OFO simulations."""

    models: list[InferenceModelSpec]
    data_sources: list[MLEnergySource]
    training_trace_params: TrainingTraceParams = TrainingTraceParams()
    data_dir: Path | None = None
    ieee_case_dir: Path
    dss_master_file: str = "IEEE13Nodeckt.dss"
    mlenergy_data_dir: Path | None = None

    source_pu: float = 1.0

    dc_sites: dict[str, DCSiteConfig] | None = None
    """Datacenter sites keyed by site ID. If None, uses single default site."""
    pv_systems: list[PVSystemConfig] = []
    time_varying_loads: list[TimeVaryingLoadConfig] = []
    initial_taps: dict[str, float | str] | None = None
    exclude_buses: list[str] = []
    tap_schedule: list[dict] | None = None

    ofo: OFOParams = OFOParams()
    training: TrainingConfig | None = None
    inference_ramp: InferenceRampConfig | None = None
    simulation: SimulationParams = SimulationParams()
    power_augmentation: PowerAugmentationConfig = PowerAugmentationConfig(
        amplitude_scale_range=(0.98, 1.02), noise_fraction=0.005,
    )

    @model_validator(mode="after")
    def _default_dc_site(self) -> IEEE13Config:
        if self.dc_sites is None:
            self.dc_sites = {
                "default": DCSiteConfig(bus="671", bus_kv=4.16, base_kw_per_phase=500.0)
            }
        return self

    @property
    def data_hash(self) -> str:
        blob = json.dumps(
            (
                sorted([s.model_dump(mode="json") for s in self.data_sources], key=lambda s: s["model_label"]),
                self.training_trace_params.model_dump(mode="json"),
            ),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


# ── Scenario Grid ───────────────────────────────────────────────────────────


class ScenarioOpenDSSGrid(OpenDSSGrid):
    """OpenDSSGrid with PV systems and external loads at arbitrary buses."""

    def __init__(
        self,
        *,
        pv_systems: list[PVSystemConfig] | None = None,
        time_varying_loads: list[TimeVaryingLoadConfig] | None = None,
        t_total: float = 3600.0,
        source_pu: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pv_specs = list(pv_systems or [])
        self._load_specs = list(time_varying_loads or [])
        self._t_total = t_total
        self._source_pu = source_pu

        # Pre-load CSV profiles
        self._pv_csv: list[tuple[np.ndarray, np.ndarray] | None] = []
        for spec in self._pv_specs:
            if spec.csv_path is not None:
                self._pv_csv.append(load_csv_profile(spec.csv_path))
            else:
                self._pv_csv.append(None)

        self._load_csv: list[tuple[np.ndarray, np.ndarray] | None] = []
        for spec in self._load_specs:
            if spec.csv_path is not None:
                self._load_csv.append(load_csv_profile(spec.csv_path))
            else:
                self._load_csv.append(None)

        # DSS load names
        self._pv_load_names: list[tuple[str, str, str]] = [
            (f"PV_{i}_A", f"PV_{i}_B", f"PV_{i}_C") for i in range(len(self._pv_specs))
        ]
        self._ext_load_names: list[tuple[str, str, str]] = [
            (f"ExtLoad_{i}_A", f"ExtLoad_{i}_B", f"ExtLoad_{i}_C") for i in range(len(self._load_specs))
        ]

    def _init_dss(self) -> None:
        super()._init_dss()
        from openg2g.grid.opendss import dss

        if self._source_pu is not None and self._source_pu != 1.0:
            dss.Text.Command(f"Edit Vsource.source pu={self._source_pu}")

        for i, spec in enumerate(self._pv_specs):
            kv_ln = spec.bus_kv / math.sqrt(3.0)
            for ph, name in zip((1, 2, 3), self._pv_load_names[i]):
                dss.Text.Command(
                    f"New Load.{name} bus1={spec.bus}.{ph} phases=1 "
                    f"conn=wye kV={kv_ln:.6f} kW=0 kvar=0 model=1"
                )

        for i, spec in enumerate(self._load_specs):
            kv_ln = spec.bus_kv / math.sqrt(3.0)
            for ph, name in zip((1, 2, 3), self._ext_load_names[i]):
                dss.Text.Command(
                    f"New Load.{name} bus1={spec.bus}.{ph} phases=1 "
                    f"conn=wye kV={kv_ln:.6f} kW=0 kvar=0 model=1"
                )

    def step(
        self,
        clock: SimulationClock,
        power_samples_w: dict[str, list[ThreePhase]],
        events: EventEmitter,
    ) -> GridState:
        from openg2g.grid.opendss import dss

        for i, spec in enumerate(self._pv_specs):
            kw = eval_profile(
                clock.time_s, peak_kw=spec.peak_kw, csv_data=self._pv_csv[i],
                profile_fn=pv_profile_kw, t_total=self._t_total,
            )
            pf = max(min(spec.power_factor, 0.999999), 1e-6)
            kvar = kw * math.tan(math.acos(pf))
            for name in self._pv_load_names[i]:
                dss.Loads.Name(name)
                dss.Loads.kW(-kw)
                dss.Loads.kvar(-kvar)

        for i, spec in enumerate(self._load_specs):
            kw = eval_profile(
                clock.time_s, peak_kw=spec.peak_kw, csv_data=self._load_csv[i],
                profile_fn=load_profile_kw, t_total=self._t_total,
            )
            pf = max(min(spec.power_factor, 0.999999), 1e-6)
            kvar = kw * math.tan(math.acos(pf))
            for name in self._ext_load_names[i]:
                dss.Loads.Name(name)
                dss.Loads.kW(kw)
                dss.Loads.kvar(kvar)

        return super().step(clock, power_samples_w, events)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _parse_fraction(s: str) -> Fraction:
    """Parse a fraction string like '1/10' or '1'."""
    if "/" in s:
        num, den = s.split("/", 1)
        return Fraction(int(num), int(den))
    return Fraction(int(s))


def _resolve_models_for_site(
    site_cfg: DCSiteConfig,
    all_models: tuple[InferenceModelSpec, ...],
) -> tuple[InferenceModelSpec, ...]:
    """Return the subset of models assigned to a site."""
    if site_cfg.models is None:
        return all_models
    label_set = set(site_cfg.models)
    matched = tuple(m for m in all_models if m.model_label in label_set)
    missing = label_set - {m.model_label for m in matched}
    if missing:
        raise ValueError(f"DC site references unknown model labels: {missing}")
    return matched


# ── Run a single mode ───────────────────────────────────────────────────────


def run_mode(
    mode: str,
    *,
    config: IEEE13Config,
    all_models: tuple[InferenceModelSpec, ...],
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    dt_dc: Fraction,
    dt_grid: Fraction,
    dt_ctrl: Fraction,
    save_dir: Path,
    tap_schedule: TapSchedule | None = None,
    folder_name: str | None = None,
) -> tuple[VoltageStats, Any]:
    """Run either 'baseline' or 'ofo' and return (voltage_stats, sim_log)."""
    sim = config.simulation
    if folder_name is None:
        folder_name = "baseline_no-tap" if mode == "baseline" else mode
    run_dir = save_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    assert config.dc_sites is not None
    site_ids = list(config.dc_sites.keys())
    dc_loads: dict[str, DCLoadSpec] = {}
    datacenters: dict[str, OfflineDatacenter] = {}
    controllers: list = []
    site_models_map: dict[str, tuple[InferenceModelSpec, ...]] = {}
    primary_bus = ""

    for site_id, site_cfg in config.dc_sites.items():
        site_models = _resolve_models_for_site(site_cfg, all_models)
        site_models_map[site_id] = site_models
        site_inference = inference_data.filter_models(site_models) if site_cfg.models is not None else inference_data

        dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=site_cfg.base_kw_per_phase)

        # Build workload with optional per-site inference ramps (preferred)
        # or top-level inference_ramp (backward compat)
        workload_kwargs: dict = {"inference_data": site_inference}
        if config.training is not None:
            tc = config.training
            workload_kwargs["training"] = TrainingRun(
                n_gpus=tc.n_gpus, trace=training_trace,
                target_peak_W_per_gpu=tc.target_peak_W_per_gpu,
            ).at(t_start=tc.t_start, t_end=tc.t_end)
        if site_cfg.inference_ramps:
            # Per-site inference ramps (new style)
            ramp_schedule = None
            for rc in site_cfg.inference_ramps:
                entry = InferenceRamp(target=rc.target, model=rc.model).at(
                    t_start=rc.t_start, t_end=rc.t_end,
                )
                ramp_schedule = entry if ramp_schedule is None else (ramp_schedule | entry)
            workload_kwargs["inference_ramps"] = ramp_schedule
        elif config.inference_ramp is not None:
            # Top-level inference_ramp (backward compat)
            rc = config.inference_ramp
            workload_kwargs["inference_ramps"] = InferenceRamp(target=rc.target, model=rc.model).at(
                t_start=rc.t_start, t_end=rc.t_end,
            )
        workload = OfflineWorkload(**workload_kwargs)

        dc = OfflineDatacenter(
            dc_config, workload, dt_s=dt_dc, seed=site_cfg.seed,
            power_augmentation=config.power_augmentation,
            total_gpu_capacity=site_cfg.total_gpu_capacity,
        )
        datacenters[site_id] = dc
        dc_loads[site_id] = DCLoadSpec(
            bus=site_cfg.bus, bus_kv=site_cfg.bus_kv,
            connection_type=site_cfg.connection_type,
        )
        if not primary_bus:
            primary_bus = site_cfg.bus

    # Build grid
    initial_taps = _taps_dict_to_position(config.initial_taps)
    exclude_buses = tuple(config.exclude_buses)

    grid = ScenarioOpenDSSGrid(
        pv_systems=config.pv_systems,
        time_varying_loads=config.time_varying_loads,
        t_total=float(sim.total_duration_s),
        source_pu=config.source_pu,
        dss_case_dir=config.ieee_case_dir,
        dss_master_file=config.dss_master_file,
        dc_loads=dc_loads,
        power_factor=DatacenterConfig(base_kw_per_phase=0).power_factor,
        dt_s=dt_grid,
        initial_tap_position=initial_taps,
        exclude_buses=exclude_buses,
    )

    # Build controllers
    sched = tap_schedule if tap_schedule is not None else TapSchedule(())
    controllers.append(TapScheduleController(schedule=sched, dt_s=dt_ctrl))

    if mode == "ofo":
        ofo_params = config.ofo
        ofo_config = OFOConfig(
            primal_step_size=ofo_params.primal_step_size,
            w_throughput=ofo_params.w_throughput,
            w_switch=ofo_params.w_switch,
            voltage_gradient_scale=ofo_params.voltage_gradient_scale,
            v_min=sim.v_min,
            v_max=sim.v_max,
            voltage_dual_step_size=ofo_params.voltage_dual_step_size,
            latency_dual_step_size=ofo_params.latency_dual_step_size,
            sensitivity_update_interval=ofo_params.sensitivity_update_interval,
            sensitivity_perturbation_kw=ofo_params.sensitivity_perturbation_kw,
        )
        for site_id, site_cfg in config.dc_sites.items():
            site_models = site_models_map[site_id]
            ofo_ctrl = OFOBatchSizeController(
                site_models,
                models=logistic_models,
                config=ofo_config,
                dt_s=dt_ctrl,
                site_id=site_id if len(config.dc_sites) > 1 else None,
            )
            controllers.append(ofo_ctrl)

    # Run simulation
    logger.info("=== %s ===", folder_name.upper())
    if len(datacenters) == 1 and "default" in datacenters:
        coord = Coordinator(
            datacenter=datacenters["default"],
            grid=grid,
            controllers=controllers,
            total_duration_s=sim.total_duration_s,
            dc_bus=primary_bus,
        )
    else:
        coord = Coordinator(
            datacenters=datacenters,
            grid=grid,
            controllers=controllers,
            total_duration_s=sim.total_duration_s,
            dc_bus=primary_bus,
        )
    log = coord.run()

    # Stats
    stats = compute_allbus_voltage_stats(
        log.grid_states, v_min=sim.v_min, v_max=sim.v_max,
        exclude_buses=exclude_buses,
    )
    logger.info("  Violation time: %.1f s, Vmin: %.4f, Vmax: %.4f, Integral: %.4f",
                stats.violation_time_s, stats.worst_vmin, stats.worst_vmax,
                stats.integral_violation_pu_s)

    # Per-site power stats
    for site_id in site_ids:
        site_states = log.dc_states_by_site.get(site_id, [])
        if site_states:
            kW = np.array([(s.power_w.a + s.power_w.b + s.power_w.c) / 3e3 for s in site_states])
            logger.info("  %s @ bus %s: avg power = %.1f kW/phase",
                        site_id, config.dc_sites[site_id].bus, kW.mean())

    # Plots
    time_s = np.array(log.time_s)

    plot_allbus_voltages_per_phase(
        log.grid_states, time_s, save_dir=run_dir,
        v_min=sim.v_min, v_max=sim.v_max,
        drop_buses=list(exclude_buses),
        title_template=f"IEEE 13 {folder_name.upper()} — Voltage (Phase {{label}})",
    )

    # Per-site plots
    for site_id in site_ids:
        site_states = log.dc_states_by_site.get(site_id, [])
        if not site_states:
            continue

        suffix = f"_{site_id}" if len(site_ids) > 1 else ""
        site_models = site_models_map[site_id]
        dc_time_s = np.array([s.time_s for s in site_states])
        kW_A = np.array([s.power_w.a / 1e3 for s in site_states])
        kW_B = np.array([s.power_w.b / 1e3 for s in site_states])
        kW_C = np.array([s.power_w.c / 1e3 for s in site_states])

        per_model = extract_per_model_timeseries(site_states)

        logger.info("=== Batch Schedule Summary%s ===", f" ({site_id})" if len(site_ids) > 1 else "")
        for label, batches in per_model.batch_size.items():
            if batches.size:
                avg = float(np.mean(batches))
                changes = int(np.sum(np.diff(batches) != 0))
                logger.info("  %s: avg_batch=%.1f, changes=%d", label, avg, changes)

        if mode == "ofo":
            plot_model_timeseries_4panel(
                per_model.time_s, per_model,
                model_labels=[m.model_label for m in site_models],
                regime_shading=False,
                save_path=run_dir / f"OFO_results{suffix}.png",
            )

    return stats, log


# ── Tap schedule builder ────────────────────────────────────────────────────


def _build_tap_schedule(entries: list[dict], initial_taps: dict[str, float | str] | None) -> TapSchedule:
    """Build a TapSchedule from config entries like [{"t": 1800, "reg1": "+10", ...}]."""
    if not entries:
        return TapSchedule(())
    base_regs = {}
    if initial_taps:
        base_regs = {k: _parse_tap(v) for k, v in initial_taps.items()}
    schedule_entries = []
    for entry in sorted(entries, key=lambda e: e["t"]):
        t = float(entry["t"])
        regs = dict(base_regs)
        for k, v in entry.items():
            if k == "t":
                continue
            regs[k] = _parse_tap(v)
        schedule_entries.append((t, TapPosition(regulators=regs)))
    return TapSchedule(tuple(schedule_entries))


# ── Main ────────────────────────────────────────────────────────────────────


def main(*, config_path: Path, mode: str = "no-tap") -> None:
    config_path = config_path.resolve()
    config = IEEE13Config.model_validate_json(config_path.read_bytes())
    sim = config.simulation

    # Resolve paths relative to config file location
    config_dir = config_path.parent
    config.ieee_case_dir = (config_dir / config.ieee_case_dir).resolve()

    all_models = tuple(config.models)
    data_sources = {s.model_label: s for s in config.data_sources}
    data_dir = config.data_dir or Path("data/offline") / config.data_hash

    dt_dc = _parse_fraction(sim.dt_dc)
    dt_grid = _parse_fraction(sim.dt_grid)
    dt_ctrl = _parse_fraction(sim.dt_ctrl)

    save_dir = (Path(__file__).resolve().parent / "outputs" / "ieee13")
    save_dir.mkdir(parents=True, exist_ok=True)


    # Load shared data
    logger.info("Loading inference data...")
    inference_data = InferenceData.ensure(
        data_dir, all_models, data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir, plot=False, dt_s=float(dt_dc),
    )
    training_trace = TrainingTrace.ensure(data_dir / "training_trace.csv", config.training_trace_params)
    logger.info("Loading logistic fits...")
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv", all_models, data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir, plot=False,
    )

    # Build tap schedule if defined
    has_tap_schedule = config.tap_schedule is not None and len(config.tap_schedule) > 0
    tap_sched = _build_tap_schedule(config.tap_schedule, config.initial_taps) if has_tap_schedule else None

    # Plot PV and load profiles into each case folder
    if config.pv_systems or config.time_varying_loads:
        case_dirs = [save_dir / "baseline_no-tap", save_dir / "ofo"]
        if has_tap_schedule:
            case_dirs.extend([save_dir / "baseline_tap-change", save_dir / "ofo_tap-change"])
        for d in case_dirs:
            d.mkdir(parents=True, exist_ok=True)
            plot_profiles(config.pv_systems, config.time_varying_loads,
                          float(sim.total_duration_s), d / "pv_load_profiles.png")

    shared = dict(
        config=config,
        all_models=all_models,
        inference_data=inference_data,
        training_trace=training_trace,
        logistic_models=logistic_models,
        dt_dc=dt_dc, dt_grid=dt_grid, dt_ctrl=dt_ctrl,
        save_dir=save_dir,
    )

    # Run cases based on --mode flag
    cases: list[tuple[str, str, TapSchedule | None]] = []
    if mode in ("no-tap", "all"):
        cases.append(("baseline", "baseline_no-tap", None))
        cases.append(("ofo", "ofo", None))
    if mode in ("tap-change", "all") and has_tap_schedule:
        cases.append(("baseline", "baseline_tap-change", tap_sched))
        cases.append(("ofo", "ofo_tap-change", tap_sched))
    if not cases:
        logger.warning("No cases to run (mode=%s, has_tap_schedule=%s)", mode, has_tap_schedule)
        return

    results: dict[str, VoltageStats] = {}
    logs: dict[str, Any] = {}
    for mode, folder, sched in cases:
        stats, sim_log = run_mode(mode, **shared, tap_schedule=sched, folder_name=folder)
        results[folder] = stats
        logs[folder] = sim_log

    # Comparison CSV
    csv_path = save_dir / "results_ieee13_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "violation_time_s", "worst_vmin", "worst_vmax",
                         "integral_violation_pu_s"])
        for case_name, s in results.items():
            writer.writerow([case_name, s.violation_time_s, s.worst_vmin,
                             s.worst_vmax, s.integral_violation_pu_s])
    logger.info("Comparison CSV: %s", csv_path)

    # Comparison bar chart
    _plot_comparison(results, save_dir)

    # Print summary table
    n_cases = len(results)
    title = "IEEE 13-Bus Comparison" + (" (4-case)" if n_cases == 4 else "")
    summary_lines = [
        "",
        "=" * 90,
        title,
        "=" * 90,
        f"{'Mode':<22s} {'Viol(s)':>10s} {'Vmin':>10s} {'Vmax':>10s} {'Integral':>14s}",
        "-" * 90,
    ]
    for case_name, s in results.items():
        summary_lines.append(
            f"{case_name:<22s} {s.violation_time_s:>10.1f} {s.worst_vmin:>10.4f} "
            f"{s.worst_vmax:>10.4f} {s.integral_violation_pu_s:>14.4f}"
        )
    summary_lines.append("-" * 90)

    b = results.get("baseline_no-tap")
    o = results.get("ofo")
    if b and o:
        viol_red = f"{(1 - o.violation_time_s / b.violation_time_s) * 100:.0f}%" if b.violation_time_s > 0 else "N/A"
        integ_red = f"{(1 - o.integral_violation_pu_s / b.integral_violation_pu_s) * 100:.0f}%" if b.integral_violation_pu_s > 0 else "N/A"
        summary_lines.append(f"OFO vs baseline:  violation time {viol_red},  integral {integ_red}")
    bt = results.get("baseline_tap-change")
    ot = results.get("ofo_tap-change")
    if bt and ot and b:
        viol_red = f"{(1 - ot.violation_time_s / b.violation_time_s) * 100:.0f}%" if b.violation_time_s > 0 else "N/A"
        integ_red = f"{(1 - ot.integral_violation_pu_s / b.integral_violation_pu_s) * 100:.0f}%" if b.integral_violation_pu_s > 0 else "N/A"
        summary_lines.append(f"OFO+tap vs baseline:  violation time {viol_red},  integral {integ_red}")

    summary_lines.append(f"Outputs: {save_dir}")

    for line in summary_lines:
        print(line)
        logger.info(line)


def _plot_comparison(results: dict[str, VoltageStats], save_dir: Path) -> None:
    """Bar chart comparing baseline vs OFO voltage metrics."""
    import matplotlib.pyplot as plt

    modes = list(results.keys())
    n = len(modes)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:n]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Violation time
    vals = [results[m].violation_time_s for m in modes]
    axes[0].bar(modes, vals, color=colors)
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Voltage Violation Time")
    axes[0].tick_params(axis="x", rotation=20)

    # Worst Vmin
    vals = [results[m].worst_vmin for m in modes]
    axes[1].bar(modes, vals, color=colors)
    axes[1].axhline(0.95, color="red", linestyle="--", alpha=0.7, label="Vmin limit")
    axes[1].set_ylabel("Per Unit")
    axes[1].set_title("Worst Vmin")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=20)

    # Integral violation
    vals = [results[m].integral_violation_pu_s for m in modes]
    axes[2].bar(modes, vals, color=colors)
    axes[2].set_ylabel("pu * s")
    axes[2].set_title("Integral Violation")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle("IEEE 13-Bus: Baseline vs OFO", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_dir / "results_ieee13_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison plot: %s", save_dir / "results_ieee13_comparison.png")


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        config: str
        """Path to the config JSON file."""
        mode: str = "no-tap"
        """Run mode: 'no-tap' (default), 'tap-change', or 'all' (both)."""
        log_level: str = "INFO"
        """Logging verbosity (DEBUG, INFO, WARNING)."""

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    main(config_path=Path(args.config), mode=args.mode)
