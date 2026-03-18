"""IEEE 13-bus baseline simulation with multi-DC, PV, and external load support.

Components: OfflineDatacenter(s) + ScenarioOpenDSSGrid + TapScheduleController + Coordinator.

Supports:
  - Multiple datacenter sites at different buses (via dc_sites config)
  - Rooftop PV systems at arbitrary buses (via pv_systems config)
  - Time-varying external loads at arbitrary buses (via time_varying_loads config)
  - PV/load profiles from CSV files or auto-generated sinusoidal curves

Two baseline modes:
  no-tap       Fixed taps throughout the simulation.
  tap-change   Scheduled tap changes (configurable via initial_taps / tap_schedule).

Usage:
    python examples/offline/run_ieee13_baseline.py --config examples/offline/config.json
    python examples/offline/run_ieee13_baseline.py --config examples/offline/config_ieee13.json --mode tap-change
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
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
from openg2g.metrics.voltage import compute_allbus_voltage_stats

from plot_all_figures import (
    extract_per_model_timeseries,
    plot_allbus_voltages_per_phase,
    plot_power_and_itl_2panel,
)

logger = logging.getLogger("run_ieee13_baseline")


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
    # Slow downward trend: 0.85 -> 0.55 over the hour
    trend = 0.85 - 0.30 * (t / T)
    # Two cloud dips
    cloud = 1.0
    cloud -= 0.55 * _smooth_bump(t, 0.17 * T, 0.033 * T)  # deep dip at ~10 min
    cloud -= 0.40 * _smooth_bump(t, 0.58 * T, 0.050 * T)  # moderate dip at ~35 min
    # Small fluctuations
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
    # Low baseline with three distinct pulses
    base = 0.10
    base += 0.45 * _smooth_bump(t, 0.20 * T, 0.08 * T)   # pulse at ~12 min
    base += 0.65 * _smooth_bump(t, 0.50 * T, 0.10 * T)   # pulse at ~30 min
    base += 0.90 * _smooth_bump(t, 0.80 * T, 0.08 * T)   # pulse at ~48 min
    # Small fluctuations
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

    # PV
    ax = axes[0]
    for i, spec in enumerate(pv_specs):
        y = np.array([pv_profile_kw(ti, spec.peak_kw, t_total) for ti in t])
        ax.plot(t / 60, y, lw=1.5, label=f"PV @ {spec.bus} ({spec.peak_kw:.0f} kW peak)")
    ax.set_ylabel("PV Output (kW/phase)")
    ax.set_title("PV Generation Profiles")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Load
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


class TapChangeEntry(BaseModel):
    """A single scheduled tap change. Extra fields are regulator names."""

    model_config = ConfigDict(extra="allow")
    t: float


class TrainingConfig(BaseModel):
    """Training run overlay configuration."""

    dc_site: str | None = None
    """DC site ID this training job runs on. ``None`` uses the first site."""
    n_gpus: int = 2400
    target_peak_W_per_gpu: float = 400.0
    t_start: float = 1000.0
    t_end: float = 2000.0


class InferenceRampConfig(BaseModel):
    """Single inference ramp event configuration."""

    target: float = 0.2
    t_start: float = 2500.0
    t_end: float = 3000.0
    model: str | None = None


class SimulationParams(BaseModel):
    """Simulation timing and voltage parameters."""

    total_duration_s: int = 3600
    dt_dc: str = "1/10"
    dt_grid: str = "1/10"
    dt_ctrl: str = "1"
    v_min: float = 0.95
    v_max: float = 1.05


class IEEE13Config(BaseModel):
    """Full configuration for IEEE 13-bus baseline/OFO simulations."""

    models: list[InferenceModelSpec]
    data_sources: list[MLEnergySource]
    training_trace_params: TrainingTraceParams = TrainingTraceParams()
    data_dir: Path | None = None
    ieee_case_dir: Path
    dss_master_file: str = "IEEE13Nodeckt.dss"
    mlenergy_data_dir: Path | None = None

    dc_sites: dict[str, DCSiteConfig] | None = None
    """Datacenter sites keyed by site ID. If None, uses single default site."""
    pv_systems: list[PVSystemConfig] = []
    time_varying_loads: list[TimeVaryingLoadConfig] = []
    initial_taps: dict[str, float | str] | None = None
    tap_schedule: list[TapChangeEntry] = []

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pv_specs = list(pv_systems or [])
        self._load_specs = list(time_varying_loads or [])
        self._t_total = t_total

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

        # DSS load names: PV_{idx}_{A,B,C} and ExtLoad_{idx}_{A,B,C}
        self._pv_load_names: list[tuple[str, str, str]] = [
            (f"PV_{i}_A", f"PV_{i}_B", f"PV_{i}_C") for i in range(len(self._pv_specs))
        ]
        self._ext_load_names: list[tuple[str, str, str]] = [
            (f"ExtLoad_{i}_A", f"ExtLoad_{i}_B", f"ExtLoad_{i}_C") for i in range(len(self._load_specs))
        ]

    def _init_dss(self) -> None:
        super()._init_dss()
        from openg2g.grid.opendss import dss

        # Create PV loads (generation = negative kW)
        for i, spec in enumerate(self._pv_specs):
            kv_ln = spec.bus_kv / math.sqrt(3.0)
            for ph, name in zip((1, 2, 3), self._pv_load_names[i]):
                dss.Text.Command(
                    f"New Load.{name} bus1={spec.bus}.{ph} phases=1 "
                    f"conn=wye kV={kv_ln:.6f} kW=0 kvar=0 model=1"
                )

        # Create external loads
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

        # Update PV outputs
        for i, spec in enumerate(self._pv_specs):
            kw = eval_profile(
                clock.time_s, peak_kw=spec.peak_kw, csv_data=self._pv_csv[i],
                profile_fn=pv_profile_kw, t_total=self._t_total,
            )
            pf = max(min(spec.power_factor, 0.999999), 1e-6)
            kvar = kw * math.tan(math.acos(pf))
            for name in self._pv_load_names[i]:
                dss.Loads.Name(name)
                dss.Loads.kW(-kw)  # Negative = generation
                dss.Loads.kvar(-kvar)

        # Update external loads
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


def _build_tap_schedule(entries: list[TapChangeEntry]) -> TapSchedule:
    """Build a TapSchedule from config entries."""
    if not entries:
        return TapSchedule(())

    def _entry_to_tap(e: TapChangeEntry) -> TapPosition:
        extras = e.model_extra or {}
        return TapPosition(regulators={k: _parse_tap(v) for k, v in extras.items()})

    schedule = _entry_to_tap(entries[0]).at(t=entries[0].t)
    for e in entries[1:]:
        schedule = schedule | _entry_to_tap(e).at(t=e.t)
    return schedule


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

    save_dir = (Path(__file__).resolve().parent / "outputs" / "ieee13" / f"baseline_{mode}")
    save_dir.mkdir(parents=True, exist_ok=True)


    # Load data
    inference_data = InferenceData.ensure(
        data_dir, all_models, data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir, plot=False, dt_s=float(dt_dc),
    )
    training_trace = TrainingTrace.ensure(data_dir / "training_trace.csv", config.training_trace_params)

    # Build per-site datacenters
    assert config.dc_sites is not None
    site_ids = list(config.dc_sites.keys())
    dc_loads: dict[str, DCLoadSpec] = {}
    datacenters: dict[str, OfflineDatacenter] = {}
    primary_bus = ""

    for site_id, site_cfg in config.dc_sites.items():
        site_models = _resolve_models_for_site(site_cfg, all_models)
        site_inference = inference_data.filter_models(site_models) if site_cfg.models is not None else inference_data

        dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=site_cfg.base_kw_per_phase)

        # Build workload
        workload_kwargs: dict = {"inference_data": site_inference}
        if config.training is not None:
            tc = config.training
            workload_kwargs["training"] = TrainingRun(
                n_gpus=tc.n_gpus, trace=training_trace,
                target_peak_W_per_gpu=tc.target_peak_W_per_gpu,
            ).at(t_start=tc.t_start, t_end=tc.t_end)
        if config.inference_ramp is not None:
            rc = config.inference_ramp
            workload_kwargs["inference_ramps"] = InferenceRamp(target=rc.target, model=rc.model).at(
                t_start=rc.t_start, t_end=rc.t_end,
            )
        workload = OfflineWorkload(**workload_kwargs)

        dc = OfflineDatacenter(
            dc_config, workload, dt_s=dt_dc, seed=site_cfg.seed,
            power_augmentation=config.power_augmentation,
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
    grid = ScenarioOpenDSSGrid(
        pv_systems=config.pv_systems,
        time_varying_loads=config.time_varying_loads,
        t_total=float(sim.total_duration_s),
        dss_case_dir=config.ieee_case_dir,
        dss_master_file=config.dss_master_file,
        dc_loads=dc_loads,
        power_factor=DatacenterConfig(base_kw_per_phase=0).power_factor,
        dt_s=dt_grid,
        initial_tap_position=initial_taps,
    )

    # Build controller
    if mode == "tap-change":
        tap_schedule = _build_tap_schedule(config.tap_schedule)
    else:
        tap_schedule = TapSchedule(())
    ctrl = TapScheduleController(schedule=tap_schedule, dt_s=dt_ctrl)

    # Plot PV and load profiles
    if config.pv_systems or config.time_varying_loads:
        plot_profiles(config.pv_systems, config.time_varying_loads,
                      float(sim.total_duration_s), save_dir / "pv_load_profiles.png")

    # Run simulation
    logger.info("Running IEEE 13-bus baseline (mode=%s, %d DC sites, %d PV, %d ext loads)...",
                mode, len(datacenters), len(config.pv_systems), len(config.time_varying_loads))

    if len(datacenters) == 1 and "default" in datacenters:
        coord = Coordinator(
            datacenter=datacenters["default"],
            grid=grid,
            controllers=[ctrl],
            total_duration_s=sim.total_duration_s,
            dc_bus=primary_bus,
        )
    else:
        coord = Coordinator(
            datacenters=datacenters,
            grid=grid,
            controllers=[ctrl],
            total_duration_s=sim.total_duration_s,
            dc_bus=primary_bus,
        )
    log = coord.run()

    # Stats
    stats = compute_allbus_voltage_stats(log.grid_states, v_min=sim.v_min, v_max=sim.v_max)
    logger.info("=== Voltage Statistics (all-bus) ===")
    logger.info("  voltage_violation_time = %.1f s", stats.violation_time_s)
    logger.info("  worst_vmin             = %.6f", stats.worst_vmin)
    logger.info("  worst_vmax             = %.6f", stats.worst_vmax)
    logger.info("  integral_violation     = %.4f pu*s", stats.integral_violation_pu_s)

    # Per-site power stats
    for site_id in site_ids:
        site_states = log.dc_states_by_site.get(site_id, [])
        if site_states:
            kW = np.array([(s.power_w.a + s.power_w.b + s.power_w.c) / 3e3 for s in site_states])
            logger.info("  Site '%s' @ bus %s: avg power = %.1f kW/phase",
                        site_id, config.dc_sites[site_id].bus, kW.mean())

    # Plots
    time_s = np.array(log.time_s)

    plot_allbus_voltages_per_phase(
        log.grid_states, time_s, save_dir=save_dir,
        v_min=sim.v_min, v_max=sim.v_max,
        title_template="IEEE 13 Baseline — Voltage (Phase {label})",
    )

    # Per-site power and latency plots
    for site_id in site_ids:
        site_states = log.dc_states_by_site.get(site_id, [])
        if not site_states:
            continue
        dc_time_s = np.array([s.time_s for s in site_states])
        kW_A = np.array([s.power_w.a / 1e3 for s in site_states])
        kW_B = np.array([s.power_w.b / 1e3 for s in site_states])
        kW_C = np.array([s.power_w.c / 1e3 for s in site_states])

        suffix = f"_{site_id}" if len(site_ids) > 1 else ""
        per_model = extract_per_model_timeseries(site_states)

        plot_power_and_itl_2panel(
            dc_time_s, kW_A, kW_B, kW_C,
            avg_itl_by_model=per_model.itl_s,
            itl_time_s=per_model.time_s,
            save_path=save_dir / f"power_latency_subfigs{suffix}.png",
        )

    logger.info("Outputs saved to: %s", save_dir)


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        config: str
        """Path to the config JSON file."""
        mode: str = "no-tap"
        """Baseline variant: 'no-tap' (fixed taps) or 'tap-change' (scheduled tap changes)."""
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
