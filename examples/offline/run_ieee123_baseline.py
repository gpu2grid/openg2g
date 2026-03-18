"""IEEE 123-bus baseline simulation with 4-zone multi-DC, PV, and time-varying loads.

Four DC sites across the feeder with PV generation, industrial loads, and
per-site inference server ramps creating time-varying voltage stress.
Fixed taps, no OFO control.

Usage:
    python run_ieee123_baseline.py --config config_ieee123.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.config import (
    DatacenterConfig,
    InferenceModelSpec,
    InferenceRamp,
    PowerAugmentationConfig,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData, MLEnergySource
from openg2g.datacenter.workloads.training import TrainingTraceParams
from openg2g.events import EventEmitter
from openg2g.grid.base import GridState
from openg2g.grid.config import DCLoadSpec, TapPosition, TapSchedule
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import compute_allbus_voltage_stats

from plot_all_figures import (
    extract_per_model_timeseries,
    plot_power_and_itl_2panel,
    plot_zone_voltage_envelope,
)

logger = logging.getLogger("run_ieee123_baseline")

TAP_STEP = 0.00625
T_TOTAL_S = 3600


# -- PV and load profile functions --------------------------------------------


def _smooth_bump(t: float, t_center: float, half_width: float) -> float:
    dt = abs(t - t_center)
    if dt >= half_width:
        return 0.0
    x = dt / half_width
    return (1 - x * x) ** 2


def _irregular_fluct(t: float, seed: float = 0.0) -> float:
    """Irregular fluctuation via superposition of incommensurate frequencies.

    Returns a value centred around 1.0 with ~±15% variation.
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

    Three distinct PV profiles with different cloud events, trends, and
    fluctuation characteristics so each site looks visually different.
    """
    T = T_TOTAL_S
    if site_idx == 0:
        # Site 0 (bus 1, z1_sw): Declining afternoon output with two cloud dips
        trend = 0.85 - 0.30 * (t / T)
        cloud = 1.0
        cloud -= 0.55 * _smooth_bump(t, 600, 120)
        cloud -= 0.40 * _smooth_bump(t, 2100, 180)
        fluct = _irregular_fluct(t, seed=0.3)
        return max(0.0, peak_kw * trend * max(cloud, 0.05) * fluct)
    elif site_idx == 1:
        # Site 1 (bus 48, z2_nw): Midday peak with passing cloud bank
        ramp = 0.55 + 0.40 * _smooth_bump(t, 1200, 900)
        cloud = 1.0
        cloud -= 0.60 * _smooth_bump(t, 1680, 240)
        cloud -= 0.25 * _smooth_bump(t, 2400, 150)
        fluct = _irregular_fluct(t, seed=2.1)
        return max(0.0, peak_kw * ramp * max(cloud, 0.05) * fluct)
    else:
        # Site 2 (bus 99, z4_ne): Morning ramp, sustained high, late cloud event
        ramp = 0.30 + 0.65 * min(1.0, t / 900.0)
        cloud = 1.0
        cloud -= 0.70 * _smooth_bump(t, 2700, 300)
        cloud -= 0.30 * _smooth_bump(t, 1200, 100)
        fluct = _irregular_fluct(t, seed=5.7)
        return max(0.0, peak_kw * ramp * max(cloud, 0.05) * fluct)


def load_profile_kw(t: float, peak_kw: float, site_idx: int = 0) -> float:
    """Time-varying industrial load (kW per phase) with per-site patterns."""
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


# -- Config models -------------------------------------------------------------


class InferenceRampConfig(BaseModel):
    """Single inference ramp event configuration."""

    target: float = 0.2
    t_start: float = 2500.0
    t_end: float = 3000.0
    model: str | None = None


class DCSiteConfig(BaseModel):
    bus: str
    bus_kv: float = 4.16
    base_kw_per_phase: float = 100.0
    models: list[str] | None = None
    connection_type: Literal["wye", "delta"] = "wye"
    seed: int = 0
    inference_ramps: list[InferenceRampConfig] = []
    total_gpu_capacity: int | None = None


class PVSystemConfig(BaseModel):
    bus: str
    bus_kv: float = 4.16
    peak_kw: float = 100.0
    csv_path: Path | None = None
    power_factor: float = 1.0


class TimeVaryingLoadConfig(BaseModel):
    bus: str
    bus_kv: float = 4.16
    peak_kw: float = 50.0
    csv_path: Path | None = None
    power_factor: float = 0.96


def _parse_tap(val: float | str | None) -> float | None:
    if val is None:
        return None
    if isinstance(val, str):
        return 1.0 + int(val) * TAP_STEP
    return float(val)


def _taps_dict_to_position(d: dict[str, float | str] | None) -> TapPosition | None:
    if not d:
        return None
    return TapPosition(regulators={k: _parse_tap(v) for k, v in d.items()})


class OFOParams(BaseModel):
    primal_step_size: float = 0.05
    w_throughput: float = 1e-3
    w_switch: float = 1.0
    voltage_gradient_scale: float = 1e6
    voltage_dual_step_size: float = 0.3
    latency_dual_step_size: float = 1.0
    sensitivity_update_interval: int = 3600
    sensitivity_perturbation_kw: float = 10.0


class SimulationParams(BaseModel):
    total_duration_s: int = 3600
    dt_dc: str = "1/10"
    dt_grid: str = "1/10"
    dt_ctrl: str = "1"
    v_min: float = 0.95
    v_max: float = 1.05


class IEEE123Config(BaseModel):
    models: list[InferenceModelSpec]
    data_sources: list[MLEnergySource]
    training_trace_params: TrainingTraceParams = TrainingTraceParams()
    data_dir: Path | None = None
    ieee_case_dir: Path
    dss_master_file: str = "IEEE123Master.dss"
    mlenergy_data_dir: Path | None = None

    source_pu: float = 1.0

    dc_sites: dict[str, DCSiteConfig]
    pv_systems: list[PVSystemConfig] = []
    time_varying_loads: list[TimeVaryingLoadConfig] = []
    initial_taps: dict[str, float | str] | None = None
    exclude_buses: list[str] = []
    zones: dict[str, list[str]] | None = None

    ofo: OFOParams = OFOParams()
    simulation: SimulationParams = SimulationParams()
    power_augmentation: PowerAugmentationConfig = PowerAugmentationConfig(
        amplitude_scale_range=(0.98, 1.02), noise_fraction=0.005,
    )

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


# -- Scenario Grid -------------------------------------------------------------


class ScenarioOpenDSSGrid(OpenDSSGrid):
    """OpenDSSGrid with time-varying PV/load and optional source voltage override."""

    def __init__(
        self,
        *,
        pv_systems: list[PVSystemConfig] | None = None,
        time_varying_loads: list[TimeVaryingLoadConfig] | None = None,
        source_pu: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pv_specs = list(pv_systems or [])
        self._load_specs = list(time_varying_loads or [])
        self._source_pu = source_pu

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
                    f"conn=wye kV={kv_ln:.6f} kW=0 kvar=0 model=1 vminpu=0.85"
                )

        for i, spec in enumerate(self._load_specs):
            kv_ln = spec.bus_kv / math.sqrt(3.0)
            for ph, name in zip((1, 2, 3), self._ext_load_names[i]):
                dss.Text.Command(
                    f"New Load.{name} bus1={spec.bus}.{ph} phases=1 "
                    f"conn=wye kV={kv_ln:.6f} kW=0 kvar=0 model=1 vminpu=0.85"
                )

    def step(
        self,
        clock: SimulationClock,
        power_samples_w: dict[str, list[ThreePhase]],
        events: EventEmitter,
    ) -> GridState:
        from openg2g.grid.opendss import dss

        for i, spec in enumerate(self._pv_specs):
            pv_kw = pv_profile_kw(clock.time_s, spec.peak_kw, site_idx=i)
            pf = max(min(spec.power_factor, 0.999999), 1e-6)
            pv_kvar = pv_kw * math.tan(math.acos(pf))
            for name in self._pv_load_names[i]:
                dss.Loads.Name(name)
                dss.Loads.kW(-pv_kw)
                dss.Loads.kvar(-pv_kvar)

        for i, spec in enumerate(self._load_specs):
            lkw = load_profile_kw(clock.time_s, spec.peak_kw, site_idx=i)
            pf = max(min(spec.power_factor, 0.999999), 1e-6)
            lkvar = lkw * math.tan(math.acos(pf))
            for name in self._ext_load_names[i]:
                dss.Loads.Name(name)
                dss.Loads.kW(lkw)
                dss.Loads.kvar(lkvar)

        return super().step(clock, power_samples_w, events)


# -- Helpers -------------------------------------------------------------------


def _parse_fraction(s: str) -> Fraction:
    if "/" in s:
        num, den = s.split("/", 1)
        return Fraction(int(num), int(den))
    return Fraction(int(s))


def _resolve_models_for_site(
    site_cfg: DCSiteConfig,
    all_models: tuple[InferenceModelSpec, ...],
) -> tuple[InferenceModelSpec, ...]:
    if site_cfg.models is None:
        return all_models
    label_set = set(site_cfg.models)
    matched = tuple(m for m in all_models if m.model_label in label_set)
    missing = label_set - {m.model_label for m in matched}
    if missing:
        raise ValueError(f"DC site references unknown model labels: {missing}")
    return matched


def _make_bus_color_map(buses: list[str]) -> dict[str, tuple[float, ...]]:
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    color_map = {}
    for i, bus in enumerate(sorted(buses, key=lambda b: b.lower())):
        color_map[bus] = cmap(i / max(len(buses) - 1, 1))
    return color_map


def _plot_event_profiles(pv_specs, load_specs, save_dirs: list[Path]) -> None:
    """Plot PV and load profiles, saving to multiple directories."""
    import matplotlib.pyplot as plt

    t = np.linspace(0, T_TOTAL_S, 4000)

    n_pv = len(pv_specs)
    n_load = len(load_specs)
    if n_pv + n_load == 0:
        return
    fig, axes = plt.subplots(n_pv + n_load, 1, figsize=(14, 3.5 * (n_pv + n_load)), sharex=True)
    if n_pv + n_load == 1:
        axes = [axes]

    for i, spec in enumerate(pv_specs):
        pv = [pv_profile_kw(ti, spec.peak_kw, site_idx=i) for ti in t]
        axes[i].plot(t / 60, pv, "orange", linewidth=1.5)
        axes[i].set_ylabel("kW/phase", fontsize=15)
        axes[i].set_title(f"PV @ bus {spec.bus} (peak={spec.peak_kw:.0f} kW/ph)", fontsize=16)
        axes[i].axhline(0, color="gray", linewidth=0.5)
        axes[i].set_ylim(bottom=-2)
        axes[i].tick_params(labelsize=13)

    for i, spec in enumerate(load_specs):
        load = [load_profile_kw(ti, spec.peak_kw, site_idx=i) for ti in t]
        ax = axes[n_pv + i]
        ax.plot(t / 60, load, "steelblue", linewidth=1.5)
        ax.set_ylabel("kW/phase", fontsize=15)
        ax.set_title(f"Load @ bus {spec.bus} (peak={spec.peak_kw:.0f} kW/ph)", fontsize=16)
        ax.tick_params(labelsize=13)

    axes[-1].set_xlabel("Time (min)", fontsize=15)
    plt.tight_layout()
    for d in save_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "pv_load_profiles.png", dpi=150)
        logger.info("Saved: %s", d / "pv_load_profiles.png")
    plt.close()


# -- Main ----------------------------------------------------------------------


def main(*, config_path: Path) -> None:
    config_path = config_path.resolve()
    config = IEEE123Config.model_validate_json(config_path.read_bytes())
    sim = config.simulation

    config_dir = config_path.parent
    config.ieee_case_dir = (config_dir / config.ieee_case_dir).resolve()

    all_models = tuple(config.models)
    data_sources = {s.model_label: s for s in config.data_sources}
    data_dir = config.data_dir or Path("data/offline") / config.data_hash

    dt_dc = _parse_fraction(sim.dt_dc)
    dt_grid = _parse_fraction(sim.dt_grid)
    dt_ctrl = _parse_fraction(sim.dt_ctrl)

    save_dir = (Path(__file__).resolve().parent / "outputs" / "ieee123" / "baseline_no-tap")
    save_dir.mkdir(parents=True, exist_ok=True)


    # Load data
    logger.info("Loading inference data...")
    inference_data = InferenceData.ensure(
        data_dir, all_models, data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir, plot=False, dt_s=float(dt_dc),
    )

    # Build per-site datacenters
    site_ids = list(config.dc_sites.keys())
    dc_loads: dict[str, DCLoadSpec] = {}
    datacenters: dict[str, OfflineDatacenter] = {}
    primary_bus = ""

    for site_id, site_cfg in config.dc_sites.items():
        site_models = _resolve_models_for_site(site_cfg, all_models)
        site_inference = inference_data.filter_models(site_models) if site_cfg.models is not None else inference_data

        dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=site_cfg.base_kw_per_phase)

        # Build workload with optional per-site inference ramps
        workload_kwargs: dict = {"inference_data": site_inference}
        if site_cfg.inference_ramps:
            ramp_schedule = None
            for rc in site_cfg.inference_ramps:
                entry = InferenceRamp(target=rc.target, model=rc.model).at(
                    t_start=rc.t_start, t_end=rc.t_end,
                )
                ramp_schedule = entry if ramp_schedule is None else (ramp_schedule | entry)
            workload_kwargs["inference_ramps"] = ramp_schedule
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

        ramp_info = ""
        if site_cfg.inference_ramps:
            parts = []
            for rc in site_cfg.inference_ramps:
                model_tag = f" ({rc.model})" if rc.model else ""
                parts.append(f"{rc.target:.0%}{model_tag} @ t={rc.t_start:.0f}-{rc.t_end:.0f}s")
            ramp_info = ", ramps: " + "; ".join(parts)
        logger.info("  Site '%s' @ bus %s: %s, base=%.1f kW/ph%s",
                    site_id, site_cfg.bus,
                    [m.model_label for m in site_models], site_cfg.base_kw_per_phase,
                    ramp_info)

    # Build grid
    initial_taps = _taps_dict_to_position(config.initial_taps)
    exclude_buses = tuple(config.exclude_buses)

    grid = ScenarioOpenDSSGrid(
        pv_systems=config.pv_systems,
        time_varying_loads=config.time_varying_loads,
        source_pu=config.source_pu,
        dss_case_dir=config.ieee_case_dir,
        dss_master_file=config.dss_master_file,
        dc_loads=dc_loads,
        power_factor=DatacenterConfig(base_kw_per_phase=0).power_factor,
        dt_s=dt_grid,
        initial_tap_position=initial_taps,
        exclude_buses=exclude_buses,
    )

    # No OFO, fixed taps
    ctrl = TapScheduleController(schedule=TapSchedule(()), dt_s=dt_ctrl)

    # Plot PV/load profiles
    _plot_event_profiles(config.pv_systems, config.time_varying_loads, [save_dir])

    logger.info("Running IEEE 123-bus baseline (%d DC sites, %d PV, %d ext loads, no OFO)...",
                len(datacenters), len(config.pv_systems), len(config.time_varying_loads))

    coord = Coordinator(
        datacenters=datacenters,
        grid=grid,
        controllers=[ctrl],
        total_duration_s=sim.total_duration_s,
        dc_bus=primary_bus,
    )
    log = coord.run()

    # Stats
    stats = compute_allbus_voltage_stats(
        log.grid_states, v_min=sim.v_min, v_max=sim.v_max,
        exclude_buses=exclude_buses,
    )
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

    if config.zones:
        plot_zone_voltage_envelope(
            log.grid_states, time_s, zones=config.zones,
            save_dir=save_dir, v_min=sim.v_min, v_max=sim.v_max,
            drop_buses=tuple(config.exclude_buses),
            title_template="IEEE 123 Baseline -- Voltage Envelope (Phase {label})",
        )
    else:
        from plot_all_figures import plot_allbus_voltages_per_phase
        drop = set(b.lower() for b in exclude_buses)
        all_buses = [b for b in log.grid_states[0].voltages.buses() if b.lower() not in drop]
        bus_colors = _make_bus_color_map(all_buses)
        plot_allbus_voltages_per_phase(
            log.grid_states, time_s, save_dir=save_dir,
            v_min=sim.v_min, v_max=sim.v_max,
            bus_color_map=bus_colors,
            drop_buses=exclude_buses,
            title_template="IEEE 123 Baseline -- Voltage (Phase {label})",
        )

    for site_id in site_ids:
        site_states = log.dc_states_by_site.get(site_id, [])
        if not site_states:
            continue
        dc_time_s = np.array([s.time_s for s in site_states])
        kW_A = np.array([s.power_w.a / 1e3 for s in site_states])
        kW_B = np.array([s.power_w.b / 1e3 for s in site_states])
        kW_C = np.array([s.power_w.c / 1e3 for s in site_states])
        per_model = extract_per_model_timeseries(site_states)
        plot_power_and_itl_2panel(
            dc_time_s, kW_A, kW_B, kW_C,
            avg_itl_by_model=per_model.itl_s,
            itl_time_s=per_model.time_s,
            show_regimes=False,
            save_path=save_dir / f"power_latency_{site_id}.png",
        )

    # Save results CSV
    csv_path = save_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "violation_time_s", "worst_vmin", "worst_vmax",
                         "integral_violation_pu_s"])
        writer.writerow(["baseline", stats.violation_time_s, stats.worst_vmin,
                         stats.worst_vmax, stats.integral_violation_pu_s])
    logger.info("Results CSV: %s", csv_path)
    logger.info("Outputs saved to: %s", save_dir)

    print(f"\n=== IEEE 123 Baseline ===")
    print(f"  Violation time:  {stats.violation_time_s:.1f} s")
    print(f"  Worst Vmin:      {stats.worst_vmin:.4f}")
    print(f"  Worst Vmax:      {stats.worst_vmax:.4f}")
    print(f"  Integral:        {stats.integral_violation_pu_s:.4f} pu*s")
    print(f"  Outputs:         {save_dir}")


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        config: str
        """Path to the config JSON file."""
        log_level: str = "INFO"

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    main(config_path=Path(args.config))
