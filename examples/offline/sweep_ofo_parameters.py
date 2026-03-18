"""OFO hyperparameter sweep.

Reads OFO baseline parameters and system configuration from a JSON config file,
builds a sweep grid centred on those baseline values, and runs OFO for each
combination. Works with any IEEE test feeder (13-, 34-, 123-bus, etc.).

Sweep mode auto-selects based on the number of DC sites in the config:
  - 1 DC site  -> 1-D sweep: varies each parameter one-at-a-time.
  - 2+ DC sites -> 2-D sweep: sweeps all per-site parameter combinations
                   independently, producing heatmap visualizations.

Outputs (in outputs/<system>/sweep_ofo_parameters/):
  results_<system>_sweep_ofo_parameters.csv   -- one row per run
  <param>__<value>/                           -- per-run plots (1-D mode)
  <param>__<site1>_<v1>__<site2>_<v2>/        -- per-run plots (2-D mode)

Usage:
    # IEEE 13 (single DC, 1-D sweep)
    python sweep_ofo_parameters.py --config config_ieee13.json --system ieee13

    # IEEE 34 (two DCs, 2-D sweep)
    python sweep_ofo_parameters.py --config config_ieee34.json --system ieee34

    # IEEE 123 (four DCs, 2-D sweep)
    python sweep_ofo_parameters.py --config config_ieee123.json --system ieee123 --dt 60

    # Force 1-D sweep on multi-DC system (shared parameters)
    python sweep_ofo_parameters.py --config config_ieee34.json --system ieee34 --sweep-mode 1d

    # Override time resolution
    python sweep_ofo_parameters.py --config config_ieee34.json --system ieee34 --dt 60
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import time
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from plot_all_figures import (
    extract_per_model_timeseries,
    plot_allbus_voltages_per_phase,
    plot_model_timeseries_4panel,
)
from pydantic import BaseModel, model_validator

from openg2g.controller.ofo import (
    LogisticModelStore,
    OFOBatchSizeController,
    OFOConfig,
)
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
from openg2g.grid.config import DCLoadSpec, TapPosition
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import compute_allbus_voltage_stats

logger = logging.getLogger("sweep_ofo")

# ── Default sweep multipliers (centred on baseline value) ────────────────────

DEFAULT_SWEEP_MULTIPLIERS: dict[str, list[float]] = {
    "primal_step_size": [0.2, 0.5, 1.0, 2.0, 5.0],
    "voltage_dual_step_size": [0.2, 0.5, 1.0, 2.0, 5.0],
    "latency_dual_step_size": [0.1, 1.0, 10.0, 50.0, 100.0],
    "w_throughput": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],
    "w_switch": [0.2, 0.5, 1.0, 2.0, 5.0],
}


def build_sweep_grid(
    baseline: OFOConfig,
    multipliers: dict[str, list[float]] | None = None,
) -> list[tuple[str, float, OFOConfig]]:
    """Build a one-at-a-time sweep grid from baseline OFO config.

    For each parameter, the baseline value is multiplied by each multiplier
    to produce the sweep values.  Multiplier 1.0 = baseline value.
    Special case: if baseline value is 0 (e.g. w_throughput=0), the multiplier
    is treated as an absolute value instead.
    """
    mults = multipliers or DEFAULT_SWEEP_MULTIPLIERS
    runs: list[tuple[str, float, OFOConfig]] = []
    seen: set[str] = set()

    for param_name, mult_list in mults.items():
        base_val = getattr(baseline, param_name)
        for mult in mult_list:
            if base_val == 0:
                value = mult  # absolute value when baseline is 0
            else:
                value = round(base_val * mult, 10)
            rid = f"{param_name}__{value:.6g}"
            if rid in seen:
                continue
            seen.add(rid)
            ofo_cfg = baseline.model_copy(update={param_name: value})
            runs.append((param_name, value, ofo_cfg))

    return runs


def _compute_sweep_values(
    baseline: OFOConfig,
    multipliers: dict[str, list[float]] | None = None,
) -> dict[str, list[float]]:
    """Compute absolute sweep values for each parameter from multipliers."""
    mults = multipliers or DEFAULT_SWEEP_MULTIPLIERS
    result: dict[str, list[float]] = {}
    for param_name, mult_list in mults.items():
        base_val = getattr(baseline, param_name)
        seen: set[float] = set()
        values: list[float] = []
        for mult in mult_list:
            if base_val == 0:
                value = mult
            else:
                value = round(base_val * mult, 10)
            if value not in seen:
                seen.add(value)
                values.append(value)
        result[param_name] = values
    return result


def build_sweep_grid_2d(
    baseline: OFOConfig,
    site_ids: list[str],
    multipliers: dict[str, list[float]] | None = None,
) -> list[tuple[str, dict[str, float], dict[str, OFOConfig]]]:
    """Build a 2-D sweep grid for multi-DC systems.

    For each parameter, creates all combinations of values across sites.
    Returns list of (param_name, {site_id: value}, {site_id: OFOConfig}).
    """
    param_values = _compute_sweep_values(baseline, multipliers)
    runs: list[tuple[str, dict[str, float], dict[str, OFOConfig]]] = []

    for param_name, values in param_values.items():
        for combo in itertools.product(values, repeat=len(site_ids)):
            site_values = dict(zip(site_ids, combo, strict=False))
            site_configs = {sid: baseline.model_copy(update={param_name: val}) for sid, val in site_values.items()}
            runs.append((param_name, site_values, site_configs))

    return runs


# ── Profile generation helpers ────────────────────────────────────────────────

T_TOTAL_S = 3600


def _smooth_bump(t: float, t_center: float, half_width: float) -> float:
    dt = abs(t - t_center)
    if dt >= half_width:
        return 0.0
    x = dt / half_width
    return (1 - x * x) ** 2


def pv_profile_kw(t: float, peak_kw: float, site_idx: int = 0) -> float:
    T = T_TOTAL_S
    if site_idx == 0:
        trend = 0.85 - 0.30 * (t / T)
        cloud = 1.0
        cloud -= 0.55 * _smooth_bump(t, 600, 120)
        cloud -= 0.40 * _smooth_bump(t, 2100, 180)
        fluct = 1.0 + 0.10 * math.sin(2 * math.pi * t / 300.0 + 0.3)
        fluct += 0.05 * math.sin(2 * math.pi * t / 90.0 + 1.7)
        return max(0.0, peak_kw * trend * max(cloud, 0.05) * fluct)
    else:
        ramp = 0.55 + 0.40 * _smooth_bump(t, 1200, 900)
        cloud = 1.0
        cloud -= 0.60 * _smooth_bump(t, 1680, 240)
        fluct = 1.0 + 0.08 * math.sin(2 * math.pi * t / 200.0 + 2.1)
        fluct += 0.04 * math.sin(2 * math.pi * t / 70.0 + 0.5)
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


def load_csv_profile(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def eval_profile(t, *, peak_kw, csv_data, profile_fn, site_idx):
    if csv_data is not None:
        return float(np.interp(t, csv_data[0], csv_data[1]))
    return profile_fn(t, peak_kw, site_idx)


# ── Config models ────────────────────────────────────────────────────────────


class DCSiteConfig(BaseModel):
    bus: str
    bus_kv: float = 4.16
    base_kw_per_phase: float = 500.0
    models: list[str] | None = None
    connection_type: Literal["wye", "delta"] = "wye"
    seed: int = 0
    total_gpu_capacity: int | None = None


class PVSystemConfig(BaseModel):
    bus: str
    bus_kv: float = 4.16
    peak_kw: float = 1000.0
    csv_path: Path | None = None
    power_factor: float = 1.0


class TimeVaryingLoadConfig(BaseModel):
    bus: str
    bus_kv: float = 4.16
    peak_kw: float = 500.0
    csv_path: Path | None = None
    power_factor: float = 0.96


TAP_STEP = 0.00625


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
    primal_step_size: float = 0.1
    w_throughput: float = 1e-3
    w_switch: float = 1.0
    voltage_gradient_scale: float = 1e6
    voltage_dual_step_size: float = 1.0
    latency_dual_step_size: float = 1.0
    sensitivity_update_interval: int = 3600
    sensitivity_perturbation_kw: float = 100.0


class TrainingConfig(BaseModel):
    dc_site: str | None = None
    """DC site ID this training job runs on. ``None`` uses the first site."""
    n_gpus: int = 2400
    target_peak_W_per_gpu: float = 400.0
    t_start: float = 1000.0
    t_end: float = 2000.0


class InferenceRampConfig(BaseModel):
    target: float = 0.2
    t_start: float = 2500.0
    t_end: float = 3000.0
    model: str | None = None


class SimulationParams(BaseModel):
    total_duration_s: int = 3600
    dt_dc: str = "1/10"
    dt_grid: str = "1/10"
    dt_ctrl: str = "1"
    v_min: float = 0.95
    v_max: float = 1.05


class SweepConfig(BaseModel):
    """Full configuration for OFO sweep simulations."""

    models: list[InferenceModelSpec]
    data_sources: list[MLEnergySource]
    training_trace_params: TrainingTraceParams = TrainingTraceParams()
    data_dir: Path | None = None
    ieee_case_dir: Path  # System-specific: path to DSS case files
    dss_master_file: str = "IEEE13Nodeckt.dss"  # System-specific: DSS master file name
    mlenergy_data_dir: Path | None = None

    source_pu: float | None = None  # System-specific: source voltage override (e.g. 1.09 for IEEE 34)

    # System-specific: DC site(s) — bus name, bus_kv, and base_kw_per_phase must match the grid.
    # Single-DC (IEEE 13): one site, all models assigned.
    # Multi-DC (IEEE 34): multiple sites, each with a model subset and different base_kw.
    dc_sites: dict[str, DCSiteConfig] | None = None
    pv_systems: list[PVSystemConfig] = []
    time_varying_loads: list[TimeVaryingLoadConfig] = []
    # System-specific: RegControl names differ per system (reg1/2/3 vs creg1a/b/c etc.)
    initial_taps: dict[str, float | str] | None = None
    # System-specific: exclude buses upstream of regulators or on different voltage laterals
    exclude_buses: list[str] = []

    ofo: OFOParams = OFOParams()
    training: TrainingConfig | None = None
    inference_ramp: InferenceRampConfig | None = None
    simulation: SimulationParams = SimulationParams()
    power_augmentation: PowerAugmentationConfig = PowerAugmentationConfig(
        amplitude_scale_range=(0.98, 1.02),
        noise_fraction=0.005,
    )

    @model_validator(mode="after")
    def _default_dc_site(self) -> SweepConfig:
        if self.dc_sites is None:
            self.dc_sites = {"default": DCSiteConfig(bus="671", bus_kv=4.16, base_kw_per_phase=500.0)}
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


# ── Scenario Grid ────────────────────────────────────────────────────────────


class ScenarioOpenDSSGrid(OpenDSSGrid):
    """OpenDSSGrid with PV systems and external loads at arbitrary buses."""

    def __init__(self, *, pv_systems=None, time_varying_loads=None, source_pu=None, **kwargs):
        super().__init__(**kwargs)
        self._pv_specs = list(pv_systems or [])
        self._load_specs = list(time_varying_loads or [])
        self._source_pu = source_pu

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
            kw = eval_profile(
                clock.time_s,
                peak_kw=spec.peak_kw,
                csv_data=self._pv_csv[i],
                profile_fn=pv_profile_kw,
                site_idx=i,
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
            )
            pf = max(min(spec.power_factor, 0.999999), 1e-6)
            kvar = kw * math.tan(math.acos(pf))
            for name in self._ext_load_names[i]:
                dss.Loads.Name(name)
                dss.Loads.kW(kw)
                dss.Loads.kvar(kvar)

        return super().step(clock, power_samples_w, events)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_fraction(s: str) -> Fraction:
    if "/" in s:
        num, den = s.split("/", 1)
        return Fraction(int(num), int(den))
    return Fraction(int(s))


def _resolve_models_for_site(site_cfg, all_models):
    if site_cfg.models is None:
        return all_models
    label_set = set(site_cfg.models)
    matched = tuple(m for m in all_models if m.model_label in label_set)
    missing = label_set - {m.model_label for m in matched}
    if missing:
        raise ValueError(f"DC site references unknown model labels: {missing}")
    return matched


def _run_id(param: str, value: float) -> str:
    return f"{param}__{value:.6g}"


def _run_id_2d(param: str, site_values: dict[str, float]) -> str:
    parts = [param]
    for sid, val in site_values.items():
        parts.append(f"{sid}_{val:.6g}")
    return "__".join(parts)


def _collect_metrics(
    log,
    *,
    models: tuple[InferenceModelSpec, ...],
    wall_time_s: float,
    param_name: str,
    param_value: float,
    v_min: float,
    v_max: float,
    exclude_buses: tuple[str, ...] = (),
) -> dict:
    """Extract scalar metrics from a SimulationLog into a flat dict."""
    vstats = compute_allbus_voltage_stats(log.grid_states, v_min=v_min, v_max=v_max, exclude_buses=exclude_buses)

    row: dict = {
        "param_name": param_name,
        "param_value": param_value,
        "violation_time_s": vstats.violation_time_s,
        "integral_violation_pu_s": vstats.integral_violation_pu_s,
        "worst_vmin": vstats.worst_vmin,
        "worst_vmax": vstats.worst_vmax,
        "wall_time_s": wall_time_s,
    }

    # Per-site metrics (use per-site states to avoid interleaving artefacts)
    deadlines = {ms.model_label: ms.itl_deadline_s for ms in models}
    for site_id, site_states in log.dc_states_by_site.items():
        if not site_states:
            continue
        kW = np.array([(s.power_w.a + s.power_w.b + s.power_w.c) / 3e3 for s in site_states])
        row[f"avg_power_kw_per_phase__{site_id}"] = float(kW.mean())

        per_model = extract_per_model_timeseries(site_states)
        for label in sorted(per_model.batch_size):
            batches = per_model.batch_size[label]
            row[f"avg_batch__{label}"] = float(np.mean(batches)) if batches.size else float("nan")
            row[f"n_batch_changes__{label}"] = int(np.sum(np.diff(batches) != 0)) if batches.size > 1 else 0

            itl = per_model.itl_s.get(label, np.array([]))
            deadline = deadlines.get(label, float("nan"))
            finite = itl[np.isfinite(itl)]
            row[f"itl_violation_frac__{label}"] = float(np.mean(finite > deadline)) if finite.size > 0 else float("nan")

            # Average throughput (tokens/s) = batch_size * active_replicas / itl
            wrep = per_model.active_replicas.get(label, np.array([]))
            if itl.size > 0 and wrep.size > 0 and batches.size > 0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    throughput = np.where(itl > 0, batches.astype(float) * wrep.astype(float) / itl, np.nan)
                finite_tp = throughput[np.isfinite(throughput)]
                row[f"avg_throughput_tokens_per_s__{label}"] = (
                    float(np.mean(finite_tp)) if finite_tp.size > 0 else float("nan")
                )
            else:
                row[f"avg_throughput_tokens_per_s__{label}"] = float("nan")

    return row


def _plot_sweep_summary(df: pd.DataFrame, save_dir: Path, model_labels: list[str]) -> None:
    """Generate 4-panel summary plots for each swept parameter."""
    import matplotlib.pyplot as plt

    param_names = df["param_name"].unique()

    for param in param_names:
        sub = df[df["param_name"] == param].sort_values("param_value")
        x = sub["param_value"].values

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

        # (1) Violation time (left y) and integral violation (right y)
        ax1 = axes[0, 0]
        color1 = "tab:blue"
        ax1.plot(x, sub["violation_time_s"].values, "o-", color=color1, lw=2, ms=6)
        ax1.set_ylabel("Violation Time (s)", color=color1, fontsize=11)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_xlabel(param, fontsize=11)
        ax1.set_title("(a) Voltage Violation Metrics", fontsize=12)
        ax1.grid(True, alpha=0.3)

        ax1r = ax1.twinx()
        color2 = "tab:red"
        ax1r.plot(x, sub["integral_violation_pu_s"].values, "s--", color=color2, lw=2, ms=6)
        ax1r.set_ylabel("Integral Violation (pu·s)", color=color2, fontsize=11)
        ax1r.tick_params(axis="y", labelcolor=color2)

        # Use log scale for x if values span > 2 orders of magnitude
        if len(x) > 1 and x.min() > 0 and x.max() / x.min() > 100:
            ax1.set_xscale("log")

        # (2) Batch changes per model
        ax2 = axes[0, 1]
        for label in model_labels:
            col = f"n_batch_changes__{label}"
            if col in sub.columns:
                ax2.plot(x, sub[col].values, "o-", lw=1.5, ms=5, label=label)
        ax2.set_xlabel(param, fontsize=11)
        ax2.set_ylabel("Number of Batch Changes", fontsize=11)
        ax2.set_title("(b) Batch Size Changes", fontsize=12)
        ax2.legend(fontsize=8, loc="best")
        ax2.grid(True, alpha=0.3)
        if len(x) > 1 and x.min() > 0 and x.max() / x.min() > 100:
            ax2.set_xscale("log")

        # (3) ITL violation fraction per model
        ax3 = axes[1, 0]
        for label in model_labels:
            col = f"itl_violation_frac__{label}"
            if col in sub.columns:
                ax3.plot(x, sub[col].values, "o-", lw=1.5, ms=5, label=label)
        ax3.set_xlabel(param, fontsize=11)
        ax3.set_ylabel("ITL Violation Fraction", fontsize=11)
        ax3.set_title("(c) ITL Deadline Violation Fraction", fontsize=12)
        ax3.legend(fontsize=8, loc="best")
        ax3.grid(True, alpha=0.3)
        if len(x) > 1 and x.min() > 0 and x.max() / x.min() > 100:
            ax3.set_xscale("log")

        # (4) Average throughput per model
        ax4 = axes[1, 1]
        for label in model_labels:
            col = f"avg_throughput_tokens_per_s__{label}"
            if col in sub.columns:
                ax4.plot(x, sub[col].values, "o-", lw=1.5, ms=5, label=label)
        ax4.set_xlabel(param, fontsize=11)
        ax4.set_ylabel("Avg Throughput (tokens/s)", fontsize=11)
        ax4.set_title("(d) Average Throughput per Model", fontsize=12)
        ax4.set_yscale("log")
        ax4.legend(fontsize=8, loc="best")
        ax4.grid(True, alpha=0.3)
        if len(x) > 1 and x.min() > 0 and x.max() / x.min() > 100:
            ax4.set_xscale("log")

        fig.suptitle(f"OFO Sweep: {param}", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(save_dir / f"sweep_{param}.png", bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved sweep plot: sweep_%s.png", param)


def _build_pairwise_grid(
    sub: pd.DataFrame,
    col0: str,
    col1: str,
    metric_col: str,
    agg: str = "mean",
) -> tuple[list[float], list[float], np.ndarray]:
    """Build a 2-D grid for one metric, aggregating over other dimensions.

    Groups by (col0, col1), applies `agg` (e.g. "mean") to collapse any
    remaining site dimensions, and returns (vals0, vals1, grid[y, x]).
    """
    grouped = sub.groupby([col0, col1])[metric_col].agg(agg).reset_index()
    vals0 = sorted(grouped[col0].unique())
    vals1 = sorted(grouped[col1].unique())
    grid = np.full((len(vals1), len(vals0)), np.nan)
    for _, row in grouped.iterrows():
        i0 = vals0.index(row[col0])
        i1 = vals1.index(row[col1])
        grid[i1, i0] = row[metric_col]
    return vals0, vals1, grid


def _plot_sweep_summary_2d(
    df: pd.DataFrame,
    save_dir: Path,
    site_ids: list[str],
    model_labels: list[str],
) -> None:
    """Generate pairwise heatmap plots for per-site parameter sweeps.

    For each pair of sites (s_i, s_j), produces heatmaps with s_i on the
    x-axis and s_j on the y-axis.  When there are more than 2 sites, each
    heatmap cell is the **mean** over the remaining sites' parameter values.
    """
    import matplotlib.pyplot as plt

    param_names = df["param_name"].unique()

    # Generate all ordered pairs so that every pair appears once
    site_pairs = list(itertools.combinations(site_ids, 2))

    voltage_metrics = [
        ("violation_time_s", "Violation Time (s)"),
        ("integral_violation_pu_s", "Integral Violation (pu*s)"),
        ("worst_vmin", "Worst Vmin (pu)"),
        ("worst_vmax", "Worst Vmax (pu)"),
    ]

    for param in param_names:
        sub = df[df["param_name"] == param]

        for s0, s1 in site_pairs:
            col0 = f"param_value__{s0}"
            col1 = f"param_value__{s1}"
            if col0 not in sub.columns or col1 not in sub.columns:
                continue

            pair_tag = f"{s0}_vs_{s1}"
            agg_note = " (mean over other sites)" if len(site_ids) > 2 else ""

            # ── Voltage metrics heatmap ──────────────────────────────
            fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)

            for ax, (metric_col, metric_label) in zip(axes.flat, voltage_metrics, strict=False):
                if metric_col not in sub.columns:
                    ax.set_visible(False)
                    continue
                vals0, vals1, hgrid = _build_pairwise_grid(sub, col0, col1, metric_col)
                if len(vals0) < 2 or len(vals1) < 2:
                    ax.set_visible(False)
                    continue

                im = ax.imshow(
                    hgrid,
                    origin="lower",
                    aspect="auto",
                    extent=[-0.5, len(vals0) - 0.5, -0.5, len(vals1) - 0.5],
                )
                ax.set_xticks(range(len(vals0)))
                ax.set_xticklabels([f"{v:.4g}" for v in vals0], fontsize=8, rotation=45)
                ax.set_yticks(range(len(vals1)))
                ax.set_yticklabels([f"{v:.4g}" for v in vals1], fontsize=8)
                ax.set_xlabel(f"{param} ({s0})", fontsize=10)
                ax.set_ylabel(f"{param} ({s1})", fontsize=10)
                ax.set_title(metric_label, fontsize=11)
                fig.colorbar(im, ax=ax, shrink=0.8)

                median = np.nanmedian(hgrid)
                for i1_idx in range(len(vals1)):
                    for i0_idx in range(len(vals0)):
                        val = hgrid[i1_idx, i0_idx]
                        if np.isfinite(val):
                            ax.text(
                                i0_idx,
                                i1_idx,
                                f"{val:.2g}",
                                ha="center",
                                va="center",
                                fontsize=7,
                                color="white" if val > median else "black",
                            )

            fig.suptitle(
                f"OFO Sweep: {param} ({s0} vs {s1}){agg_note}",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(save_dir / f"sweep_2d_{param}_{pair_tag}.png", bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved heatmap: sweep_2d_%s_%s.png", param, pair_tag)

            # ── Per-model heatmap ────────────────────────────────────
            model_metrics = []
            for label in model_labels:
                itl_col = f"itl_violation_frac__{label}"
                tp_col = f"avg_throughput_tokens_per_s__{label}"
                if itl_col in sub.columns:
                    model_metrics.append((itl_col, f"ITL Violation Frac ({label})"))
                if tp_col in sub.columns:
                    model_metrics.append((tp_col, f"Avg Throughput ({label})"))

            if not model_metrics:
                continue

            n_plots = len(model_metrics)
            ncols = min(n_plots, 2)
            nrows = math.ceil(n_plots / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), dpi=150, squeeze=False)

            for idx, (metric_col, metric_label) in enumerate(model_metrics):
                ax = axes[idx // ncols, idx % ncols]
                vals0, vals1, hgrid = _build_pairwise_grid(sub, col0, col1, metric_col)
                if len(vals0) < 2 or len(vals1) < 2:
                    ax.set_visible(False)
                    continue

                im = ax.imshow(
                    hgrid,
                    origin="lower",
                    aspect="auto",
                    extent=[-0.5, len(vals0) - 0.5, -0.5, len(vals1) - 0.5],
                )
                ax.set_xticks(range(len(vals0)))
                ax.set_xticklabels([f"{v:.4g}" for v in vals0], fontsize=8, rotation=45)
                ax.set_yticks(range(len(vals1)))
                ax.set_yticklabels([f"{v:.4g}" for v in vals1], fontsize=8)
                ax.set_xlabel(f"{param} ({s0})", fontsize=10)
                ax.set_ylabel(f"{param} ({s1})", fontsize=10)
                ax.set_title(metric_label, fontsize=11)
                fig.colorbar(im, ax=ax, shrink=0.8)

            for idx in range(n_plots, nrows * ncols):
                axes[idx // ncols, idx % ncols].set_visible(False)

            fig.suptitle(
                f"OFO Sweep (per-model): {param} ({s0} vs {s1}){agg_note}",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(save_dir / f"sweep_2d_{param}_{pair_tag}_models.png", bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved model heatmap: sweep_2d_%s_%s_models.png", param, pair_tag)


def _make_bus_color_map(buses: list[str]) -> dict[str, tuple[float, ...]]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    color_map = {}
    for i, bus in enumerate(sorted(buses, key=lambda b: b.lower())):
        color_map[bus] = cmap(i / max(len(buses) - 1, 1))
    return color_map


def _save_plots(
    log,
    *,
    run_dir: Path,
    site_models_map: dict[str, tuple[InferenceModelSpec, ...]],
    site_bus_map: dict[str, str],
    v_min: float,
    v_max: float,
    exclude_buses: tuple[str, ...] = (),
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    time_s = np.array(log.time_s)

    # Per-site OFO result plots (avoids interleaving artefacts)
    for site_id, site_states in log.dc_states_by_site.items():
        if not site_states:
            continue
        site_models = site_models_map.get(site_id)
        if not site_models:
            continue
        bus = site_bus_map.get(site_id, site_id)
        per_model = extract_per_model_timeseries(site_states)
        plot_model_timeseries_4panel(
            per_model.time_s,
            per_model,
            model_labels=[ms.model_label for ms in site_models],
            regime_shading=False,
            save_path=run_dir / f"OFO_results_{bus}.png",
        )

    # Build dynamic bus color map (consistent with run_ieee34_ofo.py)
    drop = {b.lower() for b in exclude_buses}
    all_buses = [b for b in log.grid_states[0].voltages.buses() if b.lower() not in drop]
    bus_colors = _make_bus_color_map(all_buses)

    plot_allbus_voltages_per_phase(
        log.grid_states,
        time_s,
        save_dir=run_dir,
        v_min=v_min,
        v_max=v_max,
        bus_color_map=bus_colors,
        drop_buses=exclude_buses,
        title_template="Voltage trajectories (Phase {label})",
    )


# ── Shared setup ─────────────────────────────────────────────────────────────


def _setup(
    *,
    config_path: Path,
    system: str,
    dt_override: str | None = None,
    output_dir: str | None = None,
):
    """Parse config, load shared data, build datacenters and grid.

    Returns a namespace-like dict with all objects needed by sweep runners.
    """
    config_path = config_path.resolve()
    config = SweepConfig.model_validate_json(config_path.read_bytes())
    sim = config.simulation

    # Resolve paths relative to config file location
    config_dir = config_path.parent
    config.ieee_case_dir = (config_dir / config.ieee_case_dir).resolve()

    all_models = tuple(config.models)
    data_sources = {s.model_label: s for s in config.data_sources}
    data_dir = config.data_dir or Path("data/offline") / config.data_hash

    # Data must always be generated at fine resolution
    data_dt = float(_parse_fraction(sim.dt_dc))

    dt_dc = _parse_fraction(sim.dt_dc)
    dt_grid = _parse_fraction(sim.dt_grid)
    dt_ctrl = _parse_fraction(sim.dt_ctrl)

    if dt_override is not None:
        frac = _parse_fraction(dt_override)
        dt_dc = frac
        dt_grid = frac
        dt_ctrl = frac

    if output_dir is not None:
        save_dir = Path(output_dir).resolve()
    else:
        save_dir = Path(__file__).resolve().parent / "outputs" / system / "sweep_ofo_parameters"
    # Don't mkdir yet — main() may append _1d/_2d suffix before creating

    # ── Load shared data (done once) ─────────────────────────────────────

    logger.info("Loading inference data...")
    inference_data = InferenceData.ensure(
        data_dir,
        all_models,
        data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir,
        plot=False,
        dt_s=data_dt,
    )

    logger.info("Loading training trace...")
    training_trace = TrainingTrace.ensure(data_dir / "training_trace.csv", config.training_trace_params)

    logger.info("Loading logistic fits...")
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv",
        all_models,
        data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir,
        plot=False,
    )

    # ── Build baseline OFOConfig ─────────────────────────────────────────

    ofo_params = config.ofo
    baseline_ofo = OFOConfig(
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

    logger.info("Baseline OFO config: %s", baseline_ofo)

    # ── Build shared DC sites, datacenters, and grid ─────────────────────

    assert config.dc_sites is not None
    site_ids = list(config.dc_sites.keys())
    dc_loads: dict[str, DCLoadSpec] = {}
    datacenters: dict[str, OfflineDatacenter] = {}
    site_models_map: dict[str, tuple[InferenceModelSpec, ...]] = {}
    primary_bus = ""

    for site_id, site_cfg in config.dc_sites.items():
        site_models = _resolve_models_for_site(site_cfg, all_models)
        site_models_map[site_id] = site_models
        site_inference = inference_data.filter_models(site_models) if site_cfg.models is not None else inference_data

        dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=site_cfg.base_kw_per_phase)

        workload_kwargs: dict = {"inference_data": site_inference}
        if config.training is not None:
            tc = config.training
            workload_kwargs["training"] = TrainingRun(
                n_gpus=tc.n_gpus,
                trace=training_trace,
                target_peak_W_per_gpu=tc.target_peak_W_per_gpu,
            ).at(t_start=tc.t_start, t_end=tc.t_end)
        if config.inference_ramp is not None:
            rc = config.inference_ramp
            workload_kwargs["inference_ramps"] = InferenceRamp(target=rc.target, model=rc.model).at(
                t_start=rc.t_start,
                t_end=rc.t_end,
            )
        workload = OfflineWorkload(**workload_kwargs)

        dc = OfflineDatacenter(
            dc_config,
            workload,
            dt_s=dt_dc,
            seed=site_cfg.seed,
            power_augmentation=config.power_augmentation,
        )
        datacenters[site_id] = dc
        dc_loads[site_id] = DCLoadSpec(
            bus=site_cfg.bus,
            bus_kv=site_cfg.bus_kv,
            connection_type=site_cfg.connection_type,
        )
        if not primary_bus:
            primary_bus = site_cfg.bus

    site_bus_map = {sid: config.dc_sites[sid].bus for sid in config.dc_sites}
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

    return dict(
        config=config,
        sim=sim,
        all_models=all_models,
        baseline_ofo=baseline_ofo,
        logistic_models=logistic_models,
        site_ids=site_ids,
        site_models_map=site_models_map,
        site_bus_map=site_bus_map,
        datacenters=datacenters,
        grid=grid,
        primary_bus=primary_bus,
        exclude_buses=exclude_buses,
        save_dir=save_dir,
        dt_ctrl=dt_ctrl,
        system=system,
    )


# ── 1-D sweep (single DC site) ──────────────────────────────────────────────


def _run_single_sim(
    *,
    config,
    sim,
    site_ofo_cfgs: dict[str, OFOConfig],
    logistic_models,
    site_models_map,
    site_bus_map,
    datacenters,
    grid,
    primary_bus,
    exclude_buses,
    dt_ctrl,
    all_models,
    run_dir: Path,
):
    """Run one simulation with per-site OFO configs. Returns (log, wall_time_s)."""
    controllers = []
    for site_id, _site_cfg in config.dc_sites.items():
        site_models = site_models_map[site_id]
        ofo_cfg = site_ofo_cfgs[site_id]
        ofo_ctrl = OFOBatchSizeController(
            site_models,
            models=logistic_models,
            config=ofo_cfg,
            dt_s=dt_ctrl,
            site_id=site_id if len(config.dc_sites) > 1 else None,
        )
        controllers.append(ofo_ctrl)

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

    t0 = time.monotonic()
    log = coord.run()
    wall_time_s = time.monotonic() - t0

    _save_plots(
        log,
        run_dir=run_dir,
        site_models_map=site_models_map,
        site_bus_map=site_bus_map,
        v_min=sim.v_min,
        v_max=sim.v_max,
        exclude_buses=exclude_buses,
    )

    return log, wall_time_s


def _run_sweep_1d(ctx: dict) -> None:
    """One-at-a-time sweep: same OFO config for all sites."""
    runs = build_sweep_grid(ctx["baseline_ofo"])
    total = len(runs)
    save_dir = ctx["save_dir"]
    sim = ctx["sim"]
    all_models = ctx["all_models"]
    system = ctx["system"]

    logger.info("1-D sweep: %d runs across %d parameters", total, len(DEFAULT_SWEEP_MULTIPLIERS))

    rows: list[dict] = []
    for i, (param_name, param_value, ofo_cfg) in enumerate(runs, start=1):
        rid = _run_id(param_name, param_value)
        run_dir = save_dir / rid
        logger.info("[%d/%d] %s = %s", i, total, param_name, param_value)

        try:
            # All sites use the same OFO config in 1-D mode
            site_ofo_cfgs = {sid: ofo_cfg for sid in ctx["site_ids"]}
            log, wall_time_s = _run_single_sim(
                config=ctx["config"],
                sim=sim,
                site_ofo_cfgs=site_ofo_cfgs,
                logistic_models=ctx["logistic_models"],
                site_models_map=ctx["site_models_map"],
                site_bus_map=ctx["site_bus_map"],
                datacenters=ctx["datacenters"],
                grid=ctx["grid"],
                primary_bus=ctx["primary_bus"],
                exclude_buses=ctx["exclude_buses"],
                dt_ctrl=ctx["dt_ctrl"],
                all_models=all_models,
                run_dir=run_dir,
            )

            row = _collect_metrics(
                log,
                models=all_models,
                wall_time_s=wall_time_s,
                param_name=param_name,
                param_value=param_value,
                v_min=sim.v_min,
                v_max=sim.v_max,
                exclude_buses=ctx["exclude_buses"],
            )
            rows.append(row)
            logger.info(
                "  -> violation_time=%.1fs  integral_viol=%.4f pu·s  worst_vmin=%.4f  wall=%.1fs",
                row["violation_time_s"],
                row["integral_violation_pu_s"],
                row["worst_vmin"],
                row["wall_time_s"],
            )
        except Exception:
            logger.exception("Run %s failed; skipping.", rid)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = save_dir / f"results_{system}_sweep_ofo_parameters.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved %d rows to %s", len(df), csv_path)
        print(df.to_string(index=False))

        model_labels = [m.model_label for m in all_models]
        _plot_sweep_summary(df, save_dir, model_labels)
    else:
        logger.warning("No successful runs; no CSV written.")

    logger.info("All outputs in: %s", save_dir)


# ── 2-D sweep (multiple DC sites) ───────────────────────────────────────────


def _run_sweep_2d(ctx: dict) -> None:
    """Per-site parameter sweep: independent values for each site."""
    site_ids = ctx["site_ids"]
    runs = build_sweep_grid_2d(ctx["baseline_ofo"], site_ids)
    total = len(runs)
    save_dir = ctx["save_dir"]
    sim = ctx["sim"]
    all_models = ctx["all_models"]
    system = ctx["system"]

    logger.info(
        "2-D sweep: %d runs across %d parameters, %d sites (%s)",
        total,
        len(DEFAULT_SWEEP_MULTIPLIERS),
        len(site_ids),
        ", ".join(site_ids),
    )

    rows: list[dict] = []
    for i, (param_name, site_values, site_ofo_cfgs) in enumerate(runs, start=1):
        rid = _run_id_2d(param_name, site_values)
        run_dir = save_dir / rid
        vals_str = ", ".join(f"{sid}={v:.6g}" for sid, v in site_values.items())
        logger.info("[%d/%d] %s: %s", i, total, param_name, vals_str)

        try:
            log, wall_time_s = _run_single_sim(
                config=ctx["config"],
                sim=sim,
                site_ofo_cfgs=site_ofo_cfgs,
                logistic_models=ctx["logistic_models"],
                site_models_map=ctx["site_models_map"],
                site_bus_map=ctx["site_bus_map"],
                datacenters=ctx["datacenters"],
                grid=ctx["grid"],
                primary_bus=ctx["primary_bus"],
                exclude_buses=ctx["exclude_buses"],
                dt_ctrl=ctx["dt_ctrl"],
                all_models=all_models,
                run_dir=run_dir,
            )

            row = _collect_metrics(
                log,
                models=all_models,
                wall_time_s=wall_time_s,
                param_name=param_name,
                param_value=0.0,  # placeholder
                v_min=sim.v_min,
                v_max=sim.v_max,
                exclude_buses=ctx["exclude_buses"],
            )
            # Add per-site parameter values
            for sid, val in site_values.items():
                row[f"param_value__{sid}"] = val
            # Remove the flat param_value (not meaningful for 2-D)
            row.pop("param_value", None)
            rows.append(row)

            logger.info(
                "  -> violation_time=%.1fs  integral_viol=%.4f pu·s  worst_vmin=%.4f  wall=%.1fs",
                row["violation_time_s"],
                row["integral_violation_pu_s"],
                row["worst_vmin"],
                row["wall_time_s"],
            )
        except Exception:
            logger.exception("Run %s failed; skipping.", rid)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = save_dir / f"results_{system}_sweep_ofo_parameters_2d.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved %d rows to %s", len(df), csv_path)
        print(df.to_string(index=False))

        model_labels = [m.model_label for m in all_models]
        _plot_sweep_summary_2d(df, save_dir, site_ids, model_labels)
    else:
        logger.warning("No successful runs; no CSV written.")

    logger.info("All outputs in: %s", save_dir)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(
    *,
    config_path: Path,
    system: str,
    sweep_mode: Literal["auto", "1d", "2d"] = "auto",
    dt_override: str | None = None,
    output_dir: str | None = None,
) -> None:
    ctx = _setup(
        config_path=config_path,
        system=system,
        dt_override=dt_override,
        output_dir=output_dir,
    )

    n_sites = len(ctx["site_ids"])

    if sweep_mode == "auto":
        use_2d = n_sites >= 2
    elif sweep_mode == "2d":
        use_2d = True
    else:
        use_2d = False

    if use_2d and n_sites < 2:
        logger.warning("2-D sweep requested but only %d DC site(s); falling back to 1-D.", n_sites)
        use_2d = False

    # Append _1d or _2d to output directory if using default path
    if output_dir is None:
        suffix = "_2d" if use_2d else "_1d"
        ctx["save_dir"] = ctx["save_dir"].parent / (ctx["save_dir"].name + suffix)
    ctx["save_dir"].mkdir(parents=True, exist_ok=True)

    if use_2d:
        logger.info("%d DC sites, sweep mode: 2-D (independent parameters per site)", n_sites)
        _run_sweep_2d(ctx)
    else:
        logger.info("%d DC site(s), sweep mode: 1-D (shared parameters across sites)", n_sites)
        _run_sweep_1d(ctx)


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        config: str
        """Path to the config JSON file."""
        system: str = "ieee13"
        """System name (e.g. ieee13, ieee34, ieee123). Used for output directory and CSV naming."""
        sweep_mode: Literal["auto", "1d", "2d"] = "auto"
        """Sweep mode: 'auto' selects based on number of DC sites (1 site -> 1-D, 2+ sites -> 2-D).
        '1d' forces one-at-a-time sweep with shared parameters across all sites.
        '2d' forces independent per-site parameter sweep (requires 2+ DC sites)."""
        dt: str | None = None
        """Override simulation time step (e.g. '60' for 60s resolution)."""
        output_dir: str | None = None
        """Override output directory (default: outputs/<system>/sweep_ofo_parameters/)."""
        log_level: str = "INFO"
        """Logging verbosity (DEBUG, INFO, WARNING)."""

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    main(
        config_path=Path(args.config),
        system=args.system,
        sweep_mode=args.sweep_mode,
        dt_override=args.dt,
        output_dir=args.output_dir,
    )
