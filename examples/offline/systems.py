"""IEEE test feeder constants, shared simulation constants, and data pipeline helpers.

Feeder-specific constants (DSS paths, regulator names, initial taps,
excluded buses, source voltage, bus voltage levels) are defined here.
Shared infrastructure (PV/load profile generators, ScenarioOpenDSSGrid,
experiment configuration factories, scenario randomisation) also lives here
so that application scripts never become implicit libraries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from openg2g.controller.ofo import OFOConfig
from openg2g.datacenter.config import (
    InferenceModelSpec,
    ModelDeployment,
    PowerAugmentationConfig,
    ReplicaSchedule,
    TrainingRun,
)
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.config import TapPosition, TapSchedule
from openg2g.grid.opendss import OpenDSSGrid

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GRID_DATA_DIR = PROJECT_ROOT / "data" / "grid"

TAP_STEP = 0.00625  # Standard 32-step regulator: ±10% in 32 steps


def tap(steps: int) -> float:
    """Convert integer tap step to per-unit ratio. E.g., ``tap(14)`` → 1.0875."""
    return 1.0 + steps * TAP_STEP


# ── Shared simulation constants ──────────────────────────────────────────────

DT_DC = Fraction(1)
DT_GRID = Fraction(1)
DT_CTRL = Fraction(1)
V_MIN, V_MAX = 0.95, 1.05
TOTAL_DURATION_S = 3600
POWER_AUG = PowerAugmentationConfig(amplitude_scale_range=(0.98, 1.02), noise_fraction=0.005)


# ── DC site descriptor ───────────────────────────────────────────────────────


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
        connection_type: Grid connection type (``"wye"`` or ``"delta"``).
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


# ── Model specs (hardcoded; replaces config.json which master removed) ──────

LLAMA_8B = InferenceModelSpec(
    model_label="Llama-3.1-8B",
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    gpu_model="H100",
    task="lm-arena-chat",
    precision="bfloat16",
    gpus_per_replica=1,
    tensor_parallel=1,
    itl_deadline_s=0.08,
    batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024),
    feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
)
LLAMA_70B = InferenceModelSpec(
    model_label="Llama-3.1-70B",
    model_id="meta-llama/Llama-3.1-70B-Instruct",
    gpu_model="H100",
    task="lm-arena-chat",
    precision="bfloat16",
    gpus_per_replica=4,
    tensor_parallel=4,
    itl_deadline_s=0.10,
    batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048),
    feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
)
LLAMA_405B = InferenceModelSpec(
    model_label="Llama-3.1-405B",
    model_id="meta-llama/Llama-3.1-405B-Instruct-FP8",
    gpu_model="H100",
    task="lm-arena-chat",
    precision="fp8",
    gpus_per_replica=8,
    tensor_parallel=8,
    itl_deadline_s=0.12,
    batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256, 384, 512),
    feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
)
QWEN_30B = InferenceModelSpec(
    model_label="Qwen3-30B-A3B",
    model_id="Qwen/Qwen3-30B-A3B-Thinking-2507",
    gpu_model="H100",
    task="gpqa",
    precision="bfloat16",
    gpus_per_replica=2,
    tensor_parallel=2,
    itl_deadline_s=0.06,
    batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256, 384, 512),
    feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
)
QWEN_235B = InferenceModelSpec(
    model_label="Qwen3-235B-A22B",
    model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
    gpu_model="H100",
    task="gpqa",
    precision="bfloat16",
    gpus_per_replica=8,
    tensor_parallel=8,
    itl_deadline_s=0.14,
    batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256, 384, 512),
    feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
)
ALL_MODEL_SPECS: tuple[InferenceModelSpec, ...] = (LLAMA_8B, LLAMA_70B, LLAMA_405B, QWEN_30B, QWEN_235B)
MODEL_SPECS: dict[str, InferenceModelSpec] = {s.model_label: s for s in ALL_MODEL_SPECS}

SPECS_CACHE_DIR = PROJECT_ROOT / "data" / "specs"
TRAINING_TRACE_PATH = PROJECT_ROOT / "data" / "training_trace.csv"


def model_spec(label: str) -> InferenceModelSpec:
    """Get a single model spec by label."""
    return MODEL_SPECS[label]


def all_model_specs() -> tuple[InferenceModelSpec, ...]:
    """Return all model specs as a tuple (for data pipeline loading)."""
    return ALL_MODEL_SPECS


def deploy(
    label: str,
    num_replicas: int,
    initial_batch_size: int = 128,
) -> tuple[ModelDeployment, ReplicaSchedule]:
    """Shorthand: ``deploy("Llama-3.1-8B", 720, 128)`` -> (ModelDeployment, ReplicaSchedule)."""
    return (
        ModelDeployment(spec=MODEL_SPECS[label], initial_batch_size=initial_batch_size),
        ReplicaSchedule(initial=num_replicas),
    )


def with_ramp(
    deployment: tuple[ModelDeployment, ReplicaSchedule],
    target: int,
    *,
    t_start: float,
    t_end: float,
) -> tuple[ModelDeployment, ReplicaSchedule]:
    """Inject a single ramp into a deploy() result. Convenience for ieee*_experiment factories."""
    md, sched = deployment
    return (md, sched.ramp_to(target, t_start=t_start, t_end=t_end))


def load_data_sources() -> tuple[InferenceData, None, Path]:
    """Compatibility shim — returns (inference_data, None, data_dir).

    On master, the per-spec content-addressed cache replaced the JSON-driven
    pipeline, so callers should ideally use ``InferenceData.ensure(SPECS_CACHE_DIR,
    ALL_MODEL_SPECS, ...)`` directly. This shim preserves the 3-tuple shape for
    in-flight callers while we migrate them.
    """
    inference_data = InferenceData.ensure(SPECS_CACHE_DIR, ALL_MODEL_SPECS, plot=False)
    return inference_data, None, SPECS_CACHE_DIR


# ── PV / time-varying load specs ────────────────────────────────────────────


@dataclass
class PVSystemSpec:
    """PV system at a distribution bus (used by ScenarioOpenDSSGrid)."""

    bus: str
    bus_kv: float = 4.16
    peak_kw: float = 1000.0
    csv_path: Path | None = None
    power_factor: float = 1.0
    # Optional shape randomness applied inside eval_profile:
    #   t_eff = (t - peak_t_shift_s) / time_warp
    # Defaults are no-op so existing callers are unaffected.
    peak_t_shift_s: float = 0.0
    time_warp: float = 1.0
    # Optional randomized profile selector. When profile_kind != "default" the
    # eval_profile dispatcher uses the matching generator in
    # sweep_dc_locations.py and ignores csv_path / profile_fn / time_warp /
    # peak_t_shift_s. Valid kinds: "default", "random_flat" (PV).
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
    # Valid profile_kind for TVL: "default", "random_shape" (with shape param).
    profile_kind: str = "default"
    profile_params: dict | None = None


# ── IEEE test feeder constants ───────────────────────────────────────────────


def ieee13() -> dict:
    """IEEE 13-bus test feeder constants."""
    return dict(
        dss_case_dir=GRID_DATA_DIR / "ieee13",
        dss_master_file="IEEE13Bus.dss",
        bus_kv=4.16,
        source_pu=1.0,
        initial_taps=TapPosition(
            regulators={
                "creg1a": tap(14),
                "creg1b": tap(6),
                "creg1c": tap(15),
            }
        ),
        exclude_buses=("sourcebus", "650", "rg60"),
    )


def ieee34() -> dict:
    """IEEE 34-bus (half-line variant) test feeder constants."""
    return dict(
        dss_case_dir=GRID_DATA_DIR / "ieee34",
        dss_master_file="IEEE34Bus.dss",
        bus_kv=24.9,
        source_pu=1.09,
        initial_taps=TapPosition(
            regulators={
                "creg1a": tap(11),
                "creg1b": tap(6),
                "creg1c": tap(8),
                "creg2a": tap(8),
                "creg2b": tap(8),
                "creg2c": tap(8),
            }
        ),
        exclude_buses=(
            "sourcebus",
            "800",
            "802",
            "806",
            "808",
            "810",
            "812",
            "814",
            "888",
            "890",
        ),
        regulator_zones={
            "creg1": ["814r", "850", "816", "824", "828", "830", "854"],
            "creg2": [
                "852r",
                "832",
                "858",
                "834",
                "860",
                "836",
                "840",
                "862",
                "842",
                "844",
                "846",
                "848",
            ],
        },
    )


def ieee123() -> dict:
    """IEEE 123-bus test feeder constants."""
    return dict(
        dss_case_dir=GRID_DATA_DIR / "ieee123",
        dss_master_file="IEEE123Bus.dss",
        bus_kv=4.16,
        source_pu=1.0,
        initial_taps=TapPosition(
            regulators={
                "creg1a": tap(9),
                "creg2a": tap(5),
                "creg3a": tap(5),
                "creg3c": tap(5),
                "creg4a": tap(14),
                "creg4b": tap(1),
                "creg4c": tap(4),
            }
        ),
        exclude_buses=(
            "sourcebus",
            "150",
            "150r",
            "149",
            "9r",
            "25r",
            "160r",
            "61s",
            "610",
            "300_open",
            "94_open",
            "135",
        ),
        zones={
            "z1_sw": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "34",
            ],
            "z2_nw": [
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
                "32",
                "33",
                "35",
                "36",
                "37",
                "38",
                "39",
                "40",
                "41",
                "42",
                "43",
                "44",
                "45",
                "46",
                "47",
                "48",
                "49",
                "50",
                "51",
            ],
            "z3_se": [
                "52",
                "53",
                "54",
                "55",
                "56",
                "57",
                "58",
                "59",
                "60",
                "61",
                "62",
                "63",
                "64",
                "65",
                "66",
                "67",
                "68",
                "69",
                "70",
                "71",
                "72",
                "73",
                "74",
                "75",
                "76",
                "77",
                "78",
                "79",
                "80",
                "81",
                "82",
                "83",
                "84",
                "85",
                "86",
                "87",
                "88",
                "89",
                "90",
                "91",
                "92",
                "93",
                "94",
                "95",
                "96",
            ],
            "z4_ne": [
                "97",
                "98",
                "99",
                "100",
                "101",
                "102",
                "103",
                "104",
                "105",
                "106",
                "107",
                "108",
                "109",
                "110",
                "111",
                "112",
                "113",
                "114",
                "115",
                "116",
                "117",
                "118",
                "119",
                "120",
                "121",
                "122",
                "123",
            ],
        },
    )


SYSTEMS = {"ieee13": ieee13, "ieee34": ieee34, "ieee123": ieee123}


# ── Profile helpers ───────────────────────────────────────────────────────────


def _smooth_bump(t: float, t_center: float, half_width: float) -> float:
    dt = abs(t - t_center)
    if dt >= half_width:
        return 0.0
    x = dt / half_width
    return (1 - x * x) ** 2


def _smoothstep(t: float, t_start: float, t_end: float) -> float:
    """Cubic Hermite smoothstep: zero-derivative at both endpoints."""
    if t_end <= t_start:
        return 1.0 if t >= t_start else 0.0
    if t <= t_start:
        return 0.0
    if t >= t_end:
        return 1.0
    x = (t - t_start) / (t_end - t_start)
    return x * x * (3.0 - 2.0 * x)


def _irregular_fluct(t: float, seed: float = 0.0) -> float:
    """Irregular fluctuation via superposition of incommensurate frequencies."""
    s = seed
    f1 = 0.06 * math.sin(2 * math.pi * t / 173.0 + s)
    f2 = 0.05 * math.sin(2 * math.pi * t / 97.3 + s * 2.3)
    f3 = 0.04 * math.sin(2 * math.pi * t / 251.7 + s * 0.7)
    f4 = 0.03 * math.sin(2 * math.pi * t / 41.9 + s * 4.1)
    f5 = 0.02 * math.sin(2 * math.pi * t / 317.3 + s * 1.9)
    return 1.0 + f1 + f2 + f3 + f4 + f5


def pv_profile_kw(t: float, peak_kw: float, site_idx: int = 0) -> float:
    """Solar PV output (kW per phase) with per-site cloud patterns."""
    T = TOTAL_DURATION_S
    if site_idx == 0:
        trend = 0.85 - 0.30 * (t / T)
        cloud = 1.0
        cloud -= 0.55 * _smooth_bump(t, 600, 120)
        cloud -= 0.40 * _smooth_bump(t, 2100, 180)
        fluct = _irregular_fluct(t, seed=0.3)
        return max(0.0, peak_kw * trend * max(cloud, 0.05) * fluct)
    elif site_idx == 1:
        ramp = 0.55 + 0.40 * _smooth_bump(t, 1200, 900)
        cloud = 1.0
        cloud -= 0.60 * _smooth_bump(t, 1680, 240)
        cloud -= 0.25 * _smooth_bump(t, 2400, 150)
        fluct = _irregular_fluct(t, seed=2.1)
        return max(0.0, peak_kw * ramp * max(cloud, 0.05) * fluct)
    else:
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
    """Evaluate a PV/TVL profile at simulated time ``t``.

    Dispatch order:
      1. ``profile_kind="random_flat"`` + ``profile_params`` -> pv_profile_random
      2. ``profile_kind="random_shape"`` + ``profile_params`` -> tvl_profile_random
      3. ``csv_data`` not None -> CSV interpolation
      4. ``profile_fn`` (legacy analytical pv_profile_kw / load_profile_kw)
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


# ── Scenario grid ─────────────────────────────────────────────────────────────


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


# ── Scenario randomisation helpers ────────────────────────────────────────────


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


def materialize_scenario(rec, *, training_base: dict | None) -> dict:
    """Return the per-episode scenario dict directly from a ``ScenarioRecord``,
    without rng replay.

    Mirrors the shape returned by :func:`randomize_scenario` so consumers
    (eval pipeline, training make_sim) can swap in transparently. The record
    is expected to carry resolved ``dc_sites`` / ``pv_systems`` / ``tvl`` —
    libraries built before this field set need to be upgraded with
    ``examples/offline/migrate_scenario_library.py``.
    """
    if getattr(rec, "resolved_dc_sites", None) is None:
        raise RuntimeError(
            f"ScenarioRecord(seed={rec.seed}) is missing resolved fields. "
            "Run examples/offline/migrate_scenario_library.py on the library."
        )
    overlay = getattr(rec, "training_overlay", None)
    training_run = None
    if overlay is not None and training_base is not None and training_base.get("trace") is not None:
        training_run = TrainingRun(
            n_gpus=overlay["n_gpus"],
            trace=training_base["trace"],
            target_peak_W_per_gpu=overlay["target_peak_W_per_gpu"],
        ).at(t_start=overlay["t_start"], t_end=overlay["t_end"])
    return {
        "seed": int(rec.seed),
        "dc_sites": rec.resolved_dc_sites,
        "pv_systems": list(rec.resolved_pv_systems),
        "tvl": list(rec.resolved_tvl),
        "training_run": training_run,
        "params": {
            "pv_scale": rec.pv_scale,
            "load_scale": rec.load_scale,
            "training_overlay": overlay,
        },
    }


# ── Experiment factories (shared by train_ppo, build_scenario_library, evaluate_controllers) ──


def ieee13_experiment(training_trace: TrainingTrace | None = None) -> dict:
    """IEEE 13-bus: single DC at bus 671 with 5 LLM models."""
    sys = SYSTEMS["ieee13"]()
    # Inject ramps into each model's ReplicaSchedule (was a single OR-chained
    # InferenceRampSchedule on the old API).
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


EXPERIMENTS: dict[str, object] = {
    "ieee13": ieee13_experiment,
    "ieee34": ieee34_experiment,
    "ieee123": ieee123_experiment,
}
