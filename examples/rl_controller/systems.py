"""IEEE test feeder constants, shared simulation constants, model-spec presets.

Feeder definitions (`ieee13`, `ieee34`, `ieee123`), the `SYSTEMS` lookup
dict, the regulator-tap helper (`tap`, `TAP_STEP`), shared simulation
constants (`DT_*`, `V_MIN`/`V_MAX`, `TOTAL_DURATION_S`, `POWER_AUG`),
hardcoded `InferenceModelSpec` presets (`LLAMA_*`, `QWEN_*`,
`MODEL_SPECS`), and the `deploy` / `with_ramp` deployment shortcuts. The
RL-specific scenario randomization, experiment factories, and
`ScenarioOpenDSSGrid` live in `scenarios.py`.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

from openg2g.datacenter.config import (
    InferenceModelSpec,
    ModelDeployment,
    PowerAugmentationConfig,
    ReplicaSchedule,
)
from openg2g.grid.config import TapPosition

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
