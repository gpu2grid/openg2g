"""IEEE test feeder constants, shared simulation constants, and data pipeline helpers.

Feeder-specific constants (DSS paths, regulator names, initial taps,
excluded buses, source voltage, bus voltage levels) are defined here.
Experiment-specific parameters (datacenter sizing, controller tuning,
workload scenarios, PV/load profiles) belong inline in each example script.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from openg2g import PROJECT_ROOT
from openg2g.datacenter.config import (
    InferenceModelSpec,
    InferenceRampSchedule,
    ModelDeployment,
    PowerAugmentationConfig,
)
from openg2g.datacenter.workloads.inference import MLEnergySource
from openg2g.datacenter.workloads.training import TrainingTraceParams
from openg2g.grid.config import TapPosition

GRID_DATA_DIR = PROJECT_ROOT / "data" / "grid"

TAP_STEP = 0.00625  # Standard 32-step regulator: ±10% in 32 steps


def tap(steps: int) -> float:
    """Convert integer tap step to per-unit ratio. E.g., ``tap(14)`` → 1.0875."""
    return 1.0 + steps * TAP_STEP


# ── Shared simulation constants ──────────────────────────────────────────────

DT_DC = Fraction(1, 10)
DT_GRID = Fraction(1, 10)
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
        models: Model deployments (spec + replica count) at this site.
        seed: Random seed for layout generation.
        connection_type: Grid connection type (``"wye"`` or ``"delta"``).
        inference_ramps: Ramp schedule for inference servers.
        load_shift_headroom: Fraction of extra server capacity for load shifting.
    """

    bus: str
    bus_kv: float
    base_kw_per_phase: float
    total_gpu_capacity: int
    models: tuple[ModelDeployment, ...] = ()
    seed: int = 0
    connection_type: str = "wye"
    inference_ramps: InferenceRampSchedule | None = None
    load_shift_headroom: float = 0.0


# ── Data pipeline ────────────────────────────────────────────────────────────


def load_data_sources(
    config_path: Path | None = None,
) -> tuple[dict[str, MLEnergySource], TrainingTraceParams, Path]:
    """Load ML.ENERGY data sources from ``config.json``.

    Returns:
        (data_sources, training_trace_params, data_dir) where *data_dir* is a
        hash-based cache directory under ``PROJECT_ROOT / "data" / "offline"``.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    sources_raw = cfg["data_sources"]
    data_sources = {s["model_label"]: MLEnergySource(**s) for s in sources_raw}
    ttp = TrainingTraceParams(**(cfg.get("training_trace_params") or {}))

    blob = json.dumps(
        (sorted(sources_raw, key=lambda s: s["model_label"]), cfg.get("training_trace_params") or {}),
        sort_keys=True,
    ).encode()
    data_dir = PROJECT_ROOT / "data" / "offline" / hashlib.sha256(blob).hexdigest()[:16]

    return data_sources, ttp, data_dir


# ── Model specs (loaded from config.json) ───────────────────────────────────

_MODEL_SPECS: dict[str, InferenceModelSpec] = {}


def load_model_specs(config_path: Path | None = None) -> dict[str, InferenceModelSpec]:
    """Load model specs from ``config.json``.

    Returns a dict mapping ``model_label`` to :class:`InferenceModelSpec`.
    Results are cached after the first call.
    """
    global _MODEL_SPECS
    if _MODEL_SPECS:
        return _MODEL_SPECS
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    for raw in cfg["model_specs"]:
        spec = InferenceModelSpec(**raw)
        _MODEL_SPECS[spec.model_label] = spec
    return _MODEL_SPECS


def model_spec(label: str, config_path: Path | None = None) -> InferenceModelSpec:
    """Get a single model spec by label.

    Loads from ``config.json`` on first call, then caches.
    """
    specs = load_model_specs(config_path)
    return specs[label]


def all_model_specs(config_path: Path | None = None) -> tuple[InferenceModelSpec, ...]:
    """Return all model specs as a tuple (for data pipeline loading)."""
    return tuple(load_model_specs(config_path).values())


def deploy(label: str, num_replicas: int, initial_batch_size: int = 128) -> ModelDeployment:
    """Shorthand: ``deploy("Llama-3.1-8B", 720, 128)``."""
    return ModelDeployment(spec=model_spec(label), num_replicas=num_replicas, initial_batch_size=initial_batch_size)


# ── PV / time-varying load specs ────────────────────────────────────────────


@dataclass
class PVSystemSpec:
    """PV system at a distribution bus (used by ScenarioOpenDSSGrid)."""

    bus: str
    bus_kv: float = 4.16
    peak_kw: float = 1000.0
    csv_path: Path | None = None
    power_factor: float = 1.0


@dataclass
class TimeVaryingLoadSpec:
    """Time-varying load at a distribution bus (used by ScenarioOpenDSSGrid)."""

    bus: str
    bus_kv: float = 4.16
    peak_kw: float = 500.0
    csv_path: Path | None = None
    power_factor: float = 0.96


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
                "creg1a": tap(12),
                "creg1b": tap(8),
                "creg1c": tap(10),
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
