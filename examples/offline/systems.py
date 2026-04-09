"""IEEE test feeder constants.

Feeder-specific constants (DSS paths, regulator names, initial taps,
excluded buses, source voltage, bus voltage levels) are defined here.

Experiment-specific parameters (datacenter sizing, controller tuning,
workload scenarios, model specs, simulation defaults) belong in
``experiment.py`` or inline in each example script.
"""

from __future__ import annotations

from pathlib import Path

from openg2g.grid.config import TapPosition

GRID_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "grid"

TAP_STEP = 0.00625  # Standard 32-step regulator: ±10% in 32 steps


def tap(steps: int) -> float:
    """Convert integer tap step to per-unit ratio. E.g., ``tap(14)`` -> 1.0875."""
    return 1.0 + steps * TAP_STEP


# IEEE test feeder constants


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
