"""Shared analysis utilities for offline example scripts."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from openg2g.grid.base import GridState


def discover_candidate_buses(
    case_dir: Path,
    master_file: str,
    target_bus_kv: float,
    exclude: set[str],
) -> list[str]:
    """Discover all 3-phase buses at the target voltage level.

    Opens the DSS case, solves once, and returns bus names with all three
    phases at the expected voltage level.
    """
    from opendssdirect import dss as dss_local

    dss_local.Basic.ClearAll()
    master_path = str(Path(case_dir) / master_file)
    dss_local.Text.Command(f'Compile "{master_path}"')
    dss_local.Solution.SolveNoControl()

    target_kv_ln = target_bus_kv / math.sqrt(3.0)
    tolerance = 0.05 * target_kv_ln

    bus_phases: dict[str, set[int]] = {}
    for name in dss_local.Circuit.AllNodeNames():
        parts = name.split(".")
        bus = parts[0].lower()
        phase = int(parts[1]) if len(parts) > 1 else 0
        if bus not in bus_phases:
            bus_phases[bus] = set()
        bus_phases[bus].add(phase)

    exclude_lower = {b.lower() for b in exclude}
    candidates = []

    for bus_name in dss_local.Circuit.AllBusNames():
        if bus_name.lower() in exclude_lower:
            continue
        phases = bus_phases.get(bus_name.lower(), set())
        if not {1, 2, 3}.issubset(phases):
            continue
        dss_local.Circuit.SetActiveBus(bus_name)
        kv_base = dss_local.Bus.kVBase()
        if abs(kv_base - target_kv_ln) <= tolerance:
            candidates.append(bus_name)

    return sorted(candidates)


def extract_all_voltages(
    grid_states: list[GridState],
    exclude_buses: tuple[str, ...] = (),
) -> dict[str, dict[str, list[float]]]:
    """Extract per-bus, per-phase voltage time series from grid states.

    Returns a dict mapping bus name to `{"a": [...], "b": [...], "c": [...]}`.
    """
    exclude = {b.lower() for b in exclude_buses}
    result: dict[str, dict[str, list[float]]] = {}
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


def find_violations(
    voltages_dict: dict[str, dict[str, list[float]]],
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> list[tuple[str, str, str, float, float]]:
    """Find voltage violations in extracted voltage time series.

    Returns a list of `(bus, phase, type, value, deviation)` tuples
    where *type* is `"under"` or `"over"`.
    """
    violations: list[tuple[str, str, str, float, float]] = []
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
