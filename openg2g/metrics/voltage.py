"""Voltage violation metrics for all-bus, all-phase analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openg2g.grid.base import GridState


@dataclass
class VoltageStats:
    """Summary voltage statistics over a simulation run.

    Attributes:
        worst_vmin: Lowest voltage observed across all buses and phases (pu).
        worst_vmax: Highest voltage observed across all buses and phases (pu).
        violation_time_s: Total time with at least one bus-phase violating
            voltage bounds (seconds).
        integral_violation_pu_s: Integrated voltage violation magnitude
            across all bus-phase pairs (pu * s).
    """

    worst_vmin: float
    worst_vmax: float
    violation_time_s: float
    integral_violation_pu_s: float


def compute_allbus_voltage_stats(
    grid_states: list[GridState],
    *,
    v_min: float = 0.95,
    v_max: float = 1.05,
    exclude_buses: tuple[str, ...] = ("rg60",),
) -> VoltageStats:
    """Compute voltage violation statistics across all buses and phases.

    For each snapshot the integral violation sums
    `max(v_min - v, 0) + max(v - v_max, 0)` over every non-excluded
    bus-phase pair, then integrates over time.  A snapshot counts as
    "violated" when this sum is positive.

    Args:
        grid_states: Sequence of [`GridState`][openg2g.grid.base.GridState]
            objects from a simulation run.
        v_min: Lower voltage bound (pu).
        v_max: Upper voltage bound (pu).
        exclude_buses: Bus names to exclude from statistics (case-insensitive).
    """
    if len(grid_states) < 2:
        raise ValueError(
            f"At least two grid states are required to compute voltage statistics (got {len(grid_states)})."
        )

    times = np.array([gs.time_s for gs in grid_states], dtype=float)
    dt = float(np.median(np.diff(times)))

    # Collect bus-phase columns from the first snapshot (all snapshots
    # share the same set of buses for a given OpenDSS circuit).
    exclude = {b.lower() for b in exclude_buses}
    bus_names = [b for b in grid_states[0].voltages.buses() if b.lower() not in exclude]

    # Build (T, N) voltage matrix where N = num_buses * 3.
    T = len(grid_states)
    N = len(bus_names) * 3
    V = np.empty((T, N), dtype=float)
    for t, gs in enumerate(grid_states):
        col = 0
        for bus in bus_names:
            tp = gs.voltages[bus]
            V[t, col] = tp.a
            V[t, col + 1] = tp.b
            V[t, col + 2] = tp.c
            col += 3

    valid = ~np.isnan(V)
    worst_vmin = float(np.min(np.where(valid, V, np.inf)))
    worst_vmax = float(np.max(np.where(valid, V, -np.inf)))

    # Per-timestep violation: sum over all bus-phase pairs
    viol = np.where(valid, np.maximum(v_min - V, 0.0) + np.maximum(V - v_max, 0.0), 0.0)
    viol_sum = np.sum(viol, axis=1)  # shape (T,)

    violation_steps = int(np.count_nonzero(viol_sum > 0.0))
    integral_violation = float(np.sum(viol_sum * dt))

    return VoltageStats(
        worst_vmin=float(worst_vmin),
        worst_vmax=float(worst_vmax),
        violation_time_s=float(violation_steps * dt),
        integral_violation_pu_s=float(integral_violation),
    )


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
    exclude_buses: tuple[str, ...] = ("rg60", "sourcebus"),
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
