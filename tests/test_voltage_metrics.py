"""Tests for openg2g.metrics.voltage — compute_allbus_voltage_stats."""

from __future__ import annotations

import math

import pytest

from openg2g.metrics.voltage import compute_allbus_voltage_stats
from openg2g.types import BusVoltages, GridState, ThreePhase


def _make_grid_state(
    time_s: float,
    bus_voltages: dict[str, tuple[float, float, float]],
) -> GridState:
    """Helper to build a GridState from a dict of bus -> (va, vb, vc)."""
    data = {bus: ThreePhase(a=v[0], b=v[1], c=v[2]) for bus, v in bus_voltages.items()}
    return GridState(time_s=time_s, voltages=BusVoltages(_data=data))


class TestComputeAllbusVoltageStats:
    def test_empty_states(self) -> None:
        """Empty input should return NaN for voltage extremes and zero violations."""
        stats = compute_allbus_voltage_stats([])
        assert math.isnan(stats.worst_vmin)
        assert math.isnan(stats.worst_vmax)
        assert stats.violation_time_s == 0.0
        assert stats.integral_violation_pu_s == 0.0

    def test_no_violations(self) -> None:
        """All voltages within bounds should produce zero violation metrics."""
        states = [
            _make_grid_state(0.0, {"bus1": (1.0, 1.0, 1.0)}),
            _make_grid_state(1.0, {"bus1": (1.0, 1.0, 1.0)}),
            _make_grid_state(2.0, {"bus1": (1.0, 1.0, 1.0)}),
        ]
        stats = compute_allbus_voltage_stats(states, v_min=0.95, v_max=1.05)
        assert stats.worst_vmin == 1.0
        assert stats.worst_vmax == 1.0
        assert stats.violation_time_s == 0.0
        assert stats.integral_violation_pu_s == 0.0

    def test_undervoltage_violation(self) -> None:
        """A single phase below v_min should contribute to integral violation
        and count all timesteps as violated."""
        states = [
            _make_grid_state(0.0, {"bus1": (0.93, 1.0, 1.0)}),
            _make_grid_state(1.0, {"bus1": (0.93, 1.0, 1.0)}),
        ]
        stats = compute_allbus_voltage_stats(states, v_min=0.95, v_max=1.05)
        assert stats.worst_vmin == pytest.approx(0.93)
        assert stats.integral_violation_pu_s == pytest.approx(0.04)
        assert stats.violation_time_s == pytest.approx(2.0)

    def test_overvoltage_violation(self) -> None:
        """A single phase above v_max should contribute to integral violation."""
        states = [
            _make_grid_state(0.0, {"bus1": (1.0, 1.07, 1.0)}),
            _make_grid_state(1.0, {"bus1": (1.0, 1.07, 1.0)}),
        ]
        stats = compute_allbus_voltage_stats(states, v_min=0.95, v_max=1.05)
        assert stats.worst_vmax == pytest.approx(1.07)
        assert stats.integral_violation_pu_s == pytest.approx(0.04)

    def test_mixed_under_and_over(self) -> None:
        """Both under- and overvoltage on different phases should be summed
        independently in the same timestep."""
        states = [
            _make_grid_state(0.0, {"bus1": (0.93, 1.07, 1.0)}),
        ]
        stats = compute_allbus_voltage_stats(states, v_min=0.95, v_max=1.05)
        assert stats.worst_vmin == pytest.approx(0.93)
        assert stats.worst_vmax == pytest.approx(1.07)
        assert stats.integral_violation_pu_s == pytest.approx(0.04)

    def test_multiple_buses_summed(self) -> None:
        """Violations across different buses should be summed per timestep."""
        states = [
            _make_grid_state(
                0.0,
                {
                    "bus1": (0.93, 1.0, 1.0),
                    "bus2": (1.0, 1.0, 0.94),
                },
            ),
            _make_grid_state(
                1.0,
                {
                    "bus1": (0.93, 1.0, 1.0),
                    "bus2": (1.0, 1.0, 0.94),
                },
            ),
        ]
        stats = compute_allbus_voltage_stats(states, v_min=0.95, v_max=1.05)
        assert stats.worst_vmin == pytest.approx(0.93)
        assert stats.integral_violation_pu_s == pytest.approx(0.06)

    def test_exclude_buses(self) -> None:
        """Excluded buses (e.g. rg60) should not contribute to any metric,
        even if they have severe violations."""
        states = [
            _make_grid_state(
                0.0,
                {
                    "rg60": (0.80, 0.80, 0.80),
                    "bus1": (1.0, 1.0, 1.0),
                },
            ),
            _make_grid_state(
                1.0,
                {
                    "rg60": (0.80, 0.80, 0.80),
                    "bus1": (1.0, 1.0, 1.0),
                },
            ),
        ]
        stats = compute_allbus_voltage_stats(states, v_min=0.95, v_max=1.05)
        assert stats.violation_time_s == 0.0
        assert stats.integral_violation_pu_s == 0.0
        assert stats.worst_vmin == 1.0

    def test_exclude_buses_case_insensitive(self) -> None:
        """Bus exclusion should be case-insensitive (e.g. 'RG60' excluded by 'rg60')."""
        states = [
            _make_grid_state(
                0.0,
                {
                    "RG60": (0.80, 0.80, 0.80),
                    "bus1": (1.0, 1.0, 1.0),
                },
            ),
        ]
        stats = compute_allbus_voltage_stats(
            states, v_min=0.95, v_max=1.05, exclude_buses=("rg60",)
        )
        assert stats.worst_vmin == 1.0

    def test_nan_phase_ignored(self) -> None:
        """NaN voltages (missing phases) should not affect worst-case
        min/max or contribute to violations."""
        states = [
            _make_grid_state(0.0, {"bus1": (1.0, float("nan"), 1.0)}),
            _make_grid_state(1.0, {"bus1": (1.0, float("nan"), 1.0)}),
        ]
        stats = compute_allbus_voltage_stats(states, v_min=0.95, v_max=1.05)
        assert stats.worst_vmin == 1.0
        assert stats.worst_vmax == 1.0
        assert stats.violation_time_s == 0.0

    def test_dt_from_median(self) -> None:
        """dt should be computed as the median of time diffs, making the
        integral robust to outlier gaps in timestamps."""
        states = [
            _make_grid_state(0.0, {"bus1": (0.93, 1.0, 1.0)}),
            _make_grid_state(0.1, {"bus1": (0.93, 1.0, 1.0)}),
            _make_grid_state(0.2, {"bus1": (0.93, 1.0, 1.0)}),
            _make_grid_state(0.3, {"bus1": (0.93, 1.0, 1.0)}),
            _make_grid_state(10.0, {"bus1": (0.93, 1.0, 1.0)}),
        ]
        stats = compute_allbus_voltage_stats(states, v_min=0.95, v_max=1.05)
        expected_dt = 0.1
        assert stats.integral_violation_pu_s == pytest.approx(5 * expected_dt * 0.02)
        assert stats.violation_time_s == pytest.approx(5 * expected_dt)

    def test_violation_time_partial(self) -> None:
        """Only timesteps with actual violations should count toward
        violation_time_s, not the entire simulation."""
        states = [
            _make_grid_state(0.0, {"bus1": (1.0, 1.0, 1.0)}),
            _make_grid_state(1.0, {"bus1": (0.93, 1.0, 1.0)}),
            _make_grid_state(2.0, {"bus1": (1.0, 1.0, 1.0)}),
        ]
        stats = compute_allbus_voltage_stats(states, v_min=0.95, v_max=1.05)
        assert stats.violation_time_s == pytest.approx(1.0)
        assert stats.integral_violation_pu_s == pytest.approx(0.02)

    def test_single_state(self) -> None:
        """A single snapshot should use dt=1.0 as fallback since no time
        diff is available."""
        states = [_make_grid_state(5.0, {"bus1": (0.90, 1.0, 1.0)})]
        stats = compute_allbus_voltage_stats(states, v_min=0.95, v_max=1.05)
        assert stats.worst_vmin == pytest.approx(0.90)
        assert stats.integral_violation_pu_s == pytest.approx(0.05)
        assert stats.violation_time_s == pytest.approx(1.0)
