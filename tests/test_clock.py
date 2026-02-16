"""Tests for SimulationClock: is_due alignment and live-mode lag detection."""

from __future__ import annotations

import warnings
from fractions import Fraction
from unittest.mock import patch

import pytest

from openg2g.clock import SimulationClock


def test_time_starts_at_zero():
    clock = SimulationClock(tick_s=Fraction(1, 10))
    assert clock.time_s == 0.0
    assert clock.step == 0


def test_advance_increments():
    clock = SimulationClock(tick_s=Fraction(1, 2))
    t = clock.advance()
    assert t == 0.5
    assert clock.step == 1
    t = clock.advance()
    assert t == 1.0
    assert clock.step == 2


def test_is_due_every_tick():
    clock = SimulationClock(tick_s=Fraction(1, 10))
    assert clock.is_due(Fraction(1, 10))
    clock.advance()
    assert clock.is_due(Fraction(1, 10))


def test_is_due_multi_rate():
    clock = SimulationClock(tick_s=Fraction(1, 10))
    # At step 0: everything is due
    assert clock.is_due(Fraction(1, 10))
    assert clock.is_due(Fraction(1))
    assert clock.is_due(Fraction(60))

    # Advance 1 tick (step=1, t=0.1s)
    clock.advance()
    assert clock.is_due(Fraction(1, 10))
    assert not clock.is_due(Fraction(1))
    assert not clock.is_due(Fraction(60))

    # Advance to step=10 (t=1.0s)
    for _ in range(9):
        clock.advance()
    assert clock.step == 10
    assert clock.is_due(Fraction(1, 10))
    assert clock.is_due(Fraction(1))
    assert not clock.is_due(Fraction(60))

    # Advance to step=600 (t=60.0s)
    for _ in range(590):
        clock.advance()
    assert clock.step == 600
    assert clock.is_due(Fraction(1, 10))
    assert clock.is_due(Fraction(1))
    assert clock.is_due(Fraction(60))


def test_is_due_non_power_of_ten():
    """Period 3/10 is an exact multiple of tick 1/10 → 3 ticks."""
    clock = SimulationClock(tick_s=Fraction(1, 10))
    assert clock.is_due(Fraction(3, 10))  # step 0
    clock.advance()  # step 1
    assert not clock.is_due(Fraction(3, 10))
    clock.advance()  # step 2
    assert not clock.is_due(Fraction(3, 10))
    clock.advance()  # step 3
    assert clock.is_due(Fraction(3, 10))


def test_is_due_period_not_multiple_of_tick_raises():
    """If period is not an exact multiple of tick, raise ValueError."""
    clock = SimulationClock(tick_s=Fraction(1))
    with pytest.raises(ValueError, match="not an exact multiple"):
        clock.is_due(Fraction(1, 100))


def test_tick_s_must_be_fraction():
    """Passing a float for tick_s raises TypeError."""
    with pytest.raises(TypeError, match="must be a Fraction"):
        SimulationClock(tick_s=0.1)  # type: ignore[invalid-argument-type]


def test_tick_s_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        SimulationClock(tick_s=Fraction(0))
    with pytest.raises(ValueError, match="must be positive"):
        SimulationClock(tick_s=Fraction(-1, 10))


def test_live_mode_lag_warning():
    """Live mode warns when computation takes longer than a tick."""
    clock = SimulationClock(tick_s=Fraction(1, 10), live=True)

    # Simulate wall time progressing faster than real-time
    # First advance sets _wall_t0
    times = iter([0.0, 0.0, 0.5])  # t0=0, then check at 0, then lagging at 0.5

    with patch("openg2g.clock.time") as mock_time:
        mock_time.monotonic = lambda: next(times)
        mock_time.sleep = lambda _: None

        clock.advance()  # step 1, t=0.1s. Sets _wall_t0=0.0, now=0.0 -> ahead, sleep
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clock.advance()  # step 2, t=0.2s. now=0.5 -> lag of 0.3 > tick_s
            assert len(w) == 1
            assert "Clock lag" in str(w[0].message)


def test_live_mode_sleeps_when_ahead():
    """Live mode sleeps when ahead of wall time."""
    clock = SimulationClock(tick_s=Fraction(1), live=True)

    sleep_calls = []

    with patch("openg2g.clock.time") as mock_time:
        # First advance: sets wall_t0=100.0, then checks expected=101.0 vs now=100.0
        mock_time.monotonic = lambda: 100.0
        mock_time.sleep = lambda dt: sleep_calls.append(dt)

        clock.advance()
        # Expected wall = 100.0 + 1.0 = 101.0, now = 100.0 -> sleep(1.0)
        assert len(sleep_calls) == 1
        assert abs(sleep_calls[0] - 1.0) < 0.01
