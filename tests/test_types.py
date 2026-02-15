"""Tests for openg2g.types — TapPosition, TapSchedule, ServerRamp, ServerRampSchedule,
TrainingRun, TrainingSchedule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openg2g.types import (
    ServerRamp,
    ServerRampSchedule,
    TapPosition,
    TapSchedule,
    TrainingRun,
    TrainingSchedule,
)


class TestTapPosition:
    def test_full_three_phase(self) -> None:
        """All three phases specified should be stored as given."""
        pos = TapPosition(a=1.0, b=1.05, c=1.1)
        assert pos.a == 1.0
        assert pos.b == 1.05
        assert pos.c == 1.1

    def test_partial_a_only(self) -> None:
        """Specifying only phase A should leave B and C as None."""
        pos = TapPosition(a=1.1)
        assert pos.a == 1.1
        assert pos.b is None
        assert pos.c is None

    def test_partial_a_and_c(self) -> None:
        """Specifying A and C but not B should leave B as None."""
        pos = TapPosition(a=1.0625, c=1.0625)
        assert pos.a == 1.0625
        assert pos.b is None
        assert pos.c == 1.0625

    def test_no_phase_raises(self) -> None:
        """Constructing with no phases should raise ValueError."""
        with pytest.raises(ValueError, match="at least one phase"):
            TapPosition()

    def test_as_reg_dict_full(self) -> None:
        """as_reg_dict with all phases should map a/b/c to reg1/reg2/reg3."""
        pos = TapPosition(a=1.0, b=1.05, c=1.1)
        d = pos.as_reg_dict()
        assert d == {"reg1": 1.0, "reg2": 1.05, "reg3": 1.1}

    def test_as_reg_dict_partial(self) -> None:
        """as_reg_dict with only phase A should omit reg2 and reg3."""
        pos = TapPosition(a=1.1)
        d = pos.as_reg_dict()
        assert d == {"reg1": 1.1}
        assert "reg2" not in d
        assert "reg3" not in d

    def test_as_reg_dict_two_phases(self) -> None:
        """as_reg_dict with B and C should omit reg1."""
        pos = TapPosition(b=1.0375, c=1.09375)
        d = pos.as_reg_dict()
        assert d == {"reg2": 1.0375, "reg3": 1.09375}

    def test_at_returns_schedule(self) -> None:
        """Calling .at(t) should wrap the position in a single-entry TapSchedule."""
        pos = TapPosition(a=1.0, b=1.0, c=1.0)
        sched = pos.at(t=100.0)
        assert isinstance(sched, TapSchedule)
        assert len(sched) == 1


class TestTapSchedule:
    def test_pipe_composition(self) -> None:
        """The | operator should combine multiple TapSchedules into one."""
        s = (
            TapPosition(a=1.0, b=1.0, c=1.0).at(t=0)
            | TapPosition(a=1.1).at(t=100)
            | TapPosition(a=1.05, c=1.05).at(t=200)
        )
        assert len(s) == 3

    def test_sorted_by_time(self) -> None:
        """Entries should be sorted by time regardless of pipe order."""
        s = TapPosition(a=1.1).at(t=200) | TapPosition(a=1.0).at(t=0)
        times = [t for t, _ in s]
        assert times == [0, 200]

    def test_iteration(self) -> None:
        """Iterating a schedule should yield (time, TapPosition) tuples."""
        s = TapPosition(a=1.0, b=1.0, c=1.0).at(t=50)
        entries = list(s)
        assert len(entries) == 1
        t, pos = entries[0]
        assert t == 50.0
        assert pos.a == 1.0

    def test_bool_empty(self) -> None:
        """An empty schedule should be falsy."""
        s = TapSchedule(())
        assert not s

    def test_bool_nonempty(self) -> None:
        """A non-empty schedule should be truthy."""
        s = TapPosition(a=1.0).at(t=0)
        assert s

    def test_repr_partial(self) -> None:
        """repr of a partial-phase schedule should only show specified phases."""
        s = TapPosition(a=1.1).at(t=100)
        r = repr(s)
        assert "a=1.1" in r
        assert "b=" not in r
        assert "c=" not in r


class TestServerRamp:
    def test_basic(self) -> None:
        """ServerRamp should store its start, end, and target fraction."""
        r = ServerRamp(t_start=1000, t_end=2000, target=0.5)
        assert r.t_start == 1000
        assert r.t_end == 2000
        assert r.target == 0.5

    def test_invalid_time_order(self) -> None:
        """t_end before t_start should raise ValueError."""
        with pytest.raises(ValueError, match=r"t_end.*must be >= t_start"):
            ServerRamp(t_start=2000, t_end=1000, target=0.5)

    def test_invalid_target_low(self) -> None:
        """Negative target fraction should raise ValueError."""
        with pytest.raises(ValueError, match="target must be in"):
            ServerRamp(t_start=0, t_end=100, target=-0.1)

    def test_invalid_target_high(self) -> None:
        """Target fraction above 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="target must be in"):
            ServerRamp(t_start=0, t_end=100, target=1.5)

    def test_pipe_creates_schedule(self) -> None:
        """Piping two ServerRamps should produce a ServerRampSchedule."""
        s = ServerRamp(t_start=100, t_end=200, target=0.5) | ServerRamp(
            t_start=300, t_end=400, target=1.0
        )
        assert isinstance(s, ServerRampSchedule)
        assert len(s) == 2

    def test_pipe_with_schedule(self) -> None:
        """Chaining three ServerRamps with | should accumulate all entries."""
        r1 = ServerRamp(t_start=100, t_end=200, target=0.5)
        r2 = ServerRamp(t_start=300, t_end=400, target=1.0)
        r3 = ServerRamp(t_start=500, t_end=600, target=0.3)
        s = r1 | r2 | r3
        assert isinstance(s, ServerRampSchedule)
        assert len(s) == 3


class TestServerRampSchedule:
    def test_fraction_before_first_ramp(self) -> None:
        """Before the first ramp starts, the active fraction should be 1.0."""
        s = ServerRampSchedule(entries=(ServerRamp(t_start=1000, t_end=2000, target=0.5),))
        assert s.fraction_at(0.0) == 1.0
        assert s.fraction_at(999.0) == 1.0

    def test_fraction_during_ramp(self) -> None:
        """During a ramp, the fraction should linearly interpolate from
        the previous level to the target."""
        s = ServerRampSchedule(entries=(ServerRamp(t_start=1000, t_end=2000, target=0.0),))
        assert s.fraction_at(1000.0) == 1.0
        assert s.fraction_at(1500.0) == pytest.approx(0.5)
        assert s.fraction_at(2000.0) == pytest.approx(0.0)

    def test_fraction_after_ramp(self) -> None:
        """After a ramp completes, the fraction should hold at the target."""
        s = ServerRampSchedule(entries=(ServerRamp(t_start=1000, t_end=2000, target=0.2),))
        assert s.fraction_at(3000.0) == pytest.approx(0.2)

    def test_two_ramps(self) -> None:
        """Two sequential ramps: first ramps down to 0.2, second ramps back
        up to 1.0. The fraction should hold between ramps."""
        s = ServerRamp(t_start=1000, t_end=2000, target=0.2) | ServerRamp(
            t_start=3000, t_end=3500, target=1.0
        )
        assert s.fraction_at(0.0) == 1.0
        assert s.fraction_at(1500.0) == pytest.approx(0.6)
        assert s.fraction_at(2500.0) == pytest.approx(0.2)
        assert s.fraction_at(3250.0) == pytest.approx(0.6)
        assert s.fraction_at(4000.0) == pytest.approx(1.0)

    def test_instant_ramp(self) -> None:
        """A ramp with t_start == t_end should produce an instant step change."""
        s = ServerRampSchedule(entries=(ServerRamp(t_start=1000, t_end=1000, target=0.5),))
        assert s.fraction_at(999.0) == 1.0
        assert s.fraction_at(1000.0) == 0.5
        assert s.fraction_at(1001.0) == 0.5

    def test_fraction_array(self) -> None:
        """fraction_at should accept a numpy array and return element-wise results."""
        s = ServerRampSchedule(entries=(ServerRamp(t_start=1000, t_end=2000, target=0.0),))
        t = np.array([0.0, 1000.0, 1500.0, 2000.0, 3000.0])
        result = s.fraction_at(t)
        expected = np.array([1.0, 1.0, 0.5, 0.0, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_sorted_by_start(self) -> None:
        """Ramps piped in reverse order should still be sorted by t_start."""
        s = ServerRamp(t_start=3000, t_end=3500, target=1.0) | ServerRamp(
            t_start=1000, t_end=2000, target=0.2
        )
        starts = [r.t_start for r in s]
        assert starts == [1000, 3000]


class TestTrainingRun:
    def test_basic(self) -> None:
        """TrainingRun should store its time window, GPU count, and trace path."""
        r = TrainingRun(t_start=100, t_end=200, n_gpus=2400, trace_csv=Path("/tmp/trace.csv"))
        assert r.t_start == 100
        assert r.t_end == 200
        assert r.n_gpus == 2400

    def test_invalid_time_order(self) -> None:
        """t_end before t_start should raise ValueError."""
        with pytest.raises(ValueError, match=r"t_end.*must be >= t_start"):
            TrainingRun(t_start=200, t_end=100, n_gpus=2400, trace_csv=Path("/tmp/trace.csv"))

    def test_negative_gpus(self) -> None:
        """Negative GPU count should raise ValueError."""
        with pytest.raises(ValueError, match="n_gpus must be >= 0"):
            TrainingRun(t_start=0, t_end=100, n_gpus=-1, trace_csv=Path("/tmp/trace.csv"))

    def test_default_target_peak(self) -> None:
        """target_peak_W_per_gpu should default to 400.0."""
        r = TrainingRun(t_start=0, t_end=100, n_gpus=100, trace_csv=Path("/tmp/trace.csv"))
        assert r.target_peak_W_per_gpu == 400.0

    def test_pipe_creates_schedule(self) -> None:
        """Piping two TrainingRuns should produce a TrainingSchedule."""
        r1 = TrainingRun(t_start=0, t_end=100, n_gpus=100, trace_csv=Path("/tmp/a.csv"))
        r2 = TrainingRun(t_start=200, t_end=300, n_gpus=50, trace_csv=Path("/tmp/b.csv"))
        s = r1 | r2
        assert isinstance(s, TrainingSchedule)
        assert len(s) == 2


class TestTrainingSchedule:
    def test_sorted_by_start(self) -> None:
        """Runs piped in reverse order should still be sorted by t_start."""
        r1 = TrainingRun(t_start=200, t_end=300, n_gpus=50, trace_csv=Path("/tmp/b.csv"))
        r2 = TrainingRun(t_start=0, t_end=100, n_gpus=100, trace_csv=Path("/tmp/a.csv"))
        s = r1 | r2
        starts = [r.t_start for r in s]
        assert starts == [0, 200]

    def test_pipe_three(self) -> None:
        """Chaining three TrainingRuns with | should accumulate all entries."""
        r1 = TrainingRun(t_start=0, t_end=100, n_gpus=100, trace_csv=Path("/tmp/a.csv"))
        r2 = TrainingRun(t_start=200, t_end=300, n_gpus=50, trace_csv=Path("/tmp/b.csv"))
        r3 = TrainingRun(t_start=400, t_end=500, n_gpus=25, trace_csv=Path("/tmp/c.csv"))
        s = r1 | r2 | r3
        assert len(s) == 3

    def test_bool(self) -> None:
        """An empty schedule should be falsy; a non-empty one should be truthy."""
        s = TrainingSchedule(entries=())
        assert not s
        r = TrainingRun(t_start=0, t_end=100, n_gpus=100, trace_csv=Path("/tmp/a.csv"))
        s2 = TrainingSchedule(entries=(r,))
        assert s2

    def test_repr(self) -> None:
        """repr should include 'TrainingRun' for debuggability."""
        r = TrainingRun(t_start=0, t_end=100, n_gpus=100, trace_csv=Path("/tmp/a.csv"))
        s = TrainingSchedule(entries=(r,))
        assert "TrainingRun" in repr(s)
