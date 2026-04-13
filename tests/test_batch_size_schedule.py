"""Tests for BatchSizeChange, BatchSizeSchedule, and BatchSizeScheduleController."""

from __future__ import annotations

from fractions import Fraction
from unittest.mock import MagicMock

import pytest

from openg2g.controller.batch_size_schedule import BatchSizeChange, BatchSizeSchedule, BatchSizeScheduleController
from openg2g.datacenter.command import SetBatchSize


class TestBatchSizeChange:
    def test_basic_construction(self) -> None:
        c = BatchSizeChange(batch_size=64)
        assert c.batch_size == 64
        assert c.ramp_up_rate == 0.0

    def test_with_ramp(self) -> None:
        c = BatchSizeChange(batch_size=32, ramp_up_rate=4.0)
        assert c.batch_size == 32
        assert c.ramp_up_rate == 4.0

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            BatchSizeChange(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            BatchSizeChange(batch_size=-1)

    def test_invalid_ramp_up_rate(self) -> None:
        with pytest.raises(ValueError, match="ramp_up_rate must be >= 0"):
            BatchSizeChange(batch_size=32, ramp_up_rate=-1.0)

    def test_at_creates_schedule(self) -> None:
        s = BatchSizeChange(48).at(40.0)
        assert isinstance(s, BatchSizeSchedule)
        assert len(s) == 1
        entries = list(s)
        assert entries[0][0] == 40.0
        assert entries[0][1].batch_size == 48


class TestBatchSizeSchedule:
    def test_pipe_composition(self) -> None:
        s = BatchSizeChange(48).at(40) | BatchSizeChange(32).at(60) | BatchSizeChange(64).at(80)
        assert len(s) == 3
        entries = list(s)
        assert entries[0][0] == 40
        assert entries[1][0] == 60
        assert entries[2][0] == 80

    def test_sorted_by_time(self) -> None:
        s = BatchSizeChange(32).at(60) | BatchSizeChange(48).at(40)
        entries = list(s)
        assert entries[0][0] == 40
        assert entries[1][0] == 60

    def test_bool(self) -> None:
        assert bool(BatchSizeChange(32).at(0))
        assert not bool(BatchSizeSchedule(()))

    def test_repr(self) -> None:
        s = BatchSizeChange(48).at(40) | BatchSizeChange(32, ramp_up_rate=4).at(60)
        r = repr(s)
        assert "BatchSizeChange(48)" in r
        assert "ramp_up_rate=4" in r

    def test_duplicate_timestamps_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate timestamps"):
            BatchSizeChange(48).at(40) | BatchSizeChange(32).at(40)

    def test_duplicate_timestamps_three_entries(self) -> None:
        with pytest.raises(ValueError, match="duplicate timestamps"):
            BatchSizeChange(48).at(40) | BatchSizeChange(32).at(60) | BatchSizeChange(64).at(40)

    def test_duplicate_timestamps_direct_construction(self) -> None:
        with pytest.raises(ValueError, match="duplicate timestamps"):
            BatchSizeSchedule(((40.0, BatchSizeChange(48)), (40.0, BatchSizeChange(32))))


class TestBatchSizeScheduleController:
    def _make_clock(self, time_s: float) -> MagicMock:
        clock = MagicMock()
        clock.time_s = time_s
        return clock

    def test_emits_at_scheduled_time(self) -> None:
        schedule = BatchSizeChange(48).at(10) | BatchSizeChange(32).at(20)
        dc = MagicMock()
        ctrl = BatchSizeScheduleController(
            datacenter=dc,
            schedules={"model-a": schedule},
            dt_s=Fraction(1),
        )
        events = MagicMock()

        # Before first event
        action = ctrl.step(self._make_clock(5.0), events)
        assert len(action) == 0

        # At first event
        action = ctrl.step(self._make_clock(10.0), events)
        assert len(action) == 1
        cmd = action[0]
        assert isinstance(cmd, SetBatchSize)
        assert cmd.batch_size_by_model["model-a"] == 48

        # Between events
        action = ctrl.step(self._make_clock(15.0), events)
        assert len(action) == 0

        # At second event
        action = ctrl.step(self._make_clock(20.0), events)
        assert len(action) == 1
        assert isinstance(action[0], SetBatchSize)
        assert action[0].batch_size_by_model["model-a"] == 32

    def test_multiple_models(self) -> None:
        schedules = {
            "model-a": BatchSizeChange(48).at(10),
            "model-b": BatchSizeChange(64).at(10),
        }
        dc = MagicMock()
        ctrl = BatchSizeScheduleController(datacenter=dc, schedules=schedules, dt_s=Fraction(1))
        events = MagicMock()

        action = ctrl.step(self._make_clock(10.0), events)
        assert len(action) == 1
        cmd = action[0]
        assert isinstance(cmd, SetBatchSize)
        assert cmd.batch_size_by_model["model-a"] == 48
        assert cmd.batch_size_by_model["model-b"] == 64

    def test_ramp_up_rate_per_model(self) -> None:
        schedule = BatchSizeChange(32, ramp_up_rate=4.0).at(10)
        dc = MagicMock()
        ctrl = BatchSizeScheduleController(
            datacenter=dc,
            schedules={"model-a": schedule},
            dt_s=Fraction(1),
        )
        events = MagicMock()

        action = ctrl.step(self._make_clock(10.0), events)
        cmd = action[0]
        assert isinstance(cmd, SetBatchSize)
        assert cmd.ramp_up_rate_by_model == {"model-a": 4.0}

    def test_no_ramp_up_rate_when_zero(self) -> None:
        schedule = BatchSizeChange(32).at(10)
        dc = MagicMock()
        ctrl = BatchSizeScheduleController(
            datacenter=dc,
            schedules={"model-a": schedule},
            dt_s=Fraction(1),
        )
        events = MagicMock()

        action = ctrl.step(self._make_clock(10.0), events)
        cmd = action[0]
        assert isinstance(cmd, SetBatchSize)
        assert cmd.ramp_up_rate_by_model == {}

    def test_mixed_ramp_rates_per_model(self) -> None:
        schedules = {
            "model-a": BatchSizeChange(32, ramp_up_rate=4.0).at(10),
            "model-b": BatchSizeChange(64, ramp_up_rate=8.0).at(10),
            "model-c": BatchSizeChange(128).at(10),
        }
        dc = MagicMock()
        ctrl = BatchSizeScheduleController(datacenter=dc, schedules=schedules, dt_s=Fraction(1))
        events = MagicMock()

        action = ctrl.step(self._make_clock(10.0), events)
        cmd = action[0]
        assert isinstance(cmd, SetBatchSize)
        assert cmd.ramp_up_rate_by_model == {"model-a": 4.0, "model-b": 8.0}
        assert "model-c" not in cmd.ramp_up_rate_by_model

    def test_dt_s(self) -> None:
        dc = MagicMock()
        ctrl = BatchSizeScheduleController(datacenter=dc, schedules={}, dt_s=Fraction(2))
        assert ctrl.dt_s == Fraction(2)

    def test_empty_schedule(self) -> None:
        dc = MagicMock()
        ctrl = BatchSizeScheduleController(datacenter=dc, schedules={}, dt_s=Fraction(1))
        events = MagicMock()

        action = ctrl.step(self._make_clock(100.0), events)
        assert len(action) == 0
