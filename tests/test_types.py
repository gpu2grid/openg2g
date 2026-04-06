"""Tests for config/schedule types: TapPosition, TapSchedule, InferenceRamp,
InferenceRampSchedule, ActivationPolicy, TrainingRun, TrainingSchedule."""

from __future__ import annotations

import numpy as np
import pytest

from openg2g.common import ThreePhase
from openg2g.datacenter.config import (
    InferenceRamp,
    InferenceRampSchedule,
    TrainingRun,
    TrainingSchedule,
)
from openg2g.datacenter.layout import ActivationPolicy, RampActivationPolicy
from openg2g.datacenter.online import OnlineDatacenterState
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.base import BusVoltages, GridState, PhaseVoltages
from openg2g.grid.config import TapPosition, TapSchedule

_DUMMY_TRACE = TrainingTrace(t_s=np.array([0.0, 1.0]), power_w=np.array([100.0, 200.0]))


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
        with pytest.raises(ValueError, match="at least one"):
            TapPosition()

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

    def test_duplicate_timestamps_raises(self) -> None:
        """Constructing a schedule with duplicate timestamps should raise ValueError."""
        with pytest.raises(ValueError, match="duplicate timestamps"):
            TapPosition(a=1.0).at(t=0) | TapPosition(a=1.1).at(t=0)

    def test_duplicate_timestamps_three_entries(self) -> None:
        """Two of three entries sharing a timestamp should raise ValueError."""
        with pytest.raises(ValueError, match="duplicate timestamps"):
            TapPosition(a=1.0).at(t=0) | TapPosition(a=1.1).at(t=100) | TapPosition(a=1.05).at(t=0)

    def test_duplicate_timestamps_direct_construction(self) -> None:
        """Direct TapSchedule construction with duplicates should also raise."""
        with pytest.raises(ValueError, match="duplicate timestamps"):
            TapSchedule(((0.0, TapPosition(a=1.0)), (0.0, TapPosition(a=1.1))))


class TestInferenceRamp:
    def test_basic(self) -> None:
        """InferenceRamp should store its target count and model."""
        r = InferenceRamp(target=50, model="M")
        assert r.target == 50
        assert r.model == "M"

    def test_invalid_target_low(self) -> None:
        """Negative target should raise ValueError."""
        with pytest.raises(ValueError, match=r"target must be >= 0"):
            InferenceRamp(target=-1, model="M")

    def test_target_above_initial_is_valid(self) -> None:
        """Target above initial count is valid (scale out)."""
        r = InferenceRamp(target=150, model="M")
        assert r.target == 150

    def test_at_returns_schedule(self) -> None:
        """Calling .at() should wrap in a single-entry InferenceRampSchedule."""
        s = InferenceRamp(target=50, model="M").at(t_start=1000, t_end=2000)
        assert isinstance(s, InferenceRampSchedule)
        assert len(s) == 1

    def test_at_invalid_time_order(self) -> None:
        """t_end before t_start should raise ValueError."""
        with pytest.raises(ValueError, match=r"t_end.*must be >= t_start"):
            InferenceRamp(target=50, model="M").at(t_start=2000, t_end=1000)

    def test_pipe_creates_schedule(self) -> None:
        """Piping two scheduled ramps should produce an InferenceRampSchedule."""
        s = (
            InferenceRamp(target=50, model="M").at(t_start=100, t_end=200)
            | InferenceRamp(target=100, model="M").at(t_start=300, t_end=400)
        )
        assert isinstance(s, InferenceRampSchedule)
        assert len(s) == 2

    def test_pipe_three(self) -> None:
        """Chaining three ramps with | should accumulate all entries."""
        s = (
            InferenceRamp(target=50, model="M").at(t_start=100, t_end=200)
            | InferenceRamp(target=100, model="M").at(t_start=300, t_end=400)
            | InferenceRamp(target=30, model="M").at(t_start=500, t_end=600)
        )
        assert isinstance(s, InferenceRampSchedule)
        assert len(s) == 3


class TestInferenceRampSchedule:
    def test_count_before_first_ramp(self) -> None:
        """Before the first ramp starts, count should be initial_count."""
        s = InferenceRampSchedule(
            (InferenceRamp(target=50, model="M"), 1000.0, 2000.0),
        )
        s = InferenceRamp(target=50, model="M").at(t_start=1000, t_end=2000)
        s = InferenceRampSchedule(s._entries, initial_count=100)
        assert s.count_at(0.0) == 100.0
        assert s.count_at(999.0) == 100.0

    def test_count_during_ramp(self) -> None:
        """During a ramp, the count should linearly interpolate."""
        s = InferenceRampSchedule(
            ((InferenceRamp(target=0, model="M"), 1000.0, 2000.0),),
            initial_count=100,
        )
        assert s.count_at(1000.0) == 100.0
        assert s.count_at(1500.0) == pytest.approx(50.0)
        assert s.count_at(2000.0) == pytest.approx(0.0)

    def test_count_after_ramp(self) -> None:
        """After a ramp completes, the count should hold at the target."""
        s = InferenceRampSchedule(
            ((InferenceRamp(target=20, model="M"), 1000.0, 2000.0),),
            initial_count=100,
        )
        assert s.count_at(3000.0) == pytest.approx(20.0)

    def test_two_ramps(self) -> None:
        """Two sequential ramps: first ramps down to 20, second ramps back
        up to 100. The count should hold between ramps."""
        s = (
            InferenceRamp(target=20, model="M").at(t_start=1000, t_end=2000)
            | InferenceRamp(target=100, model="M").at(t_start=3000, t_end=3500)
        )
        s = InferenceRampSchedule(s._entries, initial_count=100)
        assert s.count_at(0.0) == 100.0
        assert s.count_at(1500.0) == pytest.approx(60.0)
        assert s.count_at(2500.0) == pytest.approx(20.0)
        assert s.count_at(3250.0) == pytest.approx(60.0)
        assert s.count_at(4000.0) == pytest.approx(100.0)

    def test_instant_ramp(self) -> None:
        """A ramp with t_start == t_end should produce an instant step change."""
        s = InferenceRampSchedule(
            ((InferenceRamp(target=50, model="M"), 1000.0, 1000.0),),
            initial_count=100,
        )
        assert s.count_at(999.0) == 100.0
        assert s.count_at(1000.0) == 50.0
        assert s.count_at(1001.0) == 50.0

    def test_count_array(self) -> None:
        """count_at should accept a numpy array and return element-wise results."""
        s = InferenceRampSchedule(
            ((InferenceRamp(target=0, model="M"), 1000.0, 2000.0),),
            initial_count=100,
        )
        t = np.array([0.0, 1000.0, 1500.0, 2000.0, 3000.0])
        result = s.count_at(t)
        expected = np.array([100.0, 100.0, 50.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_sorted_by_start(self) -> None:
        """Ramps piped in reverse order should still be sorted by t_start."""
        s = (
            InferenceRamp(target=100, model="M").at(t_start=3000, t_end=3500)
            | InferenceRamp(target=20, model="M").at(t_start=1000, t_end=2000)
        )
        starts = [t_start for _, t_start, _ in s]
        assert starts == [1000, 3000]


class TestTrainingRun:
    def test_basic(self) -> None:
        """TrainingRun should store its GPU count and trace."""
        r = TrainingRun(n_gpus=2400, trace=_DUMMY_TRACE)
        assert r.n_gpus == 2400

    def test_negative_gpus(self) -> None:
        """Negative GPU count should raise ValueError."""
        with pytest.raises(ValueError, match="n_gpus must be > 0"):
            TrainingRun(n_gpus=-1, trace=_DUMMY_TRACE)

    def test_default_target_peak(self) -> None:
        """target_peak_W_per_gpu should default to 400.0."""
        r = TrainingRun(n_gpus=100, trace=_DUMMY_TRACE)
        assert r.target_peak_W_per_gpu == 400.0

    def test_at_returns_schedule(self) -> None:
        """Calling .at() should wrap in a single-entry TrainingSchedule."""
        s = TrainingRun(n_gpus=100, trace=_DUMMY_TRACE).at(t_start=0, t_end=100)
        assert isinstance(s, TrainingSchedule)
        assert len(s) == 1

    def test_at_invalid_time_order(self) -> None:
        """t_end before t_start should raise ValueError."""
        with pytest.raises(ValueError, match=r"t_end.*must be >= t_start"):
            TrainingRun(n_gpus=100, trace=_DUMMY_TRACE).at(t_start=200, t_end=100)

    def test_scheduled_fields(self) -> None:
        """Schedule entries should expose run and time fields via tuple."""
        r = TrainingRun(n_gpus=2400, trace=_DUMMY_TRACE, target_peak_W_per_gpu=500.0)
        run, t_start, t_end = next(iter(r.at(t_start=100, t_end=200)))
        assert run is r
        assert run.n_gpus == 2400
        assert run.target_peak_W_per_gpu == 500.0
        assert t_start == 100
        assert t_end == 200

    def test_pipe_creates_schedule(self) -> None:
        """Piping two scheduled training runs should produce a TrainingSchedule."""
        s = TrainingRun(n_gpus=100, trace=_DUMMY_TRACE).at(t_start=0, t_end=100) | TrainingRun(
            n_gpus=50, trace=_DUMMY_TRACE
        ).at(t_start=200, t_end=300)
        assert isinstance(s, TrainingSchedule)
        assert len(s) == 2


class TestTrainingSchedule:
    def test_sorted_by_start(self) -> None:
        """Runs piped in reverse order should still be sorted by t_start."""
        s = TrainingRun(n_gpus=50, trace=_DUMMY_TRACE).at(t_start=200, t_end=300) | TrainingRun(
            n_gpus=100, trace=_DUMMY_TRACE
        ).at(t_start=0, t_end=100)
        starts = [t_start for _, t_start, _ in s]
        assert starts == [0, 200]

    def test_pipe_three(self) -> None:
        """Chaining three scheduled runs with | should accumulate all entries."""
        s = (
            TrainingRun(n_gpus=100, trace=_DUMMY_TRACE).at(t_start=0, t_end=100)
            | TrainingRun(n_gpus=50, trace=_DUMMY_TRACE).at(t_start=200, t_end=300)
            | TrainingRun(n_gpus=25, trace=_DUMMY_TRACE).at(t_start=400, t_end=500)
        )
        assert len(s) == 3

    def test_bool(self) -> None:
        """An empty schedule should be falsy; a non-empty one should be truthy."""
        s = TrainingSchedule()
        assert not s
        s2 = TrainingRun(n_gpus=100, trace=_DUMMY_TRACE).at(t_start=0, t_end=100)
        assert s2

    def test_repr(self) -> None:
        """repr should include 'TrainingRun' for debuggability."""
        s = TrainingRun(n_gpus=100, trace=_DUMMY_TRACE).at(t_start=0, t_end=100)
        assert "TrainingRun" in repr(s)


class TestGridState:
    def test_tap_positions_default_none(self) -> None:
        state = GridState(
            time_s=0.0,
            voltages=BusVoltages({"671": PhaseVoltages(a=1.0, b=1.0, c=1.0)}),
        )
        assert state.tap_positions is None

    def test_tap_positions_populated(self) -> None:
        taps = TapPosition(a=1.0875, b=1.0375, c=1.09375)
        state = GridState(
            time_s=1.0,
            voltages=BusVoltages({"671": PhaseVoltages(a=0.98, b=0.99, c=1.01)}),
            tap_positions=taps,
        )
        assert state.tap_positions is taps
        assert state.tap_positions.a == 1.0875


class TestOnlineDatacenterState:
    def test_construction_with_all_fields(self) -> None:
        state = OnlineDatacenterState(
            time_s=0.5,
            power_w=ThreePhase(a=500e3, b=500e3, c=500e3),
            batch_size_by_model={"8B": 128},
            active_replicas_by_model={"8B": 4},
            observed_itl_s_by_model={"8B": 0.05},
            measured_power_w=ThreePhase(a=50e3, b=50e3, c=50e3),
            measured_power_w_by_model={"8B": 120e3},
            augmented_power_w_by_model={"8B": 1200e3},
            augmentation_factor_by_model={"8B": 10.0},
        )
        assert state.power_w.a + state.power_w.b + state.power_w.c == 1500e3
        assert state.measured_power_w.a + state.measured_power_w.b + state.measured_power_w.c == 150e3
        assert state.augmented_power_w_by_model["8B"] == 1200e3
        assert state.measured_power_w_by_model["8B"] == 120e3
        assert state.augmentation_factor_by_model["8B"] == 10.0

    def test_defaults(self) -> None:
        state = OnlineDatacenterState(
            time_s=0.0,
            power_w=ThreePhase(a=0.0, b=0.0, c=0.0),
        )
        assert state.measured_power_w.a + state.measured_power_w.b + state.measured_power_w.c == 0.0
        assert state.measured_power_w_by_model == {}
        assert state.augmented_power_w_by_model == {}
        assert state.augmentation_factor_by_model == {}


class TestRampActivationPolicy:
    # 10 replicas, 1 GPU/replica, 8 GPUs/server → ceil(10/8) = 2 servers
    _POLICY_KWARGS = dict(gpus_per_replica=1, gpus_per_server=8)

    def test_all_active_before_ramp(self) -> None:
        """Before the first ramp, all servers should be active."""
        schedule = InferenceRampSchedule(
            ((InferenceRamp(target=5, model="M"), 1000.0, 2000.0),),
            initial_count=10,
        )
        # 10 replicas × 1 GPU = 10 GPUs → ceil(10/8) = 2 servers
        policy = RampActivationPolicy(schedule, num_servers=2, rng=np.random.default_rng(42), **self._POLICY_KWARGS)
        mask = policy.active_mask(0.0)
        assert mask.sum() == 2

    def test_ramp_down(self) -> None:
        """After ramp to 5 replicas, only 1 server should be active (ceil(5/8))."""
        schedule = InferenceRampSchedule(
            ((InferenceRamp(target=5, model="M"), 0.0, 0.0),),
            initial_count=10,
        )
        policy = RampActivationPolicy(schedule, num_servers=2, rng=np.random.default_rng(42), **self._POLICY_KWARGS)
        mask = policy.active_mask(1.0)
        assert mask.sum() == 1  # ceil(5*1/8) = 1

    def test_ramp_up(self) -> None:
        """Ramp up: start at 2 replicas, ramp to 10."""
        schedule = InferenceRampSchedule(
            (
                (InferenceRamp(target=2, model="M"), 0.0, 0.0),
                (InferenceRamp(target=10, model="M"), 100.0, 100.0),
            ),
            initial_count=10,
        )
        policy = RampActivationPolicy(schedule, num_servers=2, rng=np.random.default_rng(42), **self._POLICY_KWARGS)
        mask_low = policy.active_mask(50.0)
        mask_high = policy.active_mask(200.0)
        assert mask_low.sum() == 1  # ceil(2/8) = 1
        assert mask_high.sum() == 2  # ceil(10/8) = 2

    def test_ramp_up_servers_superset(self) -> None:
        """Servers active at low count should be a subset of those active at high count."""
        schedule = InferenceRampSchedule(
            (
                (InferenceRamp(target=3, model="M"), 0.0, 0.0),
                (InferenceRamp(target=7, model="M"), 100.0, 100.0),
            ),
            initial_count=10,
        )
        policy = RampActivationPolicy(schedule, num_servers=2, rng=np.random.default_rng(42), **self._POLICY_KWARGS)
        mask_low = policy.active_mask(50.0)
        mask_high = policy.active_mask(200.0)
        assert np.all(mask_high[mask_low])

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed should produce the same activation mask."""
        schedule = InferenceRampSchedule(
            ((InferenceRamp(target=5, model="M"), 0.0, 0.0),),
            initial_count=10,
        )
        p1 = RampActivationPolicy(schedule, num_servers=3, rng=np.random.default_rng(0), **self._POLICY_KWARGS)
        p2 = RampActivationPolicy(schedule, num_servers=3, rng=np.random.default_rng(0), **self._POLICY_KWARGS)
        np.testing.assert_array_equal(p1.active_mask(1.0), p2.active_mask(1.0))

    def test_custom_policy_subclass(self) -> None:
        """Custom ActivationPolicy subclasses should work with ServerLayout."""

        class _AllOnPolicy(ActivationPolicy):
            def __init__(self, n: int) -> None:
                self._n = n

            def active_mask(self, t: float) -> np.ndarray:
                return np.ones(self._n, dtype=bool)

        policy = _AllOnPolicy(10)
        mask = policy.active_mask(0.0)
        assert mask.sum() == 10
        assert policy.active_indices(0.0).tolist() == list(range(10))
