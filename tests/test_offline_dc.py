"""Tests for OfflineDatacenter: step-by-step generation, batch changes."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from mlenergy_data.modeling import ITLMixtureModel

from openg2g.clock import SimulationClock
from openg2g.datacenter.config import DatacenterConfig
from openg2g.datacenter.offline import (
    ITLFitStore,
    OfflineDatacenter,
    OfflineDatacenterState,
    OfflineInferenceData,
    OfflineWorkload,
    PowerTemplateStore,
    PowerTrace,
    PowerTraceStore,
    _build_per_gpu_power_template,
)
from openg2g.models.spec import LLMInferenceModelSpec
from openg2g.types import DatacenterCommand, SetBatchSize

MODEL = LLMInferenceModelSpec(
    model_label="TestModel", num_replicas=10, gpus_per_replica=1, initial_batch_size=128, itl_deadline_s=0.1
)
DC_CFG = DatacenterConfig(gpus_per_server=8)


def _make_simple_store(dt: float = 0.1, T: float = 100.0) -> PowerTemplateStore:
    """Create a minimal PowerTemplateStore with synthetic data."""
    t = np.linspace(0, 10, 100)
    p = np.linspace(100, 200, 100)

    traces = {
        "TestModel": {
            64: PowerTrace(t_s=t, power_w=p * (64 / 128.0), measured_gpus=1),
            128: PowerTrace(t_s=t, power_w=p, measured_gpus=1),
        }
    }

    store = PowerTraceStore.from_traces(traces)
    return store.build_templates(duration_s=T, dt_s=dt)


def _make_workload(templates: PowerTemplateStore, itl_fits: ITLFitStore | None = None) -> OfflineWorkload:
    """Create an OfflineWorkload from templates and optional ITL fits."""
    return OfflineWorkload(
        inference_data=OfflineInferenceData(
            (MODEL,),
            power_templates=templates,
            itl_fits=itl_fits,
        ),
    )


def test_step_returns_offline_state():
    store = _make_simple_store()
    dc = OfflineDatacenter(DC_CFG, _make_workload(store), dt_s=Fraction(1, 10))

    clock = SimulationClock(tick_s=Fraction(1, 10))
    state = dc.step(clock)

    assert isinstance(state, OfflineDatacenterState)
    assert state.power_w.a >= 0
    assert state.power_w.b >= 0
    assert state.power_w.c >= 0
    assert "TestModel" in state.batch_size_by_model
    assert state.batch_size_by_model["TestModel"] == 128


def test_step_produces_correct_number_of_states():
    """Stepping produces one state per call with monotonically increasing times."""
    store = _make_simple_store()
    dc = OfflineDatacenter(DC_CFG, _make_workload(store), dt_s=Fraction(1, 10))

    clock = SimulationClock(tick_s=Fraction(1, 10))
    states = []
    for _ in range(10):
        states.append(dc.step(clock))
        clock.advance()

    assert len(states) == 10
    times = [s.time_s for s in states]
    for i in range(1, len(times)):
        assert times[i] > times[i - 1]


def test_batch_change_takes_effect_immediately():
    """Batch size change via apply_control takes effect on the very next step."""
    store = _make_simple_store()
    dc = OfflineDatacenter(DC_CFG, _make_workload(store), dt_s=Fraction(1, 10))

    clock = SimulationClock(tick_s=Fraction(1, 10))

    for _ in range(5):
        state = dc.step(clock)
        assert state.batch_size_by_model["TestModel"] == 128
        clock.advance()

    dc.apply_control(SetBatchSize(batch_size_by_model={"TestModel": 64}))
    assert dc.batch_by_model["TestModel"] == 64

    state = dc.step(clock)
    assert state.batch_size_by_model["TestModel"] == 64


def test_build_periodic_template_shape():
    """Template should have the right number of steps."""
    t = np.linspace(0, 10, 200)
    p = np.sin(t) * 100 + 200
    trace = PowerTrace(t_s=t, power_w=p, measured_gpus=2)

    tpl = _build_per_gpu_power_template(trace, dt_s=0.1, duration_s=50.0)

    expected_steps = int(np.ceil(50.0 / 0.1)) + 1
    assert tpl.shape[0] == expected_steps
    assert np.all(tpl >= 0)


def test_offline_datacenter_emits_observed_itl_when_latency_fits_is_set():
    store = _make_simple_store()
    fake_params = ITLMixtureModel(
        loc=0.01,
        pi_steady=0.8,
        sigma_steady=0.1,
        scale_steady=0.05,
        pi_stall=0.2,
        sigma_stall=0.2,
        scale_stall=0.1,
    )
    latency_fits = ITLFitStore({"TestModel": {128: fake_params}})
    dc = OfflineDatacenter(DC_CFG, _make_workload(store, itl_fits=latency_fits), dt_s=Fraction(1, 10))

    state = dc.step(SimulationClock(tick_s=Fraction(1, 10)))
    assert "TestModel" in state.observed_itl_s_by_model
    assert np.isfinite(state.observed_itl_s_by_model["TestModel"])


def test_apply_control_rejects_unknown_command():
    """apply_control raises TypeError for unsupported command types."""

    class _CustomCommand(DatacenterCommand):
        pass

    store = _make_simple_store()
    dc = OfflineDatacenter(DC_CFG, _make_workload(store), dt_s=Fraction(1, 10))

    with pytest.raises(TypeError, match="OfflineDatacenter does not support"):
        dc.apply_control(_CustomCommand())
