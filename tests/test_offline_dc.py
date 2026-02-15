"""Tests for OfflineDatacenter: chunk buffer, batch change invalidation."""

from __future__ import annotations

import numpy as np
from mlenergy_data.modeling import ITLMixtureModel

from openg2g.clock import SimulationClock
from openg2g.datacenter.offline import (
    OfflineDatacenter,
    TraceByBatchCache,
    build_periodic_per_gpu_template,
)
from openg2g.models.spec import ModelSpec
from openg2g.types import Command, OfflineDatacenterState


def _make_simple_cache(dt: float = 0.1, T: float = 100.0) -> TraceByBatchCache:
    """Create a minimal TraceByBatchCache with synthetic data."""
    # Synthetic trace: 10 seconds, linear ramp from 100W to 200W
    t = np.linspace(0, 10, 100)
    p = np.linspace(100, 200, 100)

    traces_by_batch = {}
    for batch in [64, 128]:
        traces_by_batch[batch] = {
            "TestModel": {
                "t": t,
                "p": p * (batch / 128.0),  # Scale power with batch size
                "measured_gpus": 1,
                "is_total": True,
                "amp_jitter": (1.0, 1.0),
                "noise_std_frac": 0.0,
            }
        }

    cache = TraceByBatchCache(traces_by_batch)
    cache.build_templates(T=T, dt=dt)
    return cache


def test_step_returns_offline_state():
    cache = _make_simple_cache()
    model = ModelSpec(model_label="TestModel", replicas=10, gpus_per_replica=1)
    dc = OfflineDatacenter(
        trace_cache=cache,
        models=[model],
        dt=0.1,
        batch_init=128,
        gpus_per_server=8,
        seed=0,
        chunk_steps=10,
        ramp_t_start=9999,
        ramp_t_end=9999,
        ramp_floor=1.0,
        base_kW_per_phase=0.0,
    )

    clock = SimulationClock(tick_s=0.1)
    state = dc.step(clock)

    assert isinstance(state, OfflineDatacenterState)
    assert state.power_w.a >= 0
    assert state.power_w.b >= 0
    assert state.power_w.c >= 0
    assert "TestModel" in state.batch_size_by_model
    assert state.batch_size_by_model["TestModel"] == 128


def test_step_produces_correct_number_of_states():
    """Stepping through a chunk should produce chunk_steps states."""
    cache = _make_simple_cache()
    model = ModelSpec(model_label="TestModel", replicas=10, gpus_per_replica=1)
    dc = OfflineDatacenter(
        trace_cache=cache,
        models=[model],
        dt=0.1,
        batch_init=128,
        gpus_per_server=8,
        seed=0,
        chunk_steps=5,
        ramp_t_start=9999,
        ramp_t_end=9999,
        ramp_floor=1.0,
    )

    clock = SimulationClock(tick_s=0.1)
    states = []
    for _ in range(10):
        states.append(dc.step(clock))
        clock.advance()

    assert len(states) == 10
    # Times should be monotonically increasing
    times = [s.time_s for s in states]
    for i in range(1, len(times)):
        assert times[i] > times[i - 1]


def test_batch_change_takes_effect_at_chunk_boundary():
    """Batch change is recorded immediately but applied at the next chunk."""
    cache = _make_simple_cache()
    model = ModelSpec(model_label="TestModel", replicas=10, gpus_per_replica=1)
    dc = OfflineDatacenter(
        trace_cache=cache,
        models=[model],
        dt=0.1,
        batch_init=128,
        gpus_per_server=8,
        seed=0,
        chunk_steps=10,
        ramp_t_start=9999,
        ramp_t_end=9999,
        ramp_floor=1.0,
    )

    clock = SimulationClock(tick_s=0.1)

    # Step a few times at batch=128
    for _ in range(5):
        dc.step(clock)
        clock.advance()

    # Change batch size, recorded but deferred
    dc.apply_control(
        Command(
            target="datacenter",
            kind="set_batch_size",
            payload={"batch_size_by_model": {"TestModel": 64}},
        )
    )
    assert dc.batch_by_model["TestModel"] == 64

    # Remaining steps in the current chunk still report old batch
    state = dc.step(clock)
    assert state.batch_size_by_model["TestModel"] == 128
    clock.advance()

    # Consume rest of chunk (steps 6..9)
    for _ in range(4):
        dc.step(clock)
        clock.advance()

    # Next step triggers new chunk and now uses batch=64
    state = dc.step(clock)
    assert state.batch_size_by_model["TestModel"] == 64


def test_build_periodic_template_shape():
    """Template should have the right number of steps."""
    t = np.linspace(0, 10, 200)
    p = np.sin(t) * 100 + 200

    tpl = build_periodic_per_gpu_template(
        trace_t=t, trace_p_total=p, measured_gpus=2, dt=0.1, T=50.0
    )

    expected_steps = int(np.ceil(50.0 / 0.1)) + 1
    assert tpl.shape[0] == expected_steps
    assert np.all(tpl >= 0)


def test_offline_datacenter_emits_observed_itl_when_latency_fits_is_set():
    cache = _make_simple_cache()
    model = ModelSpec(model_label="TestModel", replicas=10, gpus_per_replica=1)
    fake_params = ITLMixtureModel(
        loc=0.01,
        pi_steady=0.8,
        sigma_steady=0.1,
        scale_steady=0.05,
        pi_stall=0.2,
        sigma_stall=0.2,
        scale_stall=0.1,
    )
    latency_fits = {"TestModel": {128: fake_params}}
    dc = OfflineDatacenter(
        trace_cache=cache,
        models=[model],
        dt=0.1,
        batch_init=128,
        gpus_per_server=8,
        seed=0,
        chunk_steps=10,
        ramp_t_start=9999,
        ramp_t_end=9999,
        ramp_floor=1.0,
        latency_fits=latency_fits,
    )

    state = dc.step(SimulationClock(tick_s=0.1))
    assert "TestModel" in state.observed_itl_s_by_model
    assert np.isfinite(state.observed_itl_s_by_model["TestModel"])
