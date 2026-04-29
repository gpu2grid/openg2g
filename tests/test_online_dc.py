"""Tests for OnlineDatacenter pure-logic components.

Tests _RollingPowerBuffer, shared InferencePowerAugmenter integration,
_LoadGenerator.get_observed_itl, _parse_prometheus_text, and health
check functions without requiring real vLLM servers or zeusd instances.
"""

from __future__ import annotations

import collections
import contextlib
import threading
import time
from fractions import Fraction
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from zeus.monitor.power_streaming import PowerReadings, PowerStreamingClient

from openg2g.clock import SimulationClock
from openg2g.coordinator import SimulationLog
from openg2g.datacenter.config import DatacenterConfig, InferenceModelSpec, PowerAugmentationConfig
from openg2g.datacenter.layout import ServerPool
from openg2g.datacenter.online import (
    GPUEndpointMapping,
    LiveServerConfig,
    OnlineDatacenter,
    VLLMDeployment,
    _parse_prometheus_text,
    _RollingPowerBuffer,
)
from openg2g.datacenter.workloads.inference import InferencePowerAugmenter
from openg2g.events import EventEmitter

_EVENTS = EventEmitter(SimulationClock(Fraction(1, 10)), SimulationLog(), "custom")


def _make_spec(label: str, gpus_per_replica: int) -> InferenceModelSpec:
    return InferenceModelSpec(
        model_id="test/Model",
        model_label=label,
        gpu_model="H100",
        task="lm-arena-chat",
        gpus_per_replica=gpus_per_replica,
        itl_deadline_s=0.1,
        batch_sizes=(64, 128),
        feasible_batch_sizes=(64, 128),
    )


def _make_deployment(
    label: str = "test-model",
    num_replicas: int = 100,
    gpus_per_replica: int = 1,
    host: str = "node1",
    port: int = 4938,
    gpu_indices: tuple[int, ...] = (0,),
) -> VLLMDeployment:
    spec = _make_spec(label, gpus_per_replica)
    return VLLMDeployment(
        spec=spec,
        simulated_num_replicas=num_replicas,
        initial_batch_size=128,
        vllm_base_url=f"http://{host}:8000",
        gpu_endpoints=(GPUEndpointMapping(host=host, port=port, gpu_indices=gpu_indices),),
    )


def _fake_power_readings(
    deployments: list[VLLMDeployment],
    per_gpu_w: float = 300.0,
) -> dict[str, PowerReadings]:
    """Create fake power readings matching PowerStreamingClient output."""
    readings: dict[str, PowerReadings] = {}
    for d in deployments:
        for ep in d.gpu_endpoints:
            readings[ep.endpoint_key] = PowerReadings(
                timestamp_s=0.0,
                gpu_power_w={idx: per_gpu_w for idx in ep.gpu_indices},
            )
    return readings


def _make_itl_stub(model_label: str, itl_window_s: float = 1.0):
    """Create a _LoadGenerator stub with just enough state for get_observed_itl."""
    from openg2g.datacenter.online import _LoadGenerator

    lg = _LoadGenerator.__new__(_LoadGenerator)
    lg._itl_window_s = itl_window_s
    lg._lock = threading.Lock()
    lg._itl_samples = {model_label: collections.deque(maxlen=100_000)}
    return lg


@contextlib.contextmanager
def _online_dc(
    deployments: list[VLLMDeployment],
    per_gpu_w: float = 300.0,
    stagger_buffer_s: float = 10.0,
    prometheus_poll_interval_s: float = 0,
):
    """Build an OnlineDatacenter with mocked power client and load gen.

    Yields the dc within a live STAGGER_BUFFER_S patch, so both __init__
    and any subsequent _warmup calls see the same value.
    """
    fake_client = MagicMock(spec=PowerStreamingClient)
    fake_client.get_power.return_value = _fake_power_readings(deployments, per_gpu_w)

    with (
        patch("openg2g.datacenter.online._LoadGenerator"),
        patch("openg2g.datacenter.online.PowerStreamingClient", return_value=fake_client),
        patch("openg2g.datacenter.online.STAGGER_BUFFER_S", stagger_buffer_s),
    ):
        dc = OnlineDatacenter(
            DatacenterConfig(gpus_per_server=8, base_kw_per_phase=0.0),
            deployments,
            name="test",
            dt_s=Fraction(1, 10),
            seed=42,
            power_augmentation=PowerAugmentationConfig(
                noise_fraction=0.0,
                amplitude_scale_range=(1.0, 1.0),
            ),
            live_server=LiveServerConfig(
                prometheus_poll_interval_s=prometheus_poll_interval_s,
            ),
        )
        yield dc, fake_client


class TestRollingPowerBuffer:
    def test_empty_buffer_returns_zeros(self) -> None:
        buf = _RollingPowerBuffer(["m1"])
        offsets = np.array([0.0, 1.0, 2.0])
        result = buf.sample_servers("m1", now=10.0, stagger_offsets=offsets)
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, 0.0)

    def test_single_sample(self) -> None:
        buf = _RollingPowerBuffer(["m1"])
        buf.append("m1", 5.0, 300.0)
        offsets = np.array([0.0, 1.0, 10.0])
        result = buf.sample_servers("m1", now=5.0, stagger_offsets=offsets)
        np.testing.assert_array_equal(result, 300.0)

    def test_stagger_lookup(self) -> None:
        buf = _RollingPowerBuffer(["m1"])
        buf.append("m1", 1.0, 100.0)
        buf.append("m1", 3.0, 300.0)
        buf.append("m1", 5.0, 500.0)

        offsets = np.array([0.0, 2.5, 4.5])
        result = buf.sample_servers("m1", now=5.0, stagger_offsets=offsets)
        assert result[0] == 500.0
        assert result[1] == 100.0
        assert result[2] == 100.0

    def test_before_first_returns_first(self) -> None:
        buf = _RollingPowerBuffer(["m1"])
        buf.append("m1", 5.0, 42.0)
        offsets = np.array([10.0])
        result = buf.sample_servers("m1", now=5.0, stagger_offsets=offsets)
        assert result[0] == 42.0

    def test_after_last_returns_last(self) -> None:
        buf = _RollingPowerBuffer(["m1"])
        buf.append("m1", 1.0, 100.0)
        buf.append("m1", 2.0, 200.0)
        offsets = np.array([0.0])
        result = buf.sample_servers("m1", now=10.0, stagger_offsets=offsets)
        assert result[0] == 200.0

    def test_clear(self) -> None:
        buf = _RollingPowerBuffer(["m1", "m2"])
        buf.append("m1", 1.0, 100.0)
        buf.append("m2", 1.0, 200.0)
        buf.clear()
        result = buf.sample_servers("m1", now=1.0, stagger_offsets=np.array([0.0]))
        assert result[0] == 0.0

    def test_shape_matches_offsets(self) -> None:
        buf = _RollingPowerBuffer(["m1"])
        buf.append("m1", 1.0, 100.0)
        for n in [1, 5, 20]:
            offsets = np.zeros(n)
            result = buf.sample_servers("m1", now=1.0, stagger_offsets=offsets)
            assert result.shape == (n,)


class TestOnlineAugmentationPipeline:
    """Test the shared InferencePowerAugmenter with ServerPool built for online mode."""

    _NUM_REPLICAS = 100
    _GPUS_PER_REPLICA = 1

    def _build_pool_and_augmenter(
        self,
        num_replicas: int = 100,
        gpus_per_replica: int = 1,
        gpus_per_server: int = 8,
        noise_fraction: float = 0.0,
        amplitude_scale_range: tuple[float, float] = (1.0, 1.0),
        seed: int = 42,
    ) -> tuple[ServerPool, InferencePowerAugmenter]:
        import math

        from openg2g.utils import split_integer_evenly

        total_gpus = num_replicas * gpus_per_replica
        num_servers = math.ceil(total_gpus / gpus_per_server)
        rng = np.random.default_rng(seed)

        sA, sB, sC = split_integer_evenly(num_servers, 3)
        phase_list = np.asarray(([0] * sA) + ([1] * sB) + ([2] * sC), dtype=int)
        rng.shuffle(phase_list)

        stagger_offsets = rng.uniform(0.0, 10.0, size=num_servers)
        amplitude_scales = rng.uniform(
            amplitude_scale_range[0],
            amplitude_scale_range[1],
            size=num_servers,
        )

        model_priorities = {"test-model": rng.permutation(num_servers).astype(int)}

        pool = ServerPool(
            num_servers=num_servers,
            gpus_per_server=gpus_per_server,
            phase_list=phase_list,
            stagger_offsets=stagger_offsets,
            amplitude_scales=amplitude_scales,
            noise_fraction=noise_fraction,
            model_priorities=model_priorities,
        )
        augmenter = InferencePowerAugmenter(
            pool=pool,
            gpus_per_replica_by_model={"test-model": gpus_per_replica},
            seed=seed + 12345,
        )
        return pool, augmenter

    def test_stagger_offsets_are_float(self) -> None:
        pool, _ = self._build_pool_and_augmenter()
        assert pool.stagger_offsets.dtype == np.float64

    def test_uniform_power_scaling(self) -> None:
        pool, augmenter = self._build_pool_and_augmenter(
            num_replicas=100,
            gpus_per_replica=1,
            gpus_per_server=8,
            noise_fraction=0.0,
            amplitude_scale_range=(1.0, 1.0),
        )
        per_gpu = np.full(pool.num_servers, 300.0)
        aug = augmenter.augment({"test-model": per_gpu}, {"test-model": 100})
        total = aug.power_w.a + aug.power_w.b + aug.power_w.c
        # 100 replicas * 1 GPU = 100 GPUs -> ceil(100/8) = 13 servers * 8 GPUs = 104 GPUs
        expected = 300.0 * 13 * 8
        assert total == pytest.approx(expected, rel=1e-3)

    def test_noise_adds_variance(self) -> None:
        pool, augmenter = self._build_pool_and_augmenter(noise_fraction=0.1)
        per_gpu = np.full(pool.num_servers, 300.0)
        values = []
        for _ in range(50):
            aug = augmenter.augment({"test-model": per_gpu}, {"test-model": 100})
            values.append(aug.power_w.a + aug.power_w.b + aug.power_w.c)
        assert np.std(values) > 0

    def test_phase_shares_from_pool(self) -> None:
        pool, _ = self._build_pool_and_augmenter()
        counts = np.bincount(pool.phase_list, minlength=3)
        assert counts.sum() == pool.num_servers
        for c in counts:
            assert c > 0

    def test_active_replicas_reported(self) -> None:
        pool, augmenter = self._build_pool_and_augmenter(num_replicas=100, gpus_per_replica=1, gpus_per_server=8)
        per_gpu = np.full(pool.num_servers, 100.0)
        aug = augmenter.augment({"test-model": per_gpu}, {"test-model": 100})
        # ceil(100/8) = 13 servers * 8 GPUs / 1 GPU per replica = 104
        assert aug.active_replicas_by_model["test-model"] == 104

    def test_power_nonnegative(self) -> None:
        pool, augmenter = self._build_pool_and_augmenter(noise_fraction=0.5)
        per_gpu = np.full(pool.num_servers, 1.0)
        for _ in range(100):
            aug = augmenter.augment({"test-model": per_gpu}, {"test-model": 100})
            assert aug.power_w.a >= 0.0
            assert aug.power_w.b >= 0.0
            assert aug.power_w.c >= 0.0


class TestLoadGeneratorITL:
    MODEL = "test-model"

    def test_get_observed_itl_empty(self) -> None:
        lg = _make_itl_stub(self.MODEL)
        assert np.isnan(lg.get_observed_itl(self.MODEL))

    def test_get_observed_itl_windowed(self) -> None:
        lg = _make_itl_stub(self.MODEL)
        now = time.monotonic()
        lg._itl_samples[self.MODEL].append((now - 5.0, 0.100))
        lg._itl_samples[self.MODEL].append((now - 0.5, 0.050))
        lg._itl_samples[self.MODEL].append((now - 0.1, 0.030))

        result = lg.get_observed_itl(self.MODEL, window_s=1.0)
        assert result == pytest.approx(0.040, rel=1e-6)

    def test_get_observed_itl_all_expired(self) -> None:
        lg = _make_itl_stub(self.MODEL, itl_window_s=0.01)
        lg._itl_samples[self.MODEL].append((time.monotonic() - 10.0, 0.050))

        assert np.isnan(lg.get_observed_itl(self.MODEL, window_s=0.01))

    def test_per_token_itl_samples_stored(self) -> None:
        """Verify per-token ITL samples are individual entries, not averages."""
        lg = _make_itl_stub(self.MODEL, itl_window_s=10.0)
        now = time.monotonic()
        for i in range(5):
            lg._itl_samples[self.MODEL].append((now - 0.1 * i, 0.010 + 0.002 * i))

        result = lg.get_observed_itl(self.MODEL, window_s=10.0)
        expected = sum(0.010 + 0.002 * i for i in range(5)) / 5
        assert result == pytest.approx(expected, rel=1e-6)


class TestParsePrometheusText:
    def test_basic_gauges(self) -> None:
        text = """# HELP vllm:num_requests_running Number of requests running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 32
# HELP vllm:kv_cache_usage_perc KV cache usage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="m"} 0.45
"""
        result = _parse_prometheus_text(text)
        assert result["num_requests_running"] == 32.0
        assert result["kv_cache_usage_perc"] == pytest.approx(0.45)

    def test_empty_text(self) -> None:
        assert _parse_prometheus_text("") == {}

    def test_comments_only(self) -> None:
        text = "# HELP something\n# TYPE something gauge\n"
        assert _parse_prometheus_text(text) == {}

    def test_summing_multiple_labels(self) -> None:
        text = """vllm:num_requests_running{model_name="a"} 10
vllm:num_requests_running{model_name="b"} 20
"""
        result = _parse_prometheus_text(text)
        assert result["num_requests_running"] == 30.0

    def test_all_metrics(self) -> None:
        text = """vllm:num_requests_running{} 5
vllm:num_requests_waiting{} 2
vllm:num_preemptions_total{} 1
vllm:kv_cache_usage_perc{} 0.78
"""
        result = _parse_prometheus_text(text)
        assert result["num_requests_running"] == 5.0
        assert result["num_requests_waiting"] == 2.0
        assert result["num_preemptions_total"] == 1.0
        assert result["kv_cache_usage_perc"] == pytest.approx(0.78)


class TestOnlineDatacenterStep:
    """Integration test for OnlineDatacenter.step() with a fake power client.

    Exercises the full path: power reading -> rolling buffer -> shared
    InferencePowerAugmenter -> three-phase power output, without requiring
    real vLLM servers or zeusd.
    """

    def test_step_produces_nonzero_power(self) -> None:
        dep = _make_deployment(num_replicas=100, gpu_indices=(0, 1, 2, 3))
        with _online_dc([dep], per_gpu_w=300.0) as (dc, _):
            dc._started = True
            clock = SimulationClock(tick_s=Fraction(1, 10))
            state = dc.step(clock, _EVENTS)

            total_power = state.power_w.a + state.power_w.b + state.power_w.c
            assert total_power > 0
            assert dep.model_label in state.augmented_power_w_by_model
            assert state.augmented_power_w_by_model[dep.model_label] > 0
            assert dep.model_label in state.measured_power_w_by_model
            assert state.measured_power_w_by_model[dep.model_label] == pytest.approx(300.0 * 4)

    def test_step_power_scales_with_replicas(self) -> None:
        powers = []
        for n_replicas in [50, 200]:
            dep = _make_deployment(num_replicas=n_replicas, gpu_indices=(0,))
            with _online_dc([dep], per_gpu_w=300.0) as (dc, _):
                dc._started = True
                clock = SimulationClock(tick_s=Fraction(1, 10))
                state = dc.step(clock, _EVENTS)
                powers.append(state.power_w.a + state.power_w.b + state.power_w.c)

        assert powers[1] > powers[0]
        # Ratio is approximate due to whole-server allocation (ceil effects)
        assert powers[1] / powers[0] == pytest.approx(200.0 / 50.0, rel=0.15)

    def test_phase_shares_from_layout(self) -> None:
        dep = _make_deployment(num_replicas=100, gpu_indices=(0,))
        with _online_dc([dep]) as (dc, _):
            dc._started = True
            clock = SimulationClock(tick_s=Fraction(1, 10))
            dc.step(clock, _EVENTS)
            shares = dc.phase_share_by_model
            assert dep.model_label in shares
            assert shares[dep.model_label].shape == (3,)
            assert shares[dep.model_label].sum() == pytest.approx(1.0)


class TestWarmup:
    """Test the warmup phase of OnlineDatacenter.start()."""

    def test_warmup_completes_when_saturated(self) -> None:
        dep = _make_deployment(num_replicas=100, gpu_indices=(0, 1))
        with _online_dc([dep], stagger_buffer_s=0.05) as (dc, _):
            prom_mock = MagicMock()
            prom_mock.get_latest.return_value = {dep.model_label: {"num_requests_running": 128.0}}
            dc._prometheus = prom_mock
            dc._warmup(timeout_s=5.0, poll_interval_s=0.01)

    def test_warmup_waits_for_buffer_after_saturation(self) -> None:
        dep = _make_deployment(num_replicas=100, gpu_indices=(0, 1))
        with _online_dc([dep], stagger_buffer_s=0.05) as (dc, _):
            prom_mock = MagicMock()
            prom_mock.get_latest.return_value = {dep.model_label: {"num_requests_running": 128.0}}
            dc._prometheus = prom_mock

            t_before = time.monotonic()
            dc._warmup(timeout_s=5.0, poll_interval_s=0.01)
            elapsed = time.monotonic() - t_before

            # Must wait at least stagger_buffer_s after saturation
            assert elapsed >= 0.05

    def test_warmup_timeout_raises_with_trajectory(self) -> None:
        dep = _make_deployment(num_replicas=100, gpu_indices=(0, 1))
        with _online_dc([dep], stagger_buffer_s=0.05) as (dc, _):
            prom_mock = MagicMock()
            prom_mock.get_latest.return_value = {dep.model_label: {"num_requests_running": 10.0}}
            dc._prometheus = prom_mock

            with pytest.raises(RuntimeError, match="Warmup timed out") as exc_info:
                dc._warmup(timeout_s=0.1, poll_interval_s=0.01)

            msg = str(exc_info.value)
            assert dep.model_label in msg
            assert "target: 128" in msg
            assert "reached: 10" in msg
            assert "t=0s:" in msg

    def test_warmup_no_prometheus_waits_for_buffer_only(self) -> None:
        dep = _make_deployment(num_replicas=100, gpu_indices=(0, 1))
        with _online_dc([dep], stagger_buffer_s=0.05, prometheus_poll_interval_s=0) as (dc, _):
            assert dc._prometheus is None

            t_before = time.monotonic()
            dc._warmup(timeout_s=5.0, poll_interval_s=0.01)
            elapsed = time.monotonic() - t_before

            assert elapsed >= 0.05


class TestHealthChecks:
    def test_check_vllm_health_failure(self) -> None:
        from openg2g.datacenter.online import _check_vllm_health

        with pytest.raises(RuntimeError, match="vLLM health check failed"):
            _check_vllm_health("http://nonexistent-host-12345:9999", timeout_s=0.5)

    def test_check_vllm_model_failure(self) -> None:
        from openg2g.datacenter.online import _check_vllm_model

        with pytest.raises(RuntimeError, match="vLLM model check failed"):
            _check_vllm_model("http://nonexistent-host-12345:9999", "some-model", timeout_s=0.5)

    def test_check_zeusd_health_failure(self) -> None:
        from openg2g.datacenter.online import _check_zeusd_health

        with pytest.raises(RuntimeError, match="zeusd health check failed"):
            _check_zeusd_health("nonexistent-host-12345", port=9999, timeout_s=0.5)
