"""Tests for OnlineDatacenter pure-logic components.

Tests _PowerAugmenter, _lookup_power, _LoadGenerator.get_observed_itl,
_parse_prometheus_text, and health check functions without requiring
real vLLM servers or zeusd instances.
"""

from __future__ import annotations

import collections
import time

import numpy as np
import pytest
from zeus.monitor.power_streaming import PowerReadings

from openg2g.datacenter.online import (
    GPUEndpointMapping,
    LoadGenerationConfig,
    OnlineModelDeployment,
    PowerAugmentationConfig,
    _lookup_power,
    _parse_prometheus_text,
    _PowerAugmenter,
)
from openg2g.grid.base import Phase
from openg2g.models.spec import LLMInferenceModelSpec


def _make_deployment(
    label: str = "test-model",
    num_replicas: int = 100,
    gpus_per_replica: int = 1,
    host: str = "node1",
    port: int = 4938,
    gpu_indices: tuple[int, ...] = (0,),
    phase: Phase = Phase.A,
) -> OnlineModelDeployment:
    spec = LLMInferenceModelSpec(
        model_label=label,
        num_replicas=num_replicas,
        gpus_per_replica=gpus_per_replica,
        initial_batch_size=128,
        itl_deadline_s=0.1,
    )
    return OnlineModelDeployment(
        spec=spec,
        vllm_base_url=f"http://{host}:8000",
        model_name=f"org/{label}",
        gpu_endpoints=(GPUEndpointMapping(host=host, port=port, gpu_indices=gpu_indices, phase=phase),),
    )


def _make_readings(gpu_power: dict[int, float], timestamp_s: float = 0.0) -> PowerReadings:
    return PowerReadings(timestamp_s=timestamp_s, gpu_power_w=gpu_power)


class TestLookupPower:
    def test_empty_buffer(self) -> None:
        buf: collections.deque[tuple[float, float]] = collections.deque()
        assert _lookup_power(buf, 1.0) == 0.0

    def test_before_first(self) -> None:
        buf: collections.deque[tuple[float, float]] = collections.deque([(5.0, 100.0), (6.0, 200.0)])
        assert _lookup_power(buf, 3.0) == 100.0

    def test_after_last(self) -> None:
        buf: collections.deque[tuple[float, float]] = collections.deque([(5.0, 100.0), (6.0, 200.0)])
        assert _lookup_power(buf, 10.0) == 200.0

    def test_exact_match(self) -> None:
        buf: collections.deque[tuple[float, float]] = collections.deque([(1.0, 100.0), (2.0, 200.0), (3.0, 300.0)])
        assert _lookup_power(buf, 2.0) == 200.0

    def test_between_samples(self) -> None:
        buf: collections.deque[tuple[float, float]] = collections.deque([(1.0, 100.0), (3.0, 300.0)])
        assert _lookup_power(buf, 2.5) == 100.0

    def test_single_sample(self) -> None:
        buf: collections.deque[tuple[float, float]] = collections.deque([(5.0, 42.0)])
        assert _lookup_power(buf, 5.0) == 42.0
        assert _lookup_power(buf, 0.0) == 42.0
        assert _lookup_power(buf, 99.0) == 42.0


class TestPowerAugmenter:
    def _make_augmenter(
        self,
        num_replicas: int = 100,
        num_virtual_groups: int = 4,
        noise_frac: float = 0.0,
        stagger_buffer_s: float = 5.0,
    ) -> tuple[_PowerAugmenter, OnlineModelDeployment]:
        dep = _make_deployment(num_replicas=num_replicas)
        config = PowerAugmentationConfig(
            noise_frac=noise_frac,
            stagger_buffer_s=stagger_buffer_s,
            num_virtual_groups=num_virtual_groups,
            seed=42,
        )
        aug = _PowerAugmenter([dep], config)
        return aug, dep

    def test_empty_buffer_returns_zero(self) -> None:
        aug, dep = self._make_augmenter()
        total, per_gpu = aug.augmented_power(dep.model_label)
        assert total == 0.0
        assert per_gpu == 0.0

    def test_scaling_factor(self) -> None:
        aug, dep = self._make_augmenter(num_replicas=100, num_virtual_groups=1, noise_frac=0.0)
        readings = {"node1:4938": _make_readings({0: 300.0})}
        aug.update(readings, {dep.model_label: dep})
        total, per_gpu = aug.augmented_power(dep.model_label)
        assert per_gpu == 300.0
        assert total == pytest.approx(300.0 * 100 * 1, rel=1e-3)

    def test_noise_adds_variance(self) -> None:
        aug, dep = self._make_augmenter(num_replicas=100, num_virtual_groups=1, noise_frac=0.1)
        readings = {"node1:4938": _make_readings({0: 300.0})}
        aug.update(readings, {dep.model_label: dep})

        values = []
        for _ in range(50):
            total, _ = aug.augmented_power(dep.model_label)
            values.append(total)
        assert np.std(values) > 0

    def test_multi_gpu_averaging(self) -> None:
        dep = _make_deployment(gpu_indices=(0, 1, 2, 3))
        config = PowerAugmentationConfig(
            noise_frac=0.0,
            stagger_buffer_s=5.0,
            num_virtual_groups=1,
            seed=42,
        )
        aug = _PowerAugmenter([dep], config)

        readings = {"node1:4938": _make_readings({0: 200.0, 1: 300.0, 2: 400.0, 3: 500.0})}
        aug.update(readings, {dep.model_label: dep})
        total, per_gpu = aug.augmented_power(dep.model_label)
        assert per_gpu == 350.0
        assert total == pytest.approx(350.0 * 100 * 1, rel=1e-3)

    def test_augmented_power_nonnegative(self) -> None:
        aug, dep = self._make_augmenter(num_replicas=10, noise_frac=0.5)
        readings = {"node1:4938": _make_readings({0: 1.0})}
        aug.update(readings, {dep.model_label: dep})
        for _ in range(100):
            total, _ = aug.augmented_power(dep.model_label)
            assert total >= 0.0


class TestLoadGeneratorITL:
    def test_get_observed_itl_empty(self) -> None:
        from openg2g.datacenter.online import _LoadGenerator

        dep = _make_deployment()
        lg = _LoadGenerator.__new__(_LoadGenerator)
        lg._config = LoadGenerationConfig(itl_window_s=1.0)
        lg._lock = __import__("threading").Lock()
        lg._itl_samples = {dep.model_label: collections.deque(maxlen=100_000)}

        assert np.isnan(lg.get_observed_itl(dep.model_label))

    def test_get_observed_itl_windowed(self) -> None:
        from openg2g.datacenter.online import _LoadGenerator

        dep = _make_deployment()
        lg = _LoadGenerator.__new__(_LoadGenerator)
        lg._config = LoadGenerationConfig(itl_window_s=1.0)
        lg._lock = __import__("threading").Lock()
        lg._itl_samples = {dep.model_label: collections.deque(maxlen=100_000)}

        now = time.monotonic()
        lg._itl_samples[dep.model_label].append((now - 5.0, 0.100))
        lg._itl_samples[dep.model_label].append((now - 0.5, 0.050))
        lg._itl_samples[dep.model_label].append((now - 0.1, 0.030))

        result = lg.get_observed_itl(dep.model_label, window_s=1.0)
        assert result == pytest.approx(0.040, rel=1e-6)

    def test_get_observed_itl_all_expired(self) -> None:
        from openg2g.datacenter.online import _LoadGenerator

        dep = _make_deployment()
        lg = _LoadGenerator.__new__(_LoadGenerator)
        lg._config = LoadGenerationConfig(itl_window_s=0.01)
        lg._lock = __import__("threading").Lock()
        lg._itl_samples = {dep.model_label: collections.deque(maxlen=100_000)}

        lg._itl_samples[dep.model_label].append((time.monotonic() - 10.0, 0.050))

        assert np.isnan(lg.get_observed_itl(dep.model_label, window_s=0.01))

    def test_per_token_itl_samples_stored(self) -> None:
        """Verify per-token ITL samples are individual entries, not averages."""
        from openg2g.datacenter.online import _LoadGenerator

        dep = _make_deployment()
        lg = _LoadGenerator.__new__(_LoadGenerator)
        lg._config = LoadGenerationConfig(itl_window_s=10.0)
        lg._lock = __import__("threading").Lock()
        lg._itl_samples = {dep.model_label: collections.deque(maxlen=100_000)}

        now = time.monotonic()
        # Simulate 5 individual token ITL samples
        for i in range(5):
            lg._itl_samples[dep.model_label].append((now - 0.1 * i, 0.010 + 0.002 * i))

        result = lg.get_observed_itl(dep.model_label, window_s=10.0)
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
