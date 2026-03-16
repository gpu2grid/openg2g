"""Online (live GPU) datacenter backend with power augmentation.

Connects to real vLLM inference servers for load generation and ITL
measurement, and to zeusd instances for live GPU power monitoring.
Power readings from a small number of real GPUs are augmented to
datacenter scale using the shared
[`InferencePowerAugmenter`][openg2g.datacenter.workloads.inference.InferencePowerAugmenter]
pipeline.

Requires `pip install zeus aiohttp`.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import functools
import json
import logging
import math
import re
import threading
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from pydantic import BaseModel, ConfigDict
from zeus.monitor.power_streaming import PowerStreamingClient
from zeus.utils.zeusd import ZeusdConfig

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter, LLMDatacenterState
from openg2g.datacenter.command import DatacenterCommand, SetBatchSize
from openg2g.datacenter.config import (
    DatacenterConfig,
    InferenceModelSpec,
    InferenceRampSchedule,
    PowerAugmentationConfig,
)
from openg2g.datacenter.layout import (
    ActivationPolicy,
    RampActivationPolicy,
    ServerLayout,
)
from openg2g.datacenter.workloads.inference import (
    InferencePowerAugmenter,
    RequestStore,
)
from openg2g.events import EventEmitter
from openg2g.utils import split_integer_evenly

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnlineDatacenterState(LLMDatacenterState):
    """Extended state from the online (live GPU) backend.

    The base `power_w`
    field carries the augmented three-phase power (what the grid sees).
    This subclass adds the measured (pre-augmentation) breakdown for
    post-hoc analysis.

    Attributes:
        measured_power_w: Total measured three-phase power from real GPUs
            (before augmentation), plus base load.
        measured_power_w_by_model: Per-model total measured power from real
            GPUs (watts).
        augmented_power_w_by_model: Per-model augmented power (watts). This
            is the power fed to the grid for each model after scaling up.
        augmentation_factor_by_model: Per-model augmentation multiplier
            (virtual replicas / real replicas).
        prometheus_metrics_by_model: Per-model Prometheus metrics snapshot.
            Keys are model labels, values are dicts with metric names like
            `num_requests_running`, `num_requests_waiting`,
            `kv_cache_usage_perc`, `num_preemptions_total`.
    """

    measured_power_w: ThreePhase = field(default_factory=lambda: ThreePhase(a=0.0, b=0.0, c=0.0))
    measured_power_w_by_model: dict[str, float] = field(default_factory=dict)
    augmented_power_w_by_model: dict[str, float] = field(default_factory=dict)
    augmentation_factor_by_model: dict[str, float] = field(default_factory=dict)
    prometheus_metrics_by_model: dict[str, dict[str, float]] = field(default_factory=dict)


class GPUEndpointMapping(BaseModel):
    """Maps a zeusd endpoint to specific GPUs.

    Attributes:
        host: Hostname or IP of the zeusd instance.
        port: TCP port of the zeusd instance.
        gpu_indices: GPU device indices to monitor on this endpoint.
    """

    model_config = ConfigDict(frozen=True)

    host: str
    port: int = 4938
    gpu_indices: tuple[int, ...] = (0,)

    @property
    def endpoint_key(self) -> str:
        """Return the `host:port` key used by `PowerStreamingClient`."""
        return f"{self.host}:{self.port}"


class VLLMDeployment(BaseModel):
    """Deployment of one LLM model on a vLLM server.

    !!! Warning
        vLLM must be a patched version with the `POST /set_max_num_seqs`
        endpoint implemented.

    Pairs a reusable
    [`InferenceModelSpec`][openg2g.datacenter.config.InferenceModelSpec]
    with physical deployment details. `spec.initial_num_replicas` is the
    simulated (augmented) count for grid simulation. The real replica
    count is derived from `gpu_endpoints` and `spec.gpus_per_replica`.

    Tracks the current batch size (`max_num_seqs`) and provides
    `set_batch_size()` to update it on the vLLM server.

    Attributes:
        spec: Model specification (shared with offline datacenter).
        vllm_base_url: Base URL of the vLLM server (e.g. `http://node1:8000`).
        gpu_endpoints: GPU endpoint mappings for power monitoring.
        request_extra_body: Extra fields merged into every request dict
            for this model (e.g. `chat_template_kwargs`).
        batch_size: Current batch size (`max_num_seqs`). Initialized from
            `spec.initial_batch_size` if not set explicitly.
    """

    spec: InferenceModelSpec
    vllm_base_url: str
    gpu_endpoints: tuple[GPUEndpointMapping, ...] = ()
    request_extra_body: dict[str, Any] | None = None
    batch_size: int = 0

    def model_post_init(self, __context: Any) -> None:
        if self.batch_size == 0:
            self.batch_size = self.spec.initial_batch_size

    @property
    def model_label(self) -> str:
        return self.spec.model_label

    @property
    def num_real_gpus(self) -> int:
        """Total number of real GPUs for this model across all endpoints."""
        return sum(len(ep.gpu_indices) for ep in self.gpu_endpoints)

    @property
    def num_real_replicas(self) -> int:
        """Number of real replicas (real GPUs / GPUs per replica)."""
        return self.num_real_gpus // max(self.spec.gpus_per_replica, 1)

    @property
    def augmentation_factor(self) -> float:
        """Ratio of simulated replicas to real replicas."""
        return self.spec.initial_num_replicas / max(self.num_real_replicas, 1)

    def set_batch_size(self, batch_size: int, ramp_up_rate: float = 0.0) -> None:
        """Update batch size on the vLLM server and track it locally.

        Sends `POST /set_max_num_seqs` to the vLLM server.

        Args:
            batch_size: New batch size (max_num_seqs) to set.
            ramp_up_rate: Optional ramp-up rate for gradual increase.
        """
        old = self.batch_size
        url = f"{self.vllm_base_url}/set_max_num_seqs?max_num_seqs={batch_size}"
        if ramp_up_rate > 0:
            url += f"&ramp_up_rate={ramp_up_rate}"
        try:
            req = urllib.request.Request(url, method="POST", data=b"")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status >= 400:
                    raise RuntimeError(
                        f"Failed to set batch size {batch_size} on {self.vllm_base_url}: HTTP {resp.status}"
                    )
        except Exception:
            logger.error(
                "Failed to set batch size %d on %s (keeping old=%d)",
                batch_size,
                self.vllm_base_url,
                old,
                exc_info=True,
            )
            raise
        self.batch_size = batch_size
        if old != batch_size:
            logger.info("Batch size %s: %d -> %d", self.model_label, old, batch_size)


class LiveServerConfig(BaseModel):
    """Configuration for interacting with live vLLM servers.

    Groups settings related to load generation, ITL measurement, and
    Prometheus monitoring. The online counterpart of offline's
    trace/template data.

    Attributes:
        requests_dir: Directory containing per-model JSONL request files
            (e.g. `{model_label}.jsonl`). If `None`, a minimal fallback
            request is used for each model.
        prometheus_poll_interval_s: How often to poll vLLM /metrics for
            request counts and saturation monitoring. Set to 0 to disable.
        max_output_tokens: Token limit for generated load requests (used
            by the fallback request when no JSONL requests are provided).
        itl_window_s: Sliding window for ITL averaging (seconds).
    """

    model_config = ConfigDict(frozen=True)

    requests_dir: Path | None = None
    prometheus_poll_interval_s: float = 0.5
    max_output_tokens: int = 512
    itl_window_s: float = 1.0


STAGGER_BUFFER_S: float = 10.0
"""Seconds of power history for temporal staggering.

Also used as the stagger range when building
[`ServerLayout`][openg2g.datacenter.layout.ServerLayout]
(float offsets drawn from `[0, STAGGER_BUFFER_S)`).

Not user-configurable. Patchable for testing via
`openg2g.datacenter.online.STAGGER_BUFFER_S = ...`.
"""


def _check_vllm_health(base_url: str, timeout_s: float = 10.0) -> None:
    """Verify a vLLM server is reachable via GET /health.

    Args:
        base_url: Base URL of the vLLM server (e.g. `http://node1:8000`).
        timeout_s: HTTP timeout in seconds.

    Raises:
        RuntimeError: If the server is not reachable or unhealthy.
    """
    url = f"{base_url}/health"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status != 200:
                raise RuntimeError(f"vLLM health check failed: HTTP {resp.status} from {url}")
    except Exception as e:
        raise RuntimeError(f"vLLM health check failed for {url}: {e}") from e


def _check_vllm_model(base_url: str, expected_model: str, timeout_s: float = 10.0) -> None:
    """Verify a vLLM server is serving the expected model via GET /v1/models.

    Args:
        base_url: Base URL of the vLLM server.
        expected_model: Model ID to expect in the response.
        timeout_s: HTTP timeout in seconds.

    Raises:
        RuntimeError: If the model is not served or the endpoint is unreachable.
    """
    url = f"{base_url}/v1/models"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status != 200:
                raise RuntimeError(f"vLLM model check failed: HTTP {resp.status} from {url}")
            data = json.loads(resp.read().decode())
            served = [m["id"] for m in data.get("data", [])]
            if expected_model not in served:
                raise RuntimeError(f"vLLM at {base_url} serves {served}, expected '{expected_model}'")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"vLLM model check failed for {url}: {e}") from e


def _check_zeusd_health(host: str, port: int = 4938, timeout_s: float = 10.0) -> None:
    """Verify a zeusd instance is reachable via GET /discover.

    Args:
        host: Hostname of the zeusd instance.
        port: TCP port.
        timeout_s: HTTP timeout in seconds.

    Raises:
        RuntimeError: If the zeusd instance is unreachable.
    """
    url = f"http://{host}:{port}/discover"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status != 200:
                raise RuntimeError(f"zeusd health check failed: HTTP {resp.status} from {url}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"zeusd health check failed for {url}: {e}") from e


_GAUGE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{.*?\}\s+(.+)$|^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(.+)$")

_PROMETHEUS_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
    "vllm:kv_cache_usage_perc",
)


def _parse_prometheus_text(text: str) -> dict[str, float]:
    """Parse Prometheus text-format metrics and extract vLLM gauges.

    Returns a dict with metric names (without `vllm:` prefix) mapped to
    their summed values.
    """
    raw: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _GAUGE_RE.match(line)
        if m:
            name = m.group(1) or m.group(3)
            val_str = m.group(2) or m.group(4)
            if name in _PROMETHEUS_METRICS:
                with contextlib.suppress(ValueError):
                    raw[name] = raw.get(name, 0.0) + float(val_str)

    result: dict[str, float] = {}
    for metric in _PROMETHEUS_METRICS:
        if metric in raw:
            short = metric.removeprefix("vllm:")
            result[short] = raw[metric]
    return result


class _PrometheusPoller:
    """Polls vLLM /metrics endpoints for Prometheus gauges.

    Runs as an async task inside `_LoadGenerator`'s event loop.
    Provides thread-safe access to the latest snapshot per model.
    """

    def __init__(
        self,
        deployments: Sequence[VLLMDeployment],
        poll_interval_s: float = 0.5,
    ) -> None:
        self._deployments = {d.model_label: d for d in deployments}
        self._poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._latest: dict[str, dict[str, float]] = {}

    def get_latest(self) -> dict[str, dict[str, float]]:
        """Return the latest metrics snapshot per model (thread-safe)."""
        with self._lock:
            return dict(self._latest)

    async def run(self, stop_event: threading.Event) -> None:
        """Poll loop. Call as an asyncio task."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
            while not stop_event.is_set():
                for label, dep in self._deployments.items():
                    url = f"{dep.vllm_base_url}/metrics"
                    try:
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                text = await resp.text()
                                metrics = _parse_prometheus_text(text)
                                with self._lock:
                                    self._latest[label] = metrics
                    except Exception:
                        logger.debug("Prometheus poll failed for %s", label, exc_info=True)
                await asyncio.sleep(self._poll_interval_s)


class _LoadGenerator:
    """Background load generator that saturates vLLM servers and measures ITL.

    Runs a daemon thread with an asyncio event loop. For each model, a
    semaphore-gated producer loop cycles through pre-built request dicts
    endlessly. The semaphore size is `2 * max(feasible_batch_sizes)`,
    ensuring the vLLM queue never drains even at the largest batch size
    the OFO controller can set. Per-token inter-token latency (ITL) is
    measured from SSE chunk arrival times using `usage.completion_tokens`
    increments; first-token latency (TTFT) is excluded from ITL samples.
    """

    def __init__(
        self,
        deployments: Sequence[VLLMDeployment],
        *,
        request_store: RequestStore | None = None,
        max_output_tokens: int = 512,
        itl_window_s: float = 1.0,
        prometheus_poller: _PrometheusPoller | None = None,
    ) -> None:
        self._deployments = {d.model_label: d for d in deployments}
        self._requests: dict[str, list[dict]] = {}
        if request_store is not None:
            self._requests = dict(request_store.requests_by_model)
        self._max_output_tokens = max_output_tokens
        self._itl_window_s = itl_window_s
        self._prometheus = prometheus_poller

        self._lock = threading.Lock()
        self._itl_samples: dict[str, collections.deque[tuple[float, float]]] = {}
        for d in deployments:
            self._itl_samples[d.model_label] = collections.deque()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("LoadGenerator already started")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_thread,
            name="load-generator",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def get_observed_itl(self, model_label: str, window_s: float | None = None) -> float:
        """Return the windowed-average ITL for *model_label*, or NaN."""
        if window_s is None:
            window_s = self._itl_window_s
        cutoff = time.monotonic() - window_s
        with self._lock:
            samples = self._itl_samples.get(model_label)
            if not samples:
                return float("nan")
            recent = [itl for ts, itl in samples if ts >= cutoff]
        if not recent:
            return float("nan")
        return sum(recent) / len(recent)

    def _run_thread(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_async())
        except Exception:
            if not self._stop_event.is_set():
                logger.exception("LoadGenerator thread crashed")
        finally:
            self._loop.close()
            self._loop = None

    async def _run_async(self) -> None:
        tasks: list[asyncio.Task] = []

        for label, dep in self._deployments.items():
            tasks.append(asyncio.create_task(self._model_producer(label, dep)))

        if self._prometheus is not None:
            tasks.append(asyncio.create_task(self._prometheus.run(self._stop_event)))

        while not self._stop_event.is_set():
            await asyncio.sleep(0.1)

        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _model_producer(self, label: str, dep: VLLMDeployment) -> None:
        """Semaphore-gated loop that continuously submits requests for one model.

        Cycles through the JSONL request list endlessly. The semaphore
        limits in-flight requests to `2 * max(feasible_batch_sizes)`,
        ensuring the vLLM server always has a non-empty queue.
        """
        max_batch = max(dep.spec.feasible_batch_sizes)
        sem = asyncio.Semaphore(2 * max_batch)
        requests = self._requests.get(label, [])
        req_idx = 0
        active: set[asyncio.Task[None]] = set()

        connector = aiohttp.TCPConnector(limit=0, ssl=False)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300.0),
            connector=connector,
        ) as session:
            while not self._stop_event.is_set():
                await sem.acquire()
                if self._stop_event.is_set():
                    break
                if requests:
                    request_dict = requests[req_idx % len(requests)]
                    req_idx += 1
                else:
                    request_dict = self._default_request(dep)
                task = asyncio.create_task(self._single_request(label, dep, request_dict, session, sem))
                active.add(task)
                task.add_done_callback(active.discard)

    def _default_request(self, dep: VLLMDeployment) -> dict:
        """Build a minimal fallback request dict."""
        return {
            "model": dep.spec.model_id,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_completion_tokens": self._max_output_tokens,
        }

    async def _single_request(
        self,
        label: str,
        dep: VLLMDeployment,
        request_dict: dict,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
    ) -> None:
        """Send one streaming chat-completion request and measure decoding ITL.

        Uses `usage.completion_tokens` increments to correctly handle
        multi-token bundles. First-token samples (TTFT) are skipped;
        only decoding-phase ITL is recorded.
        """
        try:
            url = f"{dep.vllm_base_url}/v1/chat/completions"
            body = dict(request_dict)
            body["stream"] = True
            body["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
            if "max_tokens" in body and "max_completion_tokens" not in body:
                body["max_completion_tokens"] = body.pop("max_tokens")

            current_completion_tokens = 0
            most_recent_timestamp = time.perf_counter()
            ttft_recorded = False

            async with session.post(url, json=body) as response:
                if response.status != 200:
                    return
                async for chunk_bytes in response.content:
                    if self._stop_event.is_set():
                        return
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    chunk_str = chunk_bytes.decode("utf-8")

                    if chunk_str.startswith(":"):
                        continue

                    data_str = chunk_str.removeprefix("data: ")
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    usage = data.get("usage")
                    completion_tokens = usage and usage.get("completion_tokens")
                    if not completion_tokens:
                        continue

                    timestamp = time.perf_counter()

                    if not ttft_recorded:
                        ttft_recorded = True
                        current_completion_tokens = completion_tokens
                    else:
                        itl = timestamp - most_recent_timestamp
                        inc = completion_tokens - current_completion_tokens
                        current_completion_tokens = completion_tokens

                        now_mono = time.monotonic()
                        with self._lock:
                            self._itl_samples[label].append((now_mono, itl))
                            for _ in range(max(inc - 1, 0)):
                                self._itl_samples[label].append((now_mono, 0.0))

                    most_recent_timestamp = timestamp

        except Exception:
            if not self._stop_event.is_set():
                logger.debug("Request to %s failed for %s", dep.vllm_base_url, label, exc_info=True)
        finally:
            sem.release()


class _RollingPowerBuffer:
    """Per-model rolling buffer of (timestamp, per_gpu_watts) readings.

    Provides `sample_servers()` to look up historical per-GPU power at
    different time offsets for each virtual server, enabling temporal
    staggering of batch-size-change transients.
    """

    def __init__(self, model_labels: Sequence[str], max_samples: int = 10000) -> None:
        self._buffers: dict[str, collections.deque[tuple[float, float]]] = {
            label: collections.deque(maxlen=max_samples) for label in model_labels
        }

    def append(self, label: str, timestamp: float, per_gpu_w: float) -> None:
        """Feed a new per-GPU power reading for a model."""
        self._buffers[label].append((timestamp, per_gpu_w))

    def sample_servers(
        self,
        label: str,
        now: float,
        stagger_offsets: np.ndarray,
    ) -> np.ndarray:
        """Look up per-GPU power at `now - offset[i]` for each virtual server.

        Args:
            label: Model label.
            now: Current wall-clock time (monotonic).
            stagger_offsets: Per-server time offsets (seconds), shape `(N,)`.

        Returns:
            Array of shape `(N,)` with per-GPU power for each server.
        """
        buf = self._buffers[label]
        n = len(stagger_offsets)
        result = np.zeros(n, dtype=float)
        if not buf:
            return result
        for i in range(n):
            result[i] = self._lookup(buf, now - stagger_offsets[i])
        return result

    def clear(self) -> None:
        """Clear all buffers."""
        for buf in self._buffers.values():
            buf.clear()

    @staticmethod
    def _lookup(buf: collections.deque[tuple[float, float]], target_t: float) -> float:
        """Find the power reading at or just before `target_t`."""
        if not buf:
            return 0.0
        if target_t <= buf[0][0]:
            return buf[0][1]
        if target_t >= buf[-1][0]:
            return buf[-1][1]
        for i in range(len(buf) - 1, -1, -1):
            if buf[i][0] <= target_t:
                return buf[i][1]
        return buf[0][1]


class OnlineDatacenter(LLMBatchSizeControlledDatacenter[OnlineDatacenterState]):
    """Live GPU datacenter backend with power augmentation.

    Dispatches inference load to vLLM servers, streams GPU power from
    zeusd, measures ITL from streaming responses, and augments power
    readings to datacenter scale using the shared
    [`InferencePowerAugmenter`][openg2g.datacenter.workloads.inference.InferencePowerAugmenter]
    pipeline (same as
    [`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter]).

    Call [`start`][.start] before the first [`step`][.step] and
    [`stop`][.stop] after the simulation loop finishes.

    `PowerStreamingClient` is constructed internally from the GPU
    endpoints declared in each deployment. Health checks are always
    performed during [`start`][.start].

    Args:
        datacenter: Facility configuration (GPUs per server, base load).
        deployments: Model deployments with physical hardware mapping.
        dt_s: Simulation timestep (seconds).
        seed: Random seed for layout generation and noise.
        power_augmentation: Per-server amplitude scaling and noise
            settings.
        inference_ramps: Inference server ramp event(s). `None` keeps
            all servers active.
        live_server: Configuration for interacting with live vLLM
            servers. Request data is loaded from
            `LiveServerConfig.requests_dir`.
    """

    def __init__(
        self,
        datacenter: DatacenterConfig,
        deployments: Sequence[VLLMDeployment],
        *,
        dt_s: Fraction = Fraction(1, 10),
        seed: int = 0,
        power_augmentation: PowerAugmentationConfig | None = None,
        inference_ramps: InferenceRampSchedule | None = None,
        live_server: LiveServerConfig | None = None,
    ) -> None:
        super().__init__()
        if power_augmentation is None:
            power_augmentation = PowerAugmentationConfig()
        if live_server is None:
            live_server = LiveServerConfig()
        self._dt_s = dt_s
        self._seed = int(seed)
        self._deployments = list(deployments)
        self._deployment_map = {d.model_label: d for d in deployments}
        self._datacenter_config = datacenter
        self._power_augmentation = power_augmentation
        self._live_server_config = live_server

        self._base_W_per_phase = float(datacenter.base_kw_per_phase) * 1e3
        self._inference_ramp_schedule = inference_ramps if inference_ramps is not None else InferenceRampSchedule()

        servers_by_key: dict[str, ZeusdConfig] = {}
        gpu_indices_by_key: dict[str, list[int]] = {}
        for d in self._deployments:
            for ep in d.gpu_endpoints:
                key = ep.endpoint_key
                if key not in gpu_indices_by_key:
                    gpu_indices_by_key[key] = []
                for idx in ep.gpu_indices:
                    if idx not in gpu_indices_by_key[key]:
                        gpu_indices_by_key[key].append(idx)
                servers_by_key[key] = ZeusdConfig.tcp(
                    ep.host,
                    ep.port,
                    gpu_indices=gpu_indices_by_key[key],
                    cpu_indices=[],
                )
        self._power_client = PowerStreamingClient(servers=list(servers_by_key.values()))

        self._prometheus = (
            _PrometheusPoller(
                deployments,
                poll_interval_s=live_server.prometheus_poll_interval_s,
            )
            if live_server.prometheus_poll_interval_s > 0
            else None
        )

        self._request_store = RequestStore.load(live_server.requests_dir) if live_server.requests_dir else None
        self._load_gen = _LoadGenerator(
            deployments,
            request_store=self._request_store,
            max_output_tokens=live_server.max_output_tokens,
            itl_window_s=live_server.itl_window_s,
            prometheus_poller=self._prometheus,
        )

        self._layout_rng = np.random.default_rng(self._seed)
        self._layouts: dict[str, ServerLayout] = {}
        self._policies: dict[str, ActivationPolicy] = {}
        self._build_all_layouts()
        self._inference_augmenter = InferencePowerAugmenter(
            layouts=self._layouts,
            policies=self._policies,
            seed=self._seed + 12345,
        )
        self._rolling_buffer = _RollingPowerBuffer(
            [d.model_label for d in deployments],
            max_samples=max(int(STAGGER_BUFFER_S * 100), 1000),
        )

        self._started = False

        logger.info(
            "OnlineDatacenter: %d deployments, dt=%s s",
            len(self._deployments),
            dt_s,
        )
        for d in deployments:
            layout = self._layouts.get(d.model_label)
            n_servers = layout.num_servers if layout else 0
            logger.info(
                "  %s: %d real GPUs, %d simulated replicas (%.0fx augmentation), %d virtual servers, vllm=%s",
                d.model_label,
                d.num_real_gpus,
                d.spec.initial_num_replicas,
                d.augmentation_factor,
                n_servers,
                d.vllm_base_url,
            )

    def _build_all_layouts(self) -> None:
        """Build ServerLayout and activation policies for each deployed model.

        The RNG invocation order per model must be: phase shuffle,
        priority shuffle, stagger offsets, amplitude scales. We
        interleave policy construction between the phase shuffle
        and stagger/amplitude draws to preserve this ordering.
        """
        schedule = self._inference_ramp_schedule
        gpus_per_server = self._datacenter_config.gpus_per_server
        rng = self._layout_rng
        amp_lo, amp_hi = self._power_augmentation.amplitude_scale_range
        noise_fraction = self._power_augmentation.noise_fraction
        stagger_s = float(STAGGER_BUFFER_S)

        for d in self._deployments:
            spec = d.spec
            if spec.initial_num_replicas > 0:
                num_servers = math.ceil(spec.initial_num_replicas * spec.gpus_per_replica / gpus_per_server)

                # Phase shuffle (consumes RNG)
                sA, sB, sC = split_integer_evenly(num_servers, 3)
                phase_list = np.asarray(([0] * sA) + ([1] * sB) + ([2] * sC), dtype=int)
                rng.shuffle(phase_list)

                # Priority shuffle (consumes RNG) — must happen here
                self._policies[d.model_label] = RampActivationPolicy(
                    schedule,
                    num_servers,
                    rng,
                )

                # Stagger offsets (consumes RNG) — float for online
                stagger_offsets = rng.uniform(0.0, max(stagger_s, 1e-9), size=num_servers)

                # Amplitude scales (consumes RNG)
                amplitude_scales = rng.uniform(amp_lo, amp_hi, size=num_servers)

                total_gpus = spec.initial_num_replicas * spec.gpus_per_replica
                gpus_per_server_list = np.full(num_servers, gpus_per_server, dtype=int)
                tail = total_gpus - (num_servers - 1) * gpus_per_server
                gpus_per_server_list[-1] = int(tail) if tail > 0 else gpus_per_server

                self._layouts[d.model_label] = ServerLayout(
                    num_servers=num_servers,
                    total_gpus=total_gpus,
                    gpus_per_replica=spec.gpus_per_replica,
                    gpus_per_server_list=gpus_per_server_list,
                    phase_list=phase_list,
                    stagger_offsets=stagger_offsets,
                    amplitude_scales=amplitude_scales,
                    noise_fraction=noise_fraction,
                )

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    @property
    def phase_share_by_model(self) -> dict[str, np.ndarray]:
        """Per-model phase share vectors derived from server layout."""
        shares: dict[str, np.ndarray] = {}
        for label, layout in self._layouts.items():
            counts = np.bincount(layout.phase_list, minlength=3).astype(float)
            total = counts.sum()
            if total > 0:
                shares[label] = counts / total
            else:
                shares[label] = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        return shares

    def reset(self) -> None:
        if self._started:
            self._load_gen.stop()
        self._load_gen = _LoadGenerator(
            self._deployments,
            request_store=self._request_store,
            max_output_tokens=self._live_server_config.max_output_tokens,
            itl_window_s=self._live_server_config.itl_window_s,
            prometheus_poller=self._prometheus,
        )
        self._layout_rng = np.random.default_rng(self._seed)
        self._layouts = {}
        self._policies = {}
        self._build_all_layouts()
        self._inference_augmenter = InferencePowerAugmenter(
            layouts=self._layouts,
            policies=self._policies,
            seed=self._seed + 12345,
        )
        self._rolling_buffer.clear()
        for d in self._deployments:
            d.batch_size = d.spec.initial_batch_size
        self._started = False

    def start(self) -> None:
        """Start load generation, warm up servers, and fill the power buffer.

        Sequence:
            1. Run health checks on all vLLM servers and zeusd instances.
            2. Wait for at least one power reading per endpoint (10 s timeout).
            3. Set initial batch sizes on all vLLM servers.
            4. Start load generation threads.
            5. Warm up: poll power into the rolling buffer while waiting for
               each model's `num_requests_running` to reach 95% of its
               `initial_batch_size`. Fails after 60 s if any model does not
               saturate.
        """
        if self._started:
            raise RuntimeError("OnlineDatacenter already started")

        logger.info("Starting OnlineDatacenter with %d deployments", len(self._deployments))

        # 1. Health checks
        logger.info("Running health checks...")
        for d in self._deployments:
            _check_vllm_health(d.vllm_base_url)
            _check_vllm_model(d.vllm_base_url, d.spec.model_id)
            for ep in d.gpu_endpoints:
                _check_zeusd_health(ep.host, ep.port)
        logger.info("All health checks passed")

        # 2. Wait for power readings from all endpoints
        all_endpoints: set[str] = set()
        for d in self._deployments:
            for ep in d.gpu_endpoints:
                all_endpoints.add(ep.endpoint_key)

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            readings = self._power_client.get_power()
            if all_endpoints.issubset(readings.keys()):
                logger.info("Power readings received from all %d endpoints", len(all_endpoints))
                break
            time.sleep(0.5)
        else:
            connected = set(self._power_client.get_power().keys())
            missing = all_endpoints - connected
            logger.warning("Timed out waiting for power readings from: %s", missing)

        # 3. Set initial batch sizes on vLLM servers
        for d in self._deployments:
            d.set_batch_size(d.spec.initial_batch_size)

        # 4. Start load generation (and Prometheus poller)
        self._load_gen.start()
        logger.info("LoadGenerator started")

        # 5. Warm up: fill power buffer + wait for server saturation
        self._warmup()

        self._started = True
        logger.info("OnlineDatacenter ready")

    def _poll_power_into_buffer(self) -> tuple[float, dict[str, float]]:
        """Read GPU power from all endpoints and feed the rolling buffer.

        Returns:
            Tuple of (monotonic timestamp, per-model average per-GPU watts).
        """
        now = time.monotonic()
        raw_power = self._power_client.get_power()
        per_gpu_by_model: dict[str, float] = {}
        for d in self._deployments:
            total_w = 0.0
            n_gpus = 0
            for ep in d.gpu_endpoints:
                pr = raw_power.get(ep.endpoint_key)
                if pr is None:
                    continue
                for idx in ep.gpu_indices:
                    if idx in pr.gpu_power_w:
                        total_w += pr.gpu_power_w[idx]
                        n_gpus += 1
            per_gpu_w = total_w / n_gpus if n_gpus > 0 else 0.0
            self._rolling_buffer.append(d.model_label, now, per_gpu_w)
            per_gpu_by_model[d.model_label] = per_gpu_w
        return now, per_gpu_by_model

    def _warmup(
        self,
        timeout_s: float = 60.0,
        saturation_threshold: float = 0.95,
        poll_interval_s: float = 0.1,
    ) -> None:
        """Fill the rolling power buffer and wait for vLLM server saturation.

        Actively polls GPU power to fill the rolling buffer while monitoring
        Prometheus `num_requests_running` to verify each model has reached
        `saturation_threshold` of its `initial_batch_size`.

        Completion requires both conditions for every model:
            1. `num_requests_running >= saturation_threshold * initial_batch_size`
            2. At least `stagger_buffer_s` has elapsed since that model first
               reached saturation (so the buffer contains a full stagger
               window of steady-state power data).

        Args:
            timeout_s: Maximum warmup duration in seconds.
            saturation_threshold: Fraction of `initial_batch_size` that
                `num_requests_running` must reach (0.0-1.0).
            poll_interval_s: Seconds between power polls.

        Raises:
            RuntimeError: If any model fails to saturate within `timeout_s`.
                Includes the `num_requests_running` trajectory for failed
                models.
        """
        stagger_s = STAGGER_BUFFER_S
        logger.info(
            "Warming up: waiting for server saturation (%.0f%% of initial_batch_size) "
            "+ %.1f s buffer fill per model...",
            saturation_threshold * 100,
            stagger_s,
        )

        warmup_start = time.monotonic()
        deadline = warmup_start + timeout_s
        last_log = warmup_start

        trajectory: dict[str, list[tuple[float, float]]] = {d.model_label: [] for d in self._deployments}
        saturation_time: dict[str, float | None] = {d.model_label: None for d in self._deployments}

        while time.monotonic() < deadline:
            now = time.monotonic()
            elapsed = now - warmup_start

            self._poll_power_into_buffer()

            all_ready = True
            if self._prometheus is not None:
                prom = self._prometheus.get_latest()
                for d in self._deployments:
                    label = d.model_label
                    running = prom.get(label, {}).get("num_requests_running", 0.0)
                    trajectory[label].append((elapsed, running))
                    target = d.spec.initial_batch_size * saturation_threshold

                    if running >= target and saturation_time[label] is None:
                        saturation_time[label] = now
                        logger.info(
                            "  %s saturated at t=%.1f s (num_requests_running=%.0f)",
                            label,
                            elapsed,
                            running,
                        )

                    sat_t = saturation_time[label]
                    if sat_t is None or (now - sat_t) < stagger_s:
                        all_ready = False
            else:
                logger.warning(
                    "Prometheus polling is disabled; cannot verify server saturation. "
                    "Waiting %.1f s for power buffer only.",
                    stagger_s,
                )
                if elapsed < stagger_s:
                    all_ready = False

            if all_ready:
                logger.info("Warmup complete in %.1f s", elapsed)
                return

            if now - last_log >= 10.0:
                last_log = now
                if self._prometheus is not None:
                    prom = self._prometheus.get_latest()
                    for d in self._deployments:
                        label = d.model_label
                        running = prom.get(label, {}).get("num_requests_running", 0.0)
                        target = d.spec.initial_batch_size
                        sat_t = saturation_time[label]
                        buf_s = (now - sat_t) if sat_t is not None else 0.0
                        logger.info(
                            "  Warmup %s: num_requests_running=%.0f / %d (%.0f%%), buffer=%.1f / %.1f s",
                            label,
                            running,
                            target,
                            running / max(target, 1) * 100,
                            buf_s,
                            stagger_s,
                        )

            time.sleep(poll_interval_s)

        if self._prometheus is None:
            raise RuntimeError(
                f"Warmup timed out after {timeout_s:.0f} s waiting for power buffer to fill ({stagger_s:.1f} s)"
            )

        prom = self._prometheus.get_latest()
        failed: list[str] = []
        for d in self._deployments:
            label = d.model_label
            running = prom.get(label, {}).get("num_requests_running", 0.0)
            sat_t = saturation_time[label]
            not_saturated = running < d.spec.initial_batch_size * saturation_threshold
            not_buffered = sat_t is None or (time.monotonic() - sat_t) < stagger_s
            if not_saturated or not_buffered:
                failed.append(label)

        parts = [
            f"Warmup timed out after {timeout_s:.0f} s. "
            f"Models that failed to reach {saturation_threshold:.0%} of initial_batch_size:",
        ]
        for label in failed:
            target = self._deployment_map[label].spec.initial_batch_size
            traj = trajectory[label]
            final = traj[-1][1] if traj else 0.0
            parts.append(f"  {label} (target: {target}, reached: {final:.0f}):")
            step = max(1, int(5.0 / poll_interval_s))
            samples = traj[::step]
            if traj and (not samples or samples[-1] is not traj[-1]):
                samples.append(traj[-1])
            entries = [f"t={t:.0f}s: {r:.0f}" for t, r in samples]
            parts.append("    " + ", ".join(entries))
        raise RuntimeError("\n".join(parts))

    def stop(self) -> None:
        """Stop load generation and power streaming."""
        self._load_gen.stop()
        self._power_client.stop()
        self._started = False
        logger.info("OnlineDatacenter stopped")

    def step(self, clock: SimulationClock, events: EventEmitter) -> OnlineDatacenterState:
        """Read live power, augment to datacenter scale, and return state."""
        now, per_gpu_w_by_model = self._poll_power_into_buffer()

        measured_power_by_model: dict[str, float] = {}
        augmentation_factor_by_model: dict[str, float] = {}
        for d in self._deployments:
            label = d.model_label
            measured_power_by_model[label] = per_gpu_w_by_model.get(label, 0.0) * d.num_real_gpus
            augmentation_factor_by_model[label] = d.augmentation_factor

        per_gpu_by_model: dict[str, np.ndarray] = {}
        for d in self._deployments:
            label = d.model_label
            if label not in self._layouts:
                continue
            layout = self._layouts[label]
            per_gpu_by_model[label] = self._rolling_buffer.sample_servers(label, now, layout.stagger_offsets)

        inference_aug = self._inference_augmenter.augment(per_gpu_by_model, clock.time_s)

        measured_total = sum(measured_power_by_model.values())
        measured_per_phase = measured_total / 3.0

        observed_itl: dict[str, float] = {
            d.model_label: self._load_gen.get_observed_itl(d.model_label) for d in self._deployments
        }

        prometheus_metrics: dict[str, dict[str, float]] = {}
        if self._prometheus is not None:
            prometheus_metrics = self._prometheus.get_latest()

        state = OnlineDatacenterState(
            time_s=clock.time_s,
            power_w=ThreePhase(
                a=self._base_W_per_phase + inference_aug.power_w.a,
                b=self._base_W_per_phase + inference_aug.power_w.b,
                c=self._base_W_per_phase + inference_aug.power_w.c,
            ),
            batch_size_by_model={d.model_label: d.batch_size for d in self._deployments},
            active_replicas_by_model=inference_aug.active_replicas_by_model,
            observed_itl_s_by_model=observed_itl,
            measured_power_w=ThreePhase(
                a=measured_per_phase + self._base_W_per_phase,
                b=measured_per_phase + self._base_W_per_phase,
                c=measured_per_phase + self._base_W_per_phase,
            ),
            measured_power_w_by_model=measured_power_by_model,
            augmented_power_w_by_model=inference_aug.power_by_model_w,
            augmentation_factor_by_model=augmentation_factor_by_model,
            prometheus_metrics_by_model=prometheus_metrics,
        )
        return state

    @functools.singledispatchmethod
    def apply_control(self, command: DatacenterCommand, events: EventEmitter) -> None:
        """Apply a control command. Dispatches on command type."""
        raise TypeError(f"OnlineDatacenter does not support {type(command).__name__}")

    @apply_control.register
    def apply_control_set_batch_size(self, command: SetBatchSize, events: EventEmitter) -> None:
        """Apply batch size command by sending HTTP requests to vLLM servers."""
        for label, b in command.batch_size_by_model.items():
            label = str(label)
            b_int = int(b)
            if b_int <= 0:
                raise ValueError(f"Batch size must be positive for model {label!r}, got {b_int}.")
            dep = self._deployment_map.get(label)
            if dep is not None:
                dep.set_batch_size(b_int, ramp_up_rate=command.ramp_up_rate_by_model.get(label, 0.0))

        events.emit(
            "datacenter.batch_size.updated",
            {"batch_size_by_model": {d.model_label: d.batch_size for d in self._deployments}},
        )
