"""Online (live GPU) datacenter backend with power augmentation.

Connects to real vLLM inference servers for load generation and ITL
measurement, and to zeusd instances for live GPU power monitoring.
Power readings from a small number of real GPUs are augmented to
datacenter scale using temporal staggering.

Requires `pip install zeus aiohttp`.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import functools
import json
import logging
import re
import threading
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import aiohttp
import numpy as np
from zeus.monitor.power_streaming import PowerReadings, PowerStreamingClient

from openg2g.clock import SimulationClock
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter, LLMDatacenterState
from openg2g.events import EventEmitter
from openg2g.grid.base import Phase
from openg2g.models.spec import LLMInferenceModelSpec
from openg2g.types import (
    DatacenterCommand,
    SetBatchSize,
    ThreePhase,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnlineDatacenterState(LLMDatacenterState):
    """Extended state from the online (live GPU) backend.

    The base `power_w` field carries the augmented three-phase power
    (what the grid sees). This subclass adds the measured (pre-augmentation)
    breakdown for post-hoc analysis.

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


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPUEndpointMapping:
    """Maps a zeusd endpoint to specific GPUs on a specific electrical phase.

    Args:
        host: Hostname or IP of the zeusd instance.
        port: TCP port of the zeusd instance.
        gpu_indices: GPU device indices to monitor on this endpoint.
        phase: Electrical phase this endpoint's GPUs are connected to.
    """

    host: str
    port: int = 4938
    gpu_indices: tuple[int, ...] = (0,)
    phase: Phase = Phase.A

    @property
    def endpoint_key(self) -> str:
        """Return the `host:port` key used by `PowerStreamingClient`."""
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class OnlineModelDeployment:
    """Deployment of one model on physical hardware.

    Pairs a reusable `LLMInferenceModelSpec` with physical deployment
    details.  `spec.num_replicas` is the simulated (augmented) count
    for grid simulation.  The real replica count is derived from
    `gpu_endpoints` and `spec.gpus_per_replica`.

    Args:
        spec: Model specification (shared with offline datacenter).
        vllm_base_url: Base URL of the vLLM server (e.g. `http://node1:8000`).
        model_name: OpenAI API model name served by vLLM.
        gpu_endpoints: GPU endpoint mappings for power monitoring.
    """

    spec: LLMInferenceModelSpec
    vllm_base_url: str
    model_name: str
    gpu_endpoints: tuple[GPUEndpointMapping, ...] = ()

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
        return self.spec.num_replicas / max(self.num_real_replicas, 1)


@dataclass(frozen=True)
class LoadGenerationConfig:
    """Configuration for the request load generator.

    Args:
        max_output_tokens: Maximum output tokens per request.
        concurrency_multiplier: Number of concurrent requests per unit
            of batch size (`N_concurrent = batch_size * this`).
        itl_window_s: Seconds of ITL history to average over.
    """

    max_output_tokens: int = 512
    concurrency_multiplier: float = 3.0
    itl_window_s: float = 1.0


@dataclass(frozen=True)
class PowerAugmentationConfig:
    """Configuration for scaling real GPU power to datacenter level.

    Args:
        base_kw_per_phase: Constant base load per electrical phase (kW).
        noise_frac: Noise standard deviation as a fraction of augmented power.
        stagger_buffer_s: Seconds of power history for temporal staggering.
        num_virtual_groups: Number of virtual server groups for staggering.
            Each group reads from the power buffer at a different time offset,
            smoothing batch-size-change transients.
        seed: Random seed for staggering offsets and noise.
    """

    base_kw_per_phase: float = 0.0
    noise_frac: float = 0.02
    stagger_buffer_s: float = 10.0
    num_virtual_groups: int = 64
    seed: int = 0


# ---------------------------------------------------------------------------
# Health checks (sync, using stdlib urllib)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Prometheus metric extraction
# ---------------------------------------------------------------------------

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
    # Build a quick lookup of metric name -> accumulated value
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


# ---------------------------------------------------------------------------
# Prometheus poller (background async task)
# ---------------------------------------------------------------------------


class _PrometheusPoller:
    """Polls vLLM /metrics endpoints for Prometheus gauges.

    Runs as an async task inside `_LoadGenerator`'s event loop.
    Provides thread-safe access to the latest snapshot per model.
    """

    def __init__(
        self,
        deployments: Sequence[OnlineModelDeployment],
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


# ---------------------------------------------------------------------------
# Load generator (background thread with asyncio event loop)
# ---------------------------------------------------------------------------


class _LoadGenerator:
    """Background load generator that saturates vLLM servers and measures ITL.

    Runs a daemon thread with an asyncio event loop.  For each model, a
    supervisor coroutine maintains `batch_size * concurrency_multiplier`
    concurrent streaming requests.  Per-token inter-token latency (ITL)
    is measured from SSE chunk arrival times using `usage.completion_tokens`
    increments to correctly handle multi-token bundles.
    """

    def __init__(
        self,
        deployments: Sequence[OnlineModelDeployment],
        requests_by_model: dict[str, list[dict]],
        config: LoadGenerationConfig,
        prometheus_poller: _PrometheusPoller | None = None,
    ) -> None:
        self._deployments = {d.model_label: d for d in deployments}
        self._requests = dict(requests_by_model)
        self._config = config
        self._prometheus = prometheus_poller

        self._lock = threading.Lock()
        self._batch_by_model: dict[str, int] = {}
        self._itl_samples: dict[str, collections.deque[tuple[float, float]]] = {}
        for d in deployments:
            label = d.model_label
            self._batch_by_model[label] = d.spec.initial_batch_size
            self._itl_samples[label] = collections.deque()

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

    def set_batch_size(self, model_label: str, batch_size: int) -> None:
        with self._lock:
            self._batch_by_model[model_label] = batch_size

    def get_observed_itl(self, model_label: str, window_s: float | None = None) -> float:
        """Return the windowed-average ITL for *model_label*, or NaN."""
        if window_s is None:
            window_s = self._config.itl_window_s
        cutoff = time.monotonic() - window_s
        with self._lock:
            samples = self._itl_samples.get(model_label)
            if not samples:
                return float("nan")
            recent = [itl for ts, itl in samples if ts >= cutoff]
        if not recent:
            return float("nan")
        return sum(recent) / len(recent)

    # -- background thread ---------------------------------------------------

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

        # Start per-model supervisors
        for label, dep in self._deployments.items():
            tasks.append(asyncio.create_task(self._model_supervisor(label, dep)))

        # Start Prometheus poller if configured
        if self._prometheus is not None:
            tasks.append(asyncio.create_task(self._prometheus.run(self._stop_event)))

        while not self._stop_event.is_set():
            await asyncio.sleep(0.1)

        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _model_supervisor(self, label: str, dep: OnlineModelDeployment) -> None:
        """Maintain target concurrency of streaming requests for one model."""
        active: set[asyncio.Task[None]] = set()
        req_idx = 0
        requests = self._requests.get(label, [])

        connector = aiohttp.TCPConnector(limit=0, ssl=False)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300.0),
            connector=connector,
        ) as session:
            while not self._stop_event.is_set():
                with self._lock:
                    batch = self._batch_by_model.get(label, 1)
                target = max(1, int(batch * self._config.concurrency_multiplier))

                while len(active) < target and not self._stop_event.is_set():
                    if requests:
                        request_dict = requests[req_idx % len(requests)]
                        req_idx += 1
                    else:
                        request_dict = self._default_request(dep)
                    task = asyncio.create_task(self._single_request(label, dep, request_dict, session))
                    active.add(task)
                    task.add_done_callback(active.discard)

                if active:
                    await asyncio.wait(active, timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
                else:
                    await asyncio.sleep(0.1)

    def _default_request(self, dep: OnlineModelDeployment) -> dict:
        """Build a minimal fallback request dict."""
        return {
            "model": dep.model_name,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_completion_tokens": self._config.max_output_tokens,
        }

    async def _single_request(
        self,
        label: str,
        dep: OnlineModelDeployment,
        request_dict: dict,
        session: aiohttp.ClientSession,
    ) -> None:
        """Send one streaming chat-completion request and measure per-token ITL.

        Uses `usage.completion_tokens` increments to correctly handle
        multi-token bundles. Stores individual per-token ITL samples.
        """
        url = f"{dep.vllm_base_url}/v1/chat/completions"
        body = dict(request_dict)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
        if "max_tokens" in body and "max_completion_tokens" not in body:
            body["max_completion_tokens"] = body.pop("max_tokens")

        current_completion_tokens = 0
        most_recent_timestamp = time.perf_counter()
        ttft_recorded = False

        try:
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

                    # Skip SSE comments (often used as pings)
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
                    now_mono = time.monotonic()

                    if not ttft_recorded:
                        # First token(s)
                        ttft_recorded = True
                        inc = completion_tokens - current_completion_tokens
                        current_completion_tokens = completion_tokens
                        # Record zero ITL for bundled first tokens
                        with self._lock:
                            for _ in range(inc):
                                self._itl_samples[label].append((now_mono, 0.0))
                    else:
                        # Decoding phase
                        itl = timestamp - most_recent_timestamp
                        inc = completion_tokens - current_completion_tokens
                        current_completion_tokens = completion_tokens

                        with self._lock:
                            self._itl_samples[label].append((now_mono, itl))
                            # Bundled tokens get zero ITL
                            for _ in range(max(inc - 1, 0)):
                                self._itl_samples[label].append((now_mono, 0.0))

                    most_recent_timestamp = timestamp

        except Exception:
            if not self._stop_event.is_set():
                logger.debug("Request to %s failed for %s", dep.vllm_base_url, label, exc_info=True)


# ---------------------------------------------------------------------------
# Power augmenter (temporal staggering + scaling)
# ---------------------------------------------------------------------------


class _PowerAugmenter:
    """Augments real GPU power to datacenter scale with temporal staggering.

    Maintains a rolling buffer of per-model measured power.  At each
    query, K virtual server groups -- each with a random time offset --
    read from the buffer at different historical points.  Their average
    yields a smoothed per-GPU power that is then scaled to the full
    simulated GPU count.  This spreads batch-size-change transients over
    `stagger_buffer_s` seconds instead of reflecting them instantaneously.
    """

    def __init__(
        self,
        deployments: Sequence[OnlineModelDeployment],
        config: PowerAugmentationConfig,
    ) -> None:
        self._deployments = {d.model_label: d for d in deployments}
        self._config = config
        self._rng = np.random.default_rng(config.seed)

        self._power_buffer: dict[str, collections.deque[tuple[float, float]]] = {}
        self._group_offsets: dict[str, np.ndarray] = {}

        for d in deployments:
            label = d.model_label
            max_samples = max(int(config.stagger_buffer_s * 100), 1000)
            self._power_buffer[label] = collections.deque(maxlen=max_samples)
            n_groups = min(config.num_virtual_groups, max(d.spec.num_replicas, 1))
            self._group_offsets[label] = self._rng.uniform(
                0.0,
                config.stagger_buffer_s,
                size=n_groups,
            )

    def update(
        self,
        readings_by_endpoint: dict[str, PowerReadings],
        deployments: dict[str, OnlineModelDeployment],
    ) -> None:
        """Ingest new power readings from `PowerStreamingClient`.

        Args:
            readings_by_endpoint: Mapping of `host:port` -> `PowerReadings`.
            deployments: Model label -> deployment mapping.
        """
        now = time.monotonic()
        for label, dep in deployments.items():
            total_w = 0.0
            n_gpus = 0
            for ep in dep.gpu_endpoints:
                pr = readings_by_endpoint.get(ep.endpoint_key)
                if pr is None:
                    continue
                for idx in ep.gpu_indices:
                    if idx in pr.gpu_power_w:
                        total_w += pr.gpu_power_w[idx]
                        n_gpus += 1
            per_gpu_w = total_w / n_gpus if n_gpus > 0 else 0.0
            self._power_buffer[label].append((now, per_gpu_w))

    def augmented_power(self, model_label: str) -> tuple[float, float]:
        """Compute augmented total power and measured per-GPU power.

        Returns:
            Tuple of (augmented_total_watts, measured_per_gpu_watts).
        """
        dep = self._deployments[model_label]
        buf = self._power_buffer[model_label]
        if not buf:
            return 0.0, 0.0

        now = time.monotonic()
        offsets = self._group_offsets[model_label]

        group_powers = np.empty(len(offsets))
        for i, offset in enumerate(offsets):
            group_powers[i] = _lookup_power(buf, now - offset)

        smoothed_per_gpu = float(np.mean(group_powers))

        simulated_gpus = dep.spec.num_replicas * dep.spec.gpus_per_replica
        augmented = smoothed_per_gpu * simulated_gpus

        if self._config.noise_frac > 0 and simulated_gpus > 0:
            sigma = augmented * self._config.noise_frac / np.sqrt(simulated_gpus)
            augmented += float(self._rng.normal(0.0, sigma))
            augmented = max(augmented, 0.0)

        measured_per_gpu = buf[-1][1]
        return augmented, measured_per_gpu


def _lookup_power(
    buf: collections.deque[tuple[float, float]],
    target_t: float,
) -> float:
    """Look up the power reading closest to *target_t* from the buffer."""
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


# ---------------------------------------------------------------------------
# OnlineDatacenter
# ---------------------------------------------------------------------------


class OnlineDatacenter(LLMBatchSizeControlledDatacenter[OnlineDatacenterState]):
    """Live GPU datacenter backend with power augmentation.

    Dispatches inference load to vLLM servers, streams GPU power from
    zeusd, measures ITL from streaming responses, and augments power
    readings to datacenter scale for grid simulation.

    Call `start()` before the first `step()` and `stop()` after the
    simulation loop finishes.

    Args:
        deployments: List of model deployments with physical hardware mapping.
        power_client: Zeus `PowerStreamingClient` connected to zeusd instances.
        augmentation: Power augmentation configuration.
        load_gen: Load generation configuration.
        requests_by_model: Mapping of model_label -> list of pre-built
            OpenAI Chat Completion request dicts (from `data/online/build_requests.py`).
            Each dict should contain at least `model`, `messages`, and
            `max_completion_tokens`. Streaming fields are added automatically.
        dt_s: Simulation timestep (seconds).
        prometheus_poll_interval_s: How often to poll vLLM /metrics (seconds).
            Set to 0 to disable Prometheus polling.
        health_check: If True, run health checks on start().
    """

    def __init__(
        self,
        *,
        deployments: Sequence[OnlineModelDeployment],
        power_client: PowerStreamingClient,
        augmentation: PowerAugmentationConfig | None = None,
        load_gen: LoadGenerationConfig | None = None,
        requests_by_model: dict[str, list[dict]],
        dt_s: Fraction = Fraction(1, 10),
        prometheus_poll_interval_s: float = 0.5,
        health_check: bool = True,
    ) -> None:
        if augmentation is None:
            augmentation = PowerAugmentationConfig()
        if load_gen is None:
            load_gen = LoadGenerationConfig()
        self._dt = dt_s
        self._deployments = list(deployments)
        self._deployment_map = {d.model_label: d for d in deployments}
        self._power_client = power_client
        self._augmentation_config = augmentation
        self._load_gen_config = load_gen
        self._requests_by_model = dict(requests_by_model)
        self._health_check = health_check

        self._prometheus = (
            _PrometheusPoller(
                deployments,
                poll_interval_s=prometheus_poll_interval_s,
            )
            if prometheus_poll_interval_s > 0
            else None
        )

        self._load_gen = _LoadGenerator(
            deployments,
            requests_by_model,
            load_gen,
            prometheus_poller=self._prometheus,
        )
        self._augmenter = _PowerAugmenter(deployments, augmentation)

        self._batch_by_model: dict[str, int] = {d.model_label: d.spec.initial_batch_size for d in deployments}

        self._state: OnlineDatacenterState | None = None
        self._history: list[OnlineDatacenterState] = []
        self._events: EventEmitter | None = None
        self._started = False

        logger.info(
            "OnlineDatacenter: %d deployments, dt=%s s",
            len(self._deployments),
            dt_s,
        )
        for d in deployments:
            logger.info(
                "  %s: %d real GPUs, %d simulated replicas (%.0fx augmentation), vllm=%s",
                d.model_label,
                d.num_real_gpus,
                d.spec.num_replicas,
                d.augmentation_factor,
                d.vllm_base_url,
            )

    @property
    def dt_s(self) -> Fraction:
        return self._dt

    @property
    def state(self) -> OnlineDatacenterState:
        if self._state is None:
            raise RuntimeError("OnlineDatacenter.state accessed before first step().")
        return self._state

    @property
    def phase_share_by_model(self) -> dict[str, np.ndarray]:
        """Per-model phase share vectors derived from GPU endpoint placement."""
        _PHASE_INDEX = {Phase.A: 0, Phase.B: 1, Phase.C: 2}
        shares: dict[str, np.ndarray] = {}
        for d in self._deployments:
            counts = np.zeros(3, dtype=float)
            for ep in d.gpu_endpoints:
                counts[_PHASE_INDEX[ep.phase]] += len(ep.gpu_indices)
            total = counts.sum()
            if total > 0:
                shares[d.model_label] = counts / total
            else:
                shares[d.model_label] = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        return shares

    def history(self, n: int | None = None) -> list[OnlineDatacenterState]:
        if n is None:
            return list(self._history)
        if n <= 0:
            return []
        return list(self._history[-int(n) :])

    def reset(self) -> None:
        if self._started:
            self._load_gen.stop()
        self._load_gen = _LoadGenerator(
            self._deployments,
            self._requests_by_model,
            self._load_gen_config,
            prometheus_poller=self._prometheus,
        )
        self._augmenter = _PowerAugmenter(self._deployments, self._augmentation_config)
        self._batch_by_model = {d.model_label: d.spec.initial_batch_size for d in self._deployments}
        self._state = None
        self._history = []
        self._started = False

    def start(self) -> None:
        """Start load generation and wait for initial readings.

        Sequence:
            1. Run health checks on all vLLM servers and zeusd instances.
            2. Wait for at least one power reading per endpoint (10 s timeout).
            3. Set initial batch sizes on all vLLM servers.
            4. Start load generation threads.
            5. Wait for at least one ITL sample per model (30 s timeout).
        """
        if self._started:
            raise RuntimeError("OnlineDatacenter already started")

        logger.info("Starting OnlineDatacenter with %d deployments", len(self._deployments))

        # 1. Health checks
        if self._health_check:
            logger.info("Running health checks...")
            for d in self._deployments:
                _check_vllm_health(d.vllm_base_url)
                _check_vllm_model(d.vllm_base_url, d.model_name)
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
            batch = d.spec.initial_batch_size
            _set_vllm_batch_size(d.vllm_base_url, batch)
            logger.info("Set initial batch size for %s: %d", d.model_label, batch)

        # 4. Start load generation (and Prometheus poller)
        self._load_gen.start()
        logger.info("LoadGenerator started")

        # 5. Wait for initial ITL samples
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            all_have_itl = all(
                not np.isnan(self._load_gen.get_observed_itl(d.model_label, window_s=30.0)) for d in self._deployments
            )
            if all_have_itl:
                logger.info("ITL samples received from all models")
                break
            time.sleep(1.0)
        else:
            logger.warning("Timed out waiting for ITL samples from some models")

        self._started = True
        logger.info("OnlineDatacenter ready")

    def stop(self) -> None:
        """Stop load generation and power streaming."""
        self._load_gen.stop()
        self._power_client.stop()
        self._started = False
        logger.info("OnlineDatacenter stopped")

    def step(self, clock: SimulationClock) -> OnlineDatacenterState:
        """Read live power, augment to datacenter scale, and return state."""
        raw_power = self._power_client.get_power()
        self._augmenter.update(raw_power, self._deployment_map)

        augmented_phase_w: dict[Phase, float] = {Phase.A: 0.0, Phase.B: 0.0, Phase.C: 0.0}
        measured_phase_w: dict[Phase, float] = {Phase.A: 0.0, Phase.B: 0.0, Phase.C: 0.0}
        measured_power_by_model: dict[str, float] = {}
        augmented_power_by_model: dict[str, float] = {}
        augmentation_factor_by_model: dict[str, float] = {}
        active_replicas: dict[str, int] = {}

        for d in self._deployments:
            label = d.model_label
            augmented_w, measured_per_gpu_w = self._augmenter.augmented_power(label)

            measured_total_w = measured_per_gpu_w * d.num_real_gpus
            measured_power_by_model[label] = measured_total_w
            augmented_power_by_model[label] = augmented_w
            augmentation_factor_by_model[label] = d.augmentation_factor
            active_replicas[label] = d.spec.num_replicas

            phase_gpu_counts: dict[Phase, int] = {Phase.A: 0, Phase.B: 0, Phase.C: 0}
            for ep in d.gpu_endpoints:
                phase_gpu_counts[ep.phase] += len(ep.gpu_indices)
            total_real_gpus = sum(phase_gpu_counts.values())
            if total_real_gpus > 0:
                for phase, count in phase_gpu_counts.items():
                    frac = count / total_real_gpus
                    augmented_phase_w[phase] += augmented_w * frac
                    measured_phase_w[phase] += measured_total_w * frac

        base_w = self._augmentation_config.base_kw_per_phase * 1e3
        for phase in Phase:
            augmented_phase_w[phase] += base_w
            measured_phase_w[phase] += base_w

        observed_itl: dict[str, float] = {
            d.model_label: self._load_gen.get_observed_itl(d.model_label) for d in self._deployments
        }

        prometheus_metrics: dict[str, dict[str, float]] = {}
        if self._prometheus is not None:
            prometheus_metrics = self._prometheus.get_latest()

        state = OnlineDatacenterState(
            time_s=clock.time_s,
            power_w=ThreePhase(
                a=augmented_phase_w[Phase.A],
                b=augmented_phase_w[Phase.B],
                c=augmented_phase_w[Phase.C],
            ),
            batch_size_by_model=dict(self._batch_by_model),
            active_replicas_by_model=active_replicas,
            observed_itl_s_by_model=observed_itl,
            measured_power_w=ThreePhase(
                a=measured_phase_w[Phase.A],
                b=measured_phase_w[Phase.B],
                c=measured_phase_w[Phase.C],
            ),
            measured_power_w_by_model=measured_power_by_model,
            augmented_power_w_by_model=augmented_power_by_model,
            augmentation_factor_by_model=augmentation_factor_by_model,
            prometheus_metrics_by_model=prometheus_metrics,
        )
        self._state = state
        self._history.append(state)
        return state

    @functools.singledispatchmethod
    def apply_control(self, command: DatacenterCommand) -> None:
        """Apply a control command. Dispatches on command type."""
        raise TypeError(f"OnlineDatacenter does not support {type(command).__name__}")

    @apply_control.register
    def _(self, command: SetBatchSize) -> None:
        """Apply batch size command by sending HTTP requests to vLLM servers."""
        for label, b in command.batch_size_by_model.items():
            label = str(label)
            b_int = int(b)
            if b_int <= 0:
                raise ValueError(f"Batch size must be positive for model {label!r}, got {b_int}.")
            old = self._batch_by_model.get(label)
            self._batch_by_model[label] = b_int
            if old != b_int:
                dep = self._deployment_map.get(label)
                if dep is not None:
                    _set_vllm_batch_size(dep.vllm_base_url, b_int, ramp_up_rate=command.ramp_up_rate)
                    self._load_gen.set_batch_size(label, b_int)
                logger.info("Batch size %s: %s -> %d", label, old, b_int)

        if self._events is not None:
            self._events.emit(
                "datacenter.batch_size.updated",
                {"batch_size_by_model": dict(self._batch_by_model)},
            )

    def bind_event_emitter(self, emitter: EventEmitter) -> None:
        self._events = emitter


def _set_vllm_batch_size(
    vllm_base_url: str,
    batch_size: int,
    ramp_up_rate: float = 0.0,
) -> None:
    """Send HTTP POST to set batch size on the vLLM server."""
    url = f"{vllm_base_url}/set_max_num_seqs?max_num_seqs={batch_size}"
    if ramp_up_rate > 0:
        url += f"&ramp_up_rate={ramp_up_rate}"
    try:
        req = urllib.request.Request(url, method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status >= 400:
                logger.warning("Failed to set batch size %d on %s: HTTP %d", batch_size, vllm_base_url, resp.status)
    except Exception:
        logger.warning(
            "Failed to set batch size %d on %s",
            batch_size,
            vllm_base_url,
            exc_info=True,
        )
