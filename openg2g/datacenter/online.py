"""Online (live GPU) datacenter backend with power augmentation.

Connects to real vLLM inference servers for load generation and ITL
measurement, and to zeusd instances for live GPU power monitoring.
Power readings from a small number of real GPUs are augmented to
datacenter scale using temporal staggering.

Requires `pip install zeus httpx`.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import httpx
import numpy as np
from zeus.monitor.power_streaming import PowerReadings, PowerStreamingClient  # type: ignore[import-not-found]

from openg2g.clock import SimulationClock
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter
from openg2g.events import EventEmitter
from openg2g.models.spec import LLMInferenceModelSpec
from openg2g.types import (
    Command,
    OnlineDatacenterState,
    Phase,
    ThreePhase,
)

logger = logging.getLogger(__name__)


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
# Load generator (background thread with asyncio event loop)
# ---------------------------------------------------------------------------


class _LoadGenerator:
    """Background load generator that saturates vLLM servers and measures ITL.

    Runs a daemon thread with an asyncio event loop.  For each model, a
    supervisor coroutine maintains `batch_size * concurrency_multiplier`
    concurrent streaming requests.  Per-token inter-token latency (ITL)
    is measured from SSE chunk arrival times and stored in a rolling
    deque per model.
    """

    def __init__(
        self,
        deployments: Sequence[OnlineModelDeployment],
        prompts_by_model: dict[str, list[str]],
        config: LoadGenerationConfig,
    ) -> None:
        self._deployments = {d.model_label: d for d in deployments}
        self._prompts = dict(prompts_by_model)
        self._config = config

        self._lock = threading.Lock()
        self._batch_by_model: dict[str, int] = {}
        self._itl_samples: dict[str, collections.deque[tuple[float, float]]] = {}
        for d in deployments:
            label = d.model_label
            self._batch_by_model[label] = d.spec.initial_batch_size
            self._itl_samples[label] = collections.deque(maxlen=10_000)

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
        tasks = [asyncio.create_task(self._model_supervisor(label, dep)) for label, dep in self._deployments.items()]
        while not self._stop_event.is_set():
            await asyncio.sleep(0.1)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _model_supervisor(self, label: str, dep: OnlineModelDeployment) -> None:
        """Maintain target concurrency of streaming requests for one model."""
        active: set[asyncio.Task[None]] = set()
        prompt_idx = 0
        prompts = self._prompts.get(label, ["Hello, how are you?"])

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            while not self._stop_event.is_set():
                with self._lock:
                    batch = self._batch_by_model.get(label, 1)
                target = max(1, int(batch * self._config.concurrency_multiplier))

                while len(active) < target and not self._stop_event.is_set():
                    prompt = prompts[prompt_idx % len(prompts)]
                    prompt_idx += 1
                    task = asyncio.create_task(self._single_request(label, dep, prompt, client))
                    active.add(task)
                    task.add_done_callback(active.discard)

                if active:
                    await asyncio.wait(active, timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
                else:
                    await asyncio.sleep(0.1)

    async def _single_request(
        self,
        label: str,
        dep: OnlineModelDeployment,
        prompt: str,
        client: httpx.AsyncClient,
    ) -> None:
        """Send one streaming chat-completion request and measure ITL."""
        url = f"{dep.vllm_base_url}/v1/chat/completions"
        body = {
            "model": dep.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._config.max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }
        token_times: list[float] = []
        try:
            async with client.stream("POST", url, json=body) as response:
                response.raise_for_status()
                buffer = ""
                async for chunk in response.aiter_text():
                    if self._stop_event.is_set():
                        return
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices", [])
                        if choices and choices[0].get("delta", {}).get("content"):
                            token_times.append(time.monotonic())
        except Exception:
            if not self._stop_event.is_set():
                logger.debug("Request to %s failed for %s", dep.vllm_base_url, label, exc_info=True)
            return

        if len(token_times) >= 3:
            itls = [token_times[i + 1] - token_times[i] for i in range(len(token_times) - 1)]
            avg_itl = sum(itls) / len(itls)
            now = time.monotonic()
            with self._lock:
                self._itl_samples[label].append((now, avg_itl))


# ---------------------------------------------------------------------------
# Power augmenter (temporal staggering + scaling)
# ---------------------------------------------------------------------------


class _PowerAugmenter:
    """Augments real GPU power to datacenter scale with temporal staggering.

    Maintains a rolling buffer of per-model measured power.  At each
    query, K virtual server groups — each with a random time offset —
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
        prompts_by_model: Mapping of model_label -> list of prompt strings
            used to generate saturated load.
        dt_s: Simulation timestep (seconds).
    """

    def __init__(
        self,
        *,
        deployments: Sequence[OnlineModelDeployment],
        power_client: PowerStreamingClient,
        augmentation: PowerAugmentationConfig | None = None,
        load_gen: LoadGenerationConfig | None = None,
        prompts_by_model: dict[str, list[str]],
        dt_s: Fraction = Fraction(1, 10),
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

        self._load_gen = _LoadGenerator(deployments, prompts_by_model, load_gen)
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
    def state(self) -> OnlineDatacenterState | None:
        return self._state

    def history(self, n: int | None = None) -> list[OnlineDatacenterState]:
        if n is None:
            return list(self._history)
        if n <= 0:
            return []
        return list(self._history[-int(n) :])

    def start(self) -> None:
        """Start load generation and wait for initial readings.

        The `PowerStreamingClient` is already connected (it starts on
        construction). This method waits for readings, then starts load
        generation.

        Sequence:
            1. Wait for at least one power reading per endpoint (10 s timeout).
            2. Set initial batch sizes on all vLLM servers.
            3. Start load generation threads.
            4. Wait for at least one ITL sample per model (30 s timeout).
        """
        if self._started:
            raise RuntimeError("OnlineDatacenter already started")

        logger.info("Starting OnlineDatacenter with %d deployments", len(self._deployments))

        # 1. Wait for power readings from all endpoints
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

        # 4. Start load generation
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

            # Distribute power across phases proportional to real GPU counts
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
        )
        self._state = state
        self._history.append(state)
        return state

    def apply_control(self, command: Command) -> None:
        """Apply batch size command by sending HTTP requests to vLLM servers."""
        if command.kind != "set_batch_size":
            raise ValueError(f"OnlineDatacenter does not support command kind={command.kind!r}")
        if "batch_size_by_model" not in command.payload:
            raise ValueError("set_batch_size requires payload['batch_size_by_model'].")
        batch_map = command.payload["batch_size_by_model"]
        if not isinstance(batch_map, dict):
            raise ValueError("set_batch_size requires payload['batch_size_by_model'] as a dict.")

        for label, b in batch_map.items():
            label = str(label)
            b_int = int(b)
            if b_int <= 0:
                raise ValueError(f"Batch size must be positive for model {label!r}, got {b_int}.")
            old = self._batch_by_model.get(label)
            self._batch_by_model[label] = b_int
            if old != b_int:
                dep = self._deployment_map.get(label)
                if dep is not None:
                    _set_vllm_batch_size(dep.vllm_base_url, b_int)
                    self._load_gen.set_batch_size(label, b_int)
                logger.info("Batch size %s: %s -> %d", label, old, b_int)

        if self._events is not None:
            self._events.emit(
                "datacenter.batch_size.updated",
                {
                    "kind": command.kind,
                    "batch_size_by_model": dict(self._batch_by_model),
                },
            )

    def bind_event_emitter(self, emitter: EventEmitter) -> None:
        self._events = emitter


def _set_vllm_batch_size(vllm_base_url: str, batch_size: int) -> None:
    """Send HTTP POST to set batch size on the vLLM server."""
    url = f"{vllm_base_url}/set_max_num_seqs"
    try:
        resp = httpx.post(url, params={"max_num_seqs": batch_size}, timeout=2.0)
        resp.raise_for_status()
    except Exception:
        logger.warning(
            "Failed to set batch size %d on %s",
            batch_size,
            vllm_base_url,
            exc_info=True,
        )
