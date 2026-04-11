"""Offline (trace-based) datacenter backend."""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter, LLMDatacenterState
from openg2g.datacenter.command import DatacenterCommand, SetBatchSize, ShiftReplicas
from openg2g.datacenter.config import (
    DatacenterConfig,
    PowerAugmentationConfig,
    ReplicaSchedule,
    TrainingSchedule,
)
from openg2g.datacenter.layout import ServerPool
from openg2g.datacenter.workloads.inference import InferenceData, InferencePowerAugmenter
from openg2g.events import EventEmitter
from openg2g.utils import split_integer_evenly

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OfflineDatacenterState(LLMDatacenterState):
    """Extended state from the offline (trace-based) backend.

    Adds per-model power breakdown to
    [`LLMDatacenterState`][openg2g.datacenter.base.LLMDatacenterState].
    """

    power_by_model_w: dict[str, float] = field(default_factory=dict)


@dataclass
class OfflineWorkload:
    """Complete offline simulation workload.

    Bundles inference data with per-model replica schedules, batch sizes,
    and optional training overlays.

    Attributes:
        inference_data: LLM inference workload with offline simulation
            data (model specs, power templates, ITL fits).
        replica_schedules: Per-model replica count schedules. Each key is
            a model label, each value is a `ReplicaSchedule` specifying
            initial count and optional ramps.
        initial_batch_sizes: Mapping of model label to initial batch size.
        training: Training workload schedule. An empty schedule disables
            training overlay.
    """

    inference_data: InferenceData
    replica_schedules: dict[str, ReplicaSchedule] = field(default_factory=dict)
    initial_batch_sizes: dict[str, int] = field(default_factory=dict)
    training: TrainingSchedule = field(default_factory=TrainingSchedule)


class OfflineDatacenter(LLMBatchSizeControlledDatacenter[OfflineDatacenterState]):
    """Trace-based datacenter simulation with step-by-step interface.

    Each `step` call computes one timestep of power output by indexing
    into pre-built per-GPU templates, applying per-server amplitude
    scaling and noise, and summing across active servers per phase.

    Batch size changes via `apply_control` take effect on the next
    `step` call.

    Each model with active replicas gets a
    [`ServerPool`][openg2g.datacenter.layout.ServerPool] with a random
    priority ordering that determines server activation.

    Args:
        datacenter: Facility configuration (GPUs per server, base load).
        workload: Offline workload configuration bundling inference data,
            training overlays, and server ramp events.
        dt_s: Simulation timestep (seconds).
        seed: Random seed for layout generation, noise, and latency
            sampling. Sub-seeds are derived deterministically.
        power_augmentation: Per-server amplitude scaling and noise
            settings.
    """

    def __init__(
        self,
        datacenter: DatacenterConfig,
        workload: OfflineWorkload,
        *,
        name: str,
        dt_s: Fraction,
        seed: int = 0,
        power_augmentation: PowerAugmentationConfig | None = None,
        load_shift_headroom: float = 0.0,
        total_gpu_capacity: int,
    ) -> None:
        super().__init__(name=name)
        if power_augmentation is None:
            power_augmentation = PowerAugmentationConfig()

        self._datacenter = datacenter
        self._workload = workload
        self._power_augmentation = power_augmentation
        self._load_shift_headroom = load_shift_headroom
        self._total_gpu_capacity = total_gpu_capacity
        self._replica_counts = {label: sched.initial for label, sched in workload.replica_schedules.items()}
        self._dt_s = dt_s
        self._seed = int(seed)
        self._models = list(workload.inference_data.models)
        self._base_W_per_phase = float(datacenter.base_kw_per_phase) * 1e3

        # Validate: initial GPU usage must not exceed capacity
        initial_usage = sum(self._replica_counts.get(ms.model_label, 0) * ms.gpus_per_replica for ms in self._models)
        if initial_usage > total_gpu_capacity:
            raise ValueError(f"Initial GPU usage ({initial_usage}) exceeds total_gpu_capacity ({total_gpu_capacity}).")

        # Validate ramp schedule against GPU capacity
        self._validate_ramp_capacity()

        self._layout_rng = np.random.default_rng(self._seed)
        self._batch_by_model: dict[str, int] = {
            ms.model_label: self._workload.initial_batch_sizes.get(ms.model_label, ms.feasible_batch_sizes[0])
            for ms in self._models
        }

        self._model_schedules: dict[str, ReplicaSchedule] = {
            ms.model_label: self._workload.replica_schedules[ms.model_label]
            for ms in self._models
            if ms.model_label in self._workload.replica_schedules
        }
        self._pool: ServerPool = self._build_server_pool()
        self._inference_augmenter = InferencePowerAugmenter(
            pool=self._pool,
            gpus_per_replica_by_model={ms.model_label: ms.gpus_per_replica for ms in self._models},
            seed=self._seed + 12345,
        )

        self._global_step: int = 0
        self._latency_rng = np.random.default_rng(self._seed + 54321)
        self._replica_offset_by_model: dict[str, int] = {ms.model_label: 0 for ms in self._models}

        logger.info(
            "OfflineDatacenter: %d models, dt=%s s, seed=%d, gpu_capacity=%d",
            len(self._models),
            dt_s,
            seed,
            total_gpu_capacity,
        )
        for ms in self._models:
            n_rep = self._replica_counts.get(ms.model_label, 0)
            logger.info(
                "  %s: %d replicas, %d GPUs/replica, batch=%d",
                ms.model_label,
                n_rep,
                ms.gpus_per_replica,
                self._batch_by_model.get(ms.model_label, 0),
            )

    def _validate_ramp_capacity(self) -> None:
        """Check that the ramp schedule never exceeds total GPU capacity."""
        schedules = self._workload.replica_schedules
        # Collect all ramp boundary times
        boundary_times: set[float] = set()
        for sched in schedules.values():
            for _target, t_start, t_end in sched._ramps:
                boundary_times.add(t_start)
                boundary_times.add(t_end)
        if not boundary_times:
            return
        # At each boundary, compute total GPU usage
        for t in sorted(boundary_times):
            total_gpus = 0
            for ms in self._models:
                label = ms.model_label
                sched = schedules.get(label)
                if sched is None:
                    continue
                count = sched.count_at(t)
                total_gpus += int(round(float(count))) * ms.gpus_per_replica
            if total_gpus > self._total_gpu_capacity:
                raise ValueError(
                    f"Ramp schedule exceeds total_gpu_capacity at t={t:.1f}s: "
                    f"needs {total_gpus} GPUs but capacity is {self._total_gpu_capacity}."
                )

    @property
    def total_gpu_capacity(self) -> int:
        """Maximum number of GPUs this datacenter can host."""
        return self._total_gpu_capacity

    def current_gpu_usage(self) -> int:
        """Current total GPU usage across all models (initial + offsets)."""
        total = 0
        for ms in self._models:
            initial = self._replica_counts.get(ms.model_label, 0)
            effective = max(0, initial + self._replica_offset_by_model.get(ms.model_label, 0))
            total += effective * ms.gpus_per_replica
        return total

    def available_gpu_capacity(self) -> int:
        """Remaining GPU slots available for incoming replicas."""
        return max(0, self._total_gpu_capacity - self.current_gpu_usage())

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    def step(self, clock: SimulationClock, events: EventEmitter) -> OfflineDatacenterState:
        t_now = clock.time_s
        template_store = self._workload.inference_data.power_templates

        # Build per-GPU power (indexed into shared pool) and effective replica counts.
        pool = self._pool
        per_gpu_by_model: dict[str, np.ndarray] = {}
        replica_counts: dict[str, int] = {}
        for ms in self._models:
            label = ms.model_label
            if self._replica_counts.get(label, 0) <= 0:
                continue
            batch = int(self._batch_by_model[label])

            template = template_store.template(label, batch)
            indices = (self._global_step + pool.stagger_offsets) % len(template)
            per_gpu_by_model[label] = template[indices]

            schedule = self._model_schedules[label]
            offset = self._replica_offset_by_model.get(label, 0)
            replica_counts[label] = max(0, int(round(schedule.count_at(t_now))) + offset)

        inference_aug = self._inference_augmenter.augment(per_gpu_by_model, replica_counts)

        power_by_model = dict(inference_aug.power_by_model_w)
        active_replicas_by_model = dict(inference_aug.active_replicas_by_model)
        for ms in self._models:
            power_by_model.setdefault(ms.model_label, 0.0)
            active_replicas_by_model.setdefault(ms.model_label, 0)

        # This is where we accumulate power across workloads.
        phase_power = np.array(
            [
                self._base_W_per_phase + inference_aug.power_w.a,
                self._base_W_per_phase + inference_aug.power_w.b,
                self._base_W_per_phase + inference_aug.power_w.c,
            ]
        )

        # Training overlay
        for run, t_start, t_end in self._workload.training:
            training_power_w = run.eval_power(float(t_now), t_start, t_end)
            phase_power += training_power_w / 3.0

        # ITL sampling
        itl_fits = self._workload.inference_data.itl_fits
        observed_itl_s_by_model: dict[str, float] = {}
        for ms in self._models:
            label = ms.model_label
            n_rep = active_replicas_by_model.get(label, 0)
            if itl_fits is None or n_rep <= 0:
                observed_itl_s_by_model[label] = float("nan")
                continue
            batch = int(self._batch_by_model[label])
            observed_itl_s_by_model[label] = itl_fits.sample_avg(
                model_label=label,
                batch_size=batch,
                n_replicas=n_rep,
                rng=self._latency_rng,
            )

        state = OfflineDatacenterState(
            time_s=float(t_now),
            power_w=ThreePhase(
                a=float(phase_power[0]),
                b=float(phase_power[1]),
                c=float(phase_power[2]),
            ),
            power_by_model_w=power_by_model,
            active_replicas_by_model=active_replicas_by_model,
            batch_size_by_model=dict(self._batch_by_model),
            observed_itl_s_by_model=observed_itl_s_by_model,
        )
        self._global_step += 1
        return state

    @functools.singledispatchmethod
    def apply_control(self, command: DatacenterCommand, events: EventEmitter) -> None:
        """Apply a control command. Dispatches on command type."""
        raise TypeError(f"OfflineDatacenter does not support {type(command).__name__}")

    @apply_control.register
    def apply_control_set_batch_size(self, command: SetBatchSize, events: EventEmitter) -> None:
        """Record new batch sizes. Changes take effect on the next step."""
        if command.ramp_up_rate_by_model:
            raise ValueError(
                f"OfflineDatacenter does not support ramp_up_rate_by_model (got {command.ramp_up_rate_by_model}). "
                f"Batch size changes are always immediate in trace-based simulation."
            )
        for label, b in command.batch_size_by_model.items():
            b_int = int(b)
            if b_int <= 0:
                raise ValueError(f"Batch size must be positive for model {label!r}, got {b_int}.")
            old = self._batch_by_model.get(str(label))
            self._batch_by_model[str(label)] = b_int
            if old != b_int:
                logger.info("Batch size %s: %s -> %d", label, old, b_int)
        events.emit(
            "datacenter.batch_size.updated",
            {"batch_size_by_model": dict(self._batch_by_model)},
        )

    @apply_control.register
    def apply_control_shift_replicas(self, command: ShiftReplicas, events: EventEmitter) -> None:
        """Shift replicas for a model by adjusting the replica offset."""
        label = command.model_label
        delta = command.replica_delta
        if label not in self._replica_offset_by_model:
            raise ValueError(f"ShiftReplicas: unknown model {label!r}")

        self._replica_offset_by_model[label] += delta

        initial = self._replica_counts.get(label, 0)
        effective = max(0, initial + self._replica_offset_by_model[label])
        events.emit(
            "datacenter.replicas.shifted",
            {"model_label": label, "replica_delta": delta, "effective_replicas": effective},
        )

    def reset(self) -> None:
        self._global_step = 0
        self._batch_by_model = {
            ms.model_label: self._workload.initial_batch_sizes.get(ms.model_label, ms.feasible_batch_sizes[0])
            for ms in self._models
        }
        self._replica_offset_by_model = {ms.model_label: 0 for ms in self._models}
        self._layout_rng = np.random.default_rng(self._seed)
        self._model_schedules = {
            ms.model_label: self._workload.replica_schedules[ms.model_label]
            for ms in self._models
            if ms.model_label in self._workload.replica_schedules
        }
        self._pool = self._build_server_pool()
        self._inference_augmenter = InferencePowerAugmenter(
            pool=self._pool,
            gpus_per_replica_by_model={ms.model_label: ms.gpus_per_replica for ms in self._models},
            seed=self._seed + 12345,
        )
        self._latency_rng = np.random.default_rng(self._seed + 54321)

    def _build_server_pool(self) -> ServerPool:
        """Build shared server pool for the datacenter."""
        rng = self._layout_rng
        gpus_per_server = self._datacenter.gpus_per_server
        amp_lo, amp_hi = self._power_augmentation.amplitude_scale_range
        noise_fraction = self._power_augmentation.noise_fraction
        template_store = self._workload.inference_data.power_templates

        num_servers = math.ceil(self._total_gpu_capacity / gpus_per_server)

        # Server properties (model-independent)
        sA, sB, sC = split_integer_evenly(num_servers, 3)
        phase_list = np.asarray(([0] * sA) + ([1] * sB) + ([2] * sC), dtype=int)
        rng.shuffle(phase_list)

        # Stagger offsets: use max template length across all models
        max_tpl_len = 1
        for ms in self._models:
            if ms.model_label in self._replica_counts and self._replica_counts[ms.model_label] > 0:
                any_batch = template_store.batch_sizes(ms.model_label)[0]
                tpl_len = len(template_store.template(ms.model_label, any_batch))
                max_tpl_len = max(max_tpl_len, tpl_len)
        stagger_offsets = rng.integers(low=0, high=max_tpl_len, size=num_servers)

        amplitude_scales = rng.uniform(amp_lo, amp_hi, size=num_servers)

        # Per-model priority orderings
        model_priorities: dict[str, np.ndarray] = {}
        for ms in self._models:
            priority = np.arange(num_servers, dtype=int)
            rng.shuffle(priority)
            model_priorities[ms.model_label] = priority

        return ServerPool(
            num_servers=num_servers,
            gpus_per_server=gpus_per_server,
            phase_list=phase_list,
            stagger_offsets=stagger_offsets,
            amplitude_scales=amplitude_scales,
            noise_fraction=noise_fraction,
            model_priorities=model_priorities,
        )

    @property
    def phase_share_by_model(self) -> dict[str, np.ndarray]:
        """Per-model phase share vectors based on current pool allocation.

        Returns:
            Mapping of model label to a 3-element array `[frac_A, frac_B, frac_C]`
                representing the fraction of allocated servers on each phase.
        """
        # Compute current allocation
        gpu_demands = {}
        for ms in self._models:
            label = ms.model_label
            initial = self._replica_counts.get(label, 0)
            offset = self._replica_offset_by_model.get(label, 0)
            effective = max(0, initial + offset)
            gpu_demands[label] = effective * ms.gpus_per_replica

        allocation = self._pool.allocate(gpu_demands)

        shares: dict[str, np.ndarray] = {}
        for label, server_indices in allocation.items():
            if len(server_indices) == 0:
                continue
            counts = np.bincount(self._pool.phase_list[server_indices], minlength=3).astype(float)
            total = counts.sum()
            shares[label] = counts / total if total > 0 else np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        return shares
