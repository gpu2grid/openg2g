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
    InferenceRampSchedule,
    PowerAugmentationConfig,
    TrainingSchedule,
)
from openg2g.datacenter.layout import (
    ActivationPolicy,
    RampActivationPolicy,
    ServerLayout,
)
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

    Bundles inference data with optional training overlays and inference
    server ramp events.

    Attributes:
        inference_data: LLM inference workload with offline simulation
            data (model specs, power templates, ITL fits).
        inference_ramps: Inference server ramp schedule. `None` keeps all
            servers active.
        training: Training workload schedule. `None` disables training
            overlay.
    """

    inference_data: InferenceData
    inference_ramps: InferenceRampSchedule = field(default_factory=InferenceRampSchedule)
    training: TrainingSchedule = field(default_factory=TrainingSchedule)


class OfflineDatacenter(LLMBatchSizeControlledDatacenter[OfflineDatacenterState]):
    """Trace-based datacenter simulation with step-by-step interface.

    Each `step` call computes one timestep of power output by indexing
    into pre-built per-GPU templates, applying per-server amplitude
    scaling and noise, and summing across active servers per phase.

    Batch size changes via `apply_control` take effect on the next
    `step` call.

    If `workload.inference_ramps` is set, a
    [`RampActivationPolicy`][openg2g.datacenter.layout.RampActivationPolicy]
    is created per model.

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
        dt_s: Fraction,
        seed: int = 0,
        power_augmentation: PowerAugmentationConfig | None = None,
        load_shift_headroom: float = 0.0,
        total_gpu_capacity: int | None = None,
    ) -> None:
        super().__init__()
        if power_augmentation is None:
            power_augmentation = PowerAugmentationConfig()

        self._datacenter = datacenter
        self._workload = workload
        self._power_augmentation = power_augmentation
        self._load_shift_headroom = load_shift_headroom

        # Total GPU capacity: if not specified, compute from initial model allocation
        models = list(workload.inference_data.models)
        if total_gpu_capacity is None:
            total_gpu_capacity = sum(ms.initial_num_replicas * ms.gpus_per_replica for ms in models)
        self._total_gpu_capacity = total_gpu_capacity
        self._dt_s = dt_s
        self._seed = int(seed)
        self._models = list(workload.inference_data.models)
        self._base_W_per_phase = float(datacenter.base_kw_per_phase) * 1e3

        self._layout_rng = np.random.default_rng(self._seed)
        self._batch_by_model: dict[str, int] = {ms.model_label: ms.initial_batch_size for ms in self._models}

        self._layouts: dict[str, ServerLayout] = {}
        self._policies: dict[str, ActivationPolicy] = {}
        self._build_all_layouts()
        self._inference_augmenter = InferencePowerAugmenter(
            layouts=self._layouts,
            policies=self._policies,
            seed=self._seed + 12345,
        )

        self._global_step: int = 0
        self._latency_rng = np.random.default_rng(self._seed + 54321)
        self._replica_offset_by_model: dict[str, int] = {ms.model_label: 0 for ms in self._models}

        logger.info(
            "OfflineDatacenter: %d models, dt=%s s, seed=%d",
            len(self._models),
            dt_s,
            seed,
        )
        for ms in self._models:
            logger.info(
                "  %s: %d replicas, %d GPUs/replica, batch=%d",
                ms.model_label,
                ms.initial_num_replicas,
                ms.gpus_per_replica,
                ms.initial_batch_size,
            )

    @property
    def total_gpu_capacity(self) -> int:
        """Maximum number of GPUs this datacenter can host."""
        return self._total_gpu_capacity

    def current_gpu_usage(self) -> int:
        """Current total GPU usage across all models (initial + offsets)."""
        total = 0
        for ms in self._models:
            effective = max(0, ms.initial_num_replicas + self._replica_offset_by_model.get(ms.model_label, 0))
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

        # Build per-GPU power dict by indexing into templates with layout offsets.
        per_gpu_by_model: dict[str, np.ndarray] = {}
        for ms in self._models:
            label = ms.model_label
            if ms.initial_num_replicas <= 0:
                continue
            batch = int(self._batch_by_model[label])

            layout = self._layouts[label]
            template = template_store.template(label, batch)
            indices = (self._global_step + layout.stagger_offsets) % len(template)
            per_gpu_by_model[label] = template[indices]

        inference_aug = self._inference_augmenter.augment(per_gpu_by_model, t_now)

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
        """Shift replicas for a model by adjusting the activation policy base count."""
        label = command.model_label
        delta = command.replica_delta
        if label not in self._replica_offset_by_model:
            logger.warning("ShiftReplicas: unknown model %s, ignoring", label)
            return

        old_offset = self._replica_offset_by_model[label]
        new_offset = old_offset + delta

        # Find the model spec to compute server delta
        ms = next((m for m in self._models if m.model_label == label), None)
        if ms is None or ms.initial_num_replicas <= 0:
            return
        gpus_per_server = self._datacenter.gpus_per_server

        # Compute new effective replica count (clamped to >= 0)
        effective_replicas = max(0, ms.initial_num_replicas + new_offset)

        # Enforce total GPU capacity when adding replicas
        if delta > 0:
            gpus_needed = delta * ms.gpus_per_replica
            available = self.available_gpu_capacity()
            if gpus_needed > available:
                # Clamp to available capacity
                max_replicas = available // ms.gpus_per_replica
                if max_replicas <= 0:
                    logger.warning(
                        "ShiftReplicas %s: rejected, no GPU capacity (need %d, have %d)",
                        label,
                        gpus_needed,
                        available,
                    )
                    return
                effective_replicas = ms.initial_num_replicas + old_offset + max_replicas
                logger.info(
                    "ShiftReplicas %s: clamped from %+d to %+d replicas (GPU cap %d, used %d)",
                    label,
                    delta,
                    max_replicas,
                    self._total_gpu_capacity,
                    self.current_gpu_usage(),
                )

        math.ceil(ms.initial_num_replicas * ms.gpus_per_replica / gpus_per_server)
        new_base = math.ceil(effective_replicas * ms.gpus_per_replica / gpus_per_server)

        # Clamp to allocated server capacity
        policy = self._policies.get(label)
        if policy is None:
            return
        new_base = max(0, min(new_base, policy._n))

        old_base = policy._base
        policy._base = new_base
        self._replica_offset_by_model[label] = effective_replicas - ms.initial_num_replicas

        if old_base != new_base:
            logger.info(
                "ShiftReplicas %s: offset %+d -> %+d, base_servers %d -> %d (cap %d)",
                label,
                old_offset,
                self._replica_offset_by_model[label],
                old_base,
                new_base,
                policy._n,
            )

        events.emit(
            "datacenter.replicas.shifted",
            {"model_label": label, "replica_delta": delta, "new_base_servers": new_base},
        )

    def reset(self) -> None:
        self._global_step = 0
        self._batch_by_model = {ms.model_label: ms.initial_batch_size for ms in self._models}
        self._replica_offset_by_model = {ms.model_label: 0 for ms in self._models}
        self._layout_rng = np.random.default_rng(self._seed)
        self._layouts = {}
        self._policies = {}
        self._build_all_layouts()
        self._inference_augmenter = InferencePowerAugmenter(
            layouts=self._layouts,
            policies=self._policies,
            seed=self._seed + 12345,
        )
        self._latency_rng = np.random.default_rng(self._seed + 54321)

    def _build_all_layouts(self) -> None:
        """Build layouts and activation policies for all models."""
        schedule = self._workload.inference_ramps
        rng = self._layout_rng
        gpus_per_server = self._datacenter.gpus_per_server
        amp_lo, amp_hi = self._power_augmentation.amplitude_scale_range
        noise_fraction = self._power_augmentation.noise_fraction
        template_store = self._workload.inference_data.power_templates

        for ms in self._models:
            if ms.initial_num_replicas > 0:
                any_batch = template_store.batch_sizes(ms.model_label)[0]
                tpl_len = len(template_store.template(ms.model_label, any_batch))

                # Per-model ramp schedule (filters by model label).
                model_schedule = schedule.for_model(ms.model_label)
                max_target = model_schedule.max_target()

                base_servers = math.ceil(ms.initial_num_replicas * ms.gpus_per_replica / gpus_per_server)
                # Allocate extra servers when ramps scale beyond 1.0 or load-shift headroom is set.
                effective_max = max(max_target, 1.0 + self._load_shift_headroom)
                num_servers = math.ceil(base_servers * effective_max) if effective_max > 1.0 else base_servers

                # Phase shuffle
                sA, sB, sC = split_integer_evenly(num_servers, 3)
                phase_list = np.asarray(([0] * sA) + ([1] * sB) + ([2] * sC), dtype=int)
                rng.shuffle(phase_list)

                # Policy dictates which servers are active at a given time.
                self._policies[ms.model_label] = RampActivationPolicy(
                    model_schedule,
                    num_servers,
                    rng,
                    base_servers=base_servers if effective_max > 1.0 else None,
                )

                # This offset determines for each server, how much to stagger its power template indexing.
                stagger_offsets = rng.integers(low=0, high=max(tpl_len, 1), size=num_servers)

                # Amplitude scales
                amplitude_scales = rng.uniform(amp_lo, amp_hi, size=num_servers)

                total_gpus = num_servers * gpus_per_server
                gpus_per_server_list = np.full(num_servers, gpus_per_server, dtype=int)
                # Adjust last server for the actual GPU count at peak.
                peak_gpus = int(math.ceil(ms.initial_num_replicas * ms.gpus_per_replica * effective_max))
                tail = peak_gpus - (num_servers - 1) * gpus_per_server
                gpus_per_server_list[-1] = max(1, int(tail)) if tail > 0 else gpus_per_server
                total_gpus = int(gpus_per_server_list.sum())

                self._layouts[ms.model_label] = ServerLayout(
                    num_servers=num_servers,
                    total_gpus=total_gpus,
                    gpus_per_replica=ms.gpus_per_replica,
                    gpus_per_server_list=gpus_per_server_list,
                    phase_list=phase_list,
                    stagger_offsets=stagger_offsets,
                    amplitude_scales=amplitude_scales,
                    noise_fraction=noise_fraction,
                )

    @property
    def phase_share_by_model(self) -> dict[str, np.ndarray]:
        """Per-model phase share vectors derived from server placement.

        Returns:
            Mapping of model label to a 3-element array `[frac_A, frac_B, frac_C]`
                representing the fraction of servers on each phase.
        """
        shares: dict[str, np.ndarray] = {}
        for label, layout in self._layouts.items():
            counts = np.bincount(layout.phase_list, minlength=3).astype(float)
            total = counts.sum()
            if total > 0:
                shares[label] = counts / total
            else:
                shares[label] = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        return shares
