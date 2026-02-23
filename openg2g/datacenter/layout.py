"""Server layout and power augmentation primitives.

Provides the shared components for scaling per-GPU power measurements
to datacenter-level three-phase power output. These primitives are
backend-agnostic and can be used by both offline (trace-based) and
online (live GPU) datacenters.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from openg2g.datacenter.config import ServerRampSchedule
from openg2g.models.spec import LLMInferenceModelSpec
from openg2g.types import ThreePhase
from openg2g.utils import split_integer_evenly


class ActivationPolicy(ABC):
    """Per-model activation policy that answers "which servers are active?"

    Created by `ActivationStrategy.for_model` and bound to a specific
    model's server pool.
    """

    @abstractmethod
    def active_mask(self, t: float) -> np.ndarray:
        """Boolean mask of active servers at time *t*.

        Returns:
            Array of shape `(num_servers,)` with `True` for active servers.
        """

    def active_indices(self, t: float) -> np.ndarray:
        """Indices of active servers at time *t*.

        The default implementation returns indices in ascending order via
        `np.where(active_mask(t))`. Subclasses may override to return
        indices in a specific order (e.g., priority order) to control
        floating-point summation order in the datacenter.

        Returns:
            1-D int array of active server indices.
        """
        return np.where(self.active_mask(t))[0]


class ActivationStrategy(ABC):
    """Factory that creates per-model `ActivationPolicy` instances.

    A strategy is instantiated once and passed to the datacenter. When
    the datacenter builds each model's server layout, it calls
    `for_model` to create a model-specific `ActivationPolicy`.

    Subclass to implement custom activation strategies. The `phase_list`
    argument in `for_model` enables phase-aware load balancing.
    """

    @abstractmethod
    def for_model(
        self,
        *,
        num_servers: int,
        phase_list: np.ndarray,
        rng: np.random.Generator,
    ) -> ActivationPolicy:
        """Create a policy for one model's server pool.

        Args:
            num_servers: Number of physical servers for this model.
            phase_list: Phase assignment per server (0=A, 1=B, 2=C), shape
                `(num_servers,)`.
            rng: RNG for randomized decisions (priority ordering, etc.).
                Implementations must consume RNG calls deterministically
                so that downstream layout generation is reproducible.

        Returns:
            Policy that answers `active_mask(t)` queries.
        """


class _RampActivationPolicy(ActivationPolicy):
    """Policy for `RampActivationStrategy`."""

    __slots__ = ("_n", "_priority", "_schedule")

    def __init__(
        self,
        schedule: ServerRampSchedule,
        num_servers: int,
        priority: np.ndarray,
    ) -> None:
        self._schedule = schedule
        self._n = num_servers
        self._priority = priority

    def active_mask(self, t: float) -> np.ndarray:
        frac = self._schedule.fraction_at(t)
        k = max(0, min(self._n, int(round(float(frac) * self._n))))
        mask = np.zeros(self._n, dtype=bool)
        mask[self._priority[:k]] = True
        return mask

    def active_indices(self, t: float) -> np.ndarray:
        """Return active server indices in priority order."""
        frac = self._schedule.fraction_at(t)
        k = max(0, min(self._n, int(round(float(frac) * self._n))))
        return self._priority[:k].copy()


class RampActivationStrategy(ActivationStrategy):
    """Activate servers by fixed random priority, following a `ServerRampSchedule`.

    At time *t*, the top-*k* servers (by random priority) are active, where
    `k = round(schedule.fraction_at(t) * num_servers)`.

    This is the default strategy used by `OfflineDatacenter`.

    Args:
        schedule: Temporal ramp schedule mapping time to active-server fraction.
    """

    def __init__(self, schedule: ServerRampSchedule) -> None:
        self._schedule = schedule

    def for_model(
        self,
        *,
        num_servers: int,
        phase_list: np.ndarray,
        rng: np.random.Generator,
    ) -> ActivationPolicy:
        priority = np.arange(num_servers, dtype=int)
        rng.shuffle(priority)
        return _RampActivationPolicy(self._schedule, num_servers, priority)


@dataclass
class ServerLayout:
    """Per-model server layout describing how GPUs are organized.

    Attributes:
        num_servers: Number of physical servers for this model.
        total_gpus: Total GPU count across all servers.
        gpus_per_replica: GPUs per model replica.
        gpus_per_server_list: GPU count per server (last may be partial).
        phase_list: Phase assignment per server (0=A, 1=B, 2=C).
        activation_policy: Determines which servers are active at time *t*.
        stagger_offsets: Per-server offsets for desynchronization. In offline
            mode these are integer indices into a power template; in online
            mode they can be float time offsets into a rolling buffer.
        amplitude_scales: Per-server power multiplier for inter-server variation.
        noise_fraction: Gaussian noise standard deviation as a fraction of
            per-server power.
    """

    num_servers: int
    total_gpus: int
    gpus_per_replica: int
    gpus_per_server_list: np.ndarray
    phase_list: np.ndarray
    activation_policy: ActivationPolicy
    stagger_offsets: np.ndarray
    amplitude_scales: np.ndarray
    noise_fraction: float


def build_server_layout(
    model_spec: LLMInferenceModelSpec,
    *,
    gpus_per_server: int,
    template_length: int,
    activation_strategy: ActivationStrategy,
    amplitude_scale_range: tuple[float, float],
    noise_fraction: float,
    rng: np.random.Generator,
) -> ServerLayout:
    """Build a server layout for one model.

    This is a pure function of its inputs (plus RNG state). The caller
    is responsible for providing a consistently-seeded RNG so that
    layout generation is reproducible.

    Args:
        model_spec: Model specification (replicas, GPUs per replica, etc.).
        gpus_per_server: Number of GPUs per physical server rack.
        template_length: Length of the power template array, used to bound
            stagger offsets.
        activation_strategy: Strategy for determining active servers.
        amplitude_scale_range: `(low, high)` range for per-server amplitude
            scaling. Each server draws a uniform multiplier from this range.
        noise_fraction: Gaussian noise standard deviation as a fraction
            of per-server power.
        rng: Random number generator (consumed for phase assignment,
            activation policy, stagger offsets, and amplitude scales).

    Returns:
        Frozen `ServerLayout` for the model.
    """
    num_replicas = int(model_spec.num_replicas)
    gpus_per_replica = int(model_spec.gpus_per_replica)
    total_gpus = num_replicas * gpus_per_replica
    num_servers = int(math.ceil(total_gpus / gpus_per_server))

    gpus_per_server_list = np.full(num_servers, gpus_per_server, dtype=int)
    tail = total_gpus - (num_servers - 1) * gpus_per_server
    gpus_per_server_list[-1] = int(tail) if tail > 0 else gpus_per_server

    sA, sB, sC = split_integer_evenly(num_servers, 3)
    phase_list = np.asarray(([0] * sA) + ([1] * sB) + ([2] * sC), dtype=int)
    rng.shuffle(phase_list)

    bound_policy = activation_strategy.for_model(
        num_servers=num_servers,
        phase_list=phase_list,
        rng=rng,
    )

    stagger_offsets = rng.integers(low=0, high=max(template_length, 1), size=num_servers)
    amplitude_scales = rng.uniform(
        float(amplitude_scale_range[0]),
        float(amplitude_scale_range[1]),
        size=num_servers,
    )

    return ServerLayout(
        num_servers=num_servers,
        total_gpus=total_gpus,
        gpus_per_replica=gpus_per_replica,
        gpus_per_server_list=gpus_per_server_list,
        phase_list=phase_list,
        activation_policy=bound_policy,
        stagger_offsets=stagger_offsets,
        amplitude_scales=amplitude_scales,
        noise_fraction=float(noise_fraction),
    )


@dataclass(frozen=True)
class AugmentedPower:
    """Result of power augmentation for one simulation timestep.

    Attributes:
        power_w: Three-phase total power (watts), including base load.
        power_by_model_w: Per-model total active power (watts).
        active_replicas_by_model: Per-model active replica count.
    """

    power_w: ThreePhase
    power_by_model_w: dict[str, float] = field(default_factory=dict)
    active_replicas_by_model: dict[str, int] = field(default_factory=dict)


class PowerAugmenter:
    """Scales per-GPU power through server layouts to three-phase power.

    Given per-GPU power values for each server (one value per server per
    model), applies per-server scaling, noise, activation masking, and
    phase summation to produce datacenter-level three-phase power.

    This class is backend-agnostic. The offline datacenter feeds it
    template-indexed values; the online datacenter can feed it
    live-measured values.

    Args:
        layouts: Per-model server layouts.
        base_w_per_phase: Constant base load per phase (watts).
        seed: Random seed for noise RNG.
    """

    def __init__(
        self,
        layouts: dict[str, ServerLayout],
        base_w_per_phase: float = 0.0,
        seed: int = 0,
    ) -> None:
        self._layouts = layouts
        self._base_w_per_phase = float(base_w_per_phase)
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    def step(
        self,
        per_gpu_by_model: dict[str, np.ndarray],
        t: float,
    ) -> AugmentedPower:
        """Augment per-server per-GPU power to three-phase power.

        Args:
            per_gpu_by_model: Mapping of model label to per-GPU power
                array of shape `(num_servers,)`. Only models with active
                replicas should be included.
            t: Current simulation time (seconds), passed to activation
                policies.

        Returns:
            `AugmentedPower` with three-phase power, per-model power,
            and per-model active replica counts.
        """
        phase_power = np.full(3, self._base_w_per_phase, dtype=float)
        power_by_model: dict[str, float] = {}
        active_replicas_by_model: dict[str, int] = {}

        for label, per_gpu in per_gpu_by_model.items():
            layout = self._layouts[label]

            server_powers = per_gpu * layout.gpus_per_server_list * layout.amplitude_scales
            if layout.noise_fraction > 0:
                levels = np.maximum(server_powers, 1.0)
                server_powers = (
                    server_powers + self._rng.normal(0.0, 1.0, size=layout.num_servers) * layout.noise_fraction * levels
                )
            server_powers = np.maximum(server_powers, 0.0)

            active_indices = layout.activation_policy.active_indices(t)
            active_powers = server_powers[active_indices]
            active_phases = layout.phase_list[active_indices]

            model_phase_power = np.zeros(3, dtype=float)
            np.add.at(model_phase_power, active_phases, active_powers)
            phase_power += model_phase_power

            power_by_model[label] = float(np.sum(active_powers))
            active_gpus = int(np.sum(layout.gpus_per_server_list[active_indices]))
            active_replicas_by_model[label] = active_gpus // layout.gpus_per_replica

        return AugmentedPower(
            power_w=ThreePhase(
                a=float(phase_power[0]),
                b=float(phase_power[1]),
                c=float(phase_power[2]),
            ),
            power_by_model_w=power_by_model,
            active_replicas_by_model=active_replicas_by_model,
        )

    def reset(self) -> None:
        """Re-seed the noise RNG to its initial state."""
        self._rng = np.random.default_rng(self._seed)
