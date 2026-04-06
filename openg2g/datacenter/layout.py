"""Server layout and activation policy primitives.

Provides the topology and activation-policy building blocks used by
datacenter backends. Power augmentation (scaling per-GPU power to
three-phase datacenter power) lives in
`openg2g.datacenter.workloads.inference`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from openg2g.datacenter.config import InferenceRampSchedule


class ActivationPolicy(ABC):
    """Per-model activation policy that answers "which servers are active?"

    Subclass to implement custom activation logic. The datacenter creates
    one policy per model and passes it to
    [`InferencePowerAugmenter`][openg2g.datacenter.workloads.inference.InferencePowerAugmenter].
    """

    @abstractmethod
    def active_mask(self, t: float) -> np.ndarray:
        """Boolean mask of active servers at time *t*.

        Returns:
            Array of shape `(num_servers,)` with `True` for active servers.
        """

    def active_indices(self, t: float) -> np.ndarray:
        """Indices of active servers at time *t*.

        The default implementation returns indices in ascending order
        via `np.where(`[`active_mask`][..active_mask]`(t))`. Subclasses
        may override to return
        indices in a specific order (e.g., priority order) to control
        floating-point summation order in the datacenter.

        Returns:
            1-D int array of active server indices.
        """
        return np.where(self.active_mask(t))[0]


class RampActivationPolicy(ActivationPolicy):
    """Activate servers by fixed random priority, following an
    [`InferenceRampSchedule`][openg2g.datacenter.config.InferenceRampSchedule].

    At time *t*, the top-*k* servers (by random priority) are active,
    where *k* is derived from the schedule's absolute replica count and
    the model's GPU requirements.

    This is the default policy used by
    [`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter].

    Args:
        schedule: Per-model ramp schedule (absolute replica counts).
        num_servers: Total allocated servers (may exceed baseline when
            ramp targets exceed the initial replica count).
        rng: RNG for randomizing priority ordering. Consumed once at
            construction time.
        gpus_per_replica: GPUs required per model replica.
        gpus_per_server: GPUs per physical server.
    """

    __slots__ = ("_gpus_per_replica", "_gpus_per_server", "_n", "_priority", "_replica_offset", "_schedule")

    def __init__(
        self,
        schedule: InferenceRampSchedule,
        num_servers: int,
        rng: np.random.Generator,
        *,
        gpus_per_replica: int,
        gpus_per_server: int,
    ) -> None:
        self._schedule = schedule
        self._n = num_servers
        self._gpus_per_replica = gpus_per_replica
        self._gpus_per_server = gpus_per_server
        self._replica_offset: int = 0
        priority = np.arange(num_servers, dtype=int)
        rng.shuffle(priority)
        self._priority = priority

    def _servers_for_count(self, replica_count: float) -> int:
        """Convert a replica count to a server count."""
        import math

        gpus_needed = max(0.0, replica_count + self._replica_offset) * self._gpus_per_replica
        return max(0, min(self._n, math.ceil(gpus_needed / self._gpus_per_server)))

    def active_mask(self, t: float) -> np.ndarray:
        count = self._schedule.count_at(t)
        k = self._servers_for_count(float(count))
        mask = np.zeros(self._n, dtype=bool)
        mask[self._priority[:k]] = True
        return mask

    def active_indices(self, t: float) -> np.ndarray:
        """Return active server indices in priority order."""
        count = self._schedule.count_at(t)
        k = self._servers_for_count(float(count))
        return self._priority[:k].copy()


@dataclass
class ServerLayout:
    """Per-model server layout describing how GPUs are organized.

    This describes the physical topology only. Activation policies (which
    servers are on/off at a given time) are managed separately by the
    datacenter and passed to
    [`InferencePowerAugmenter`][openg2g.datacenter.workloads.inference.InferencePowerAugmenter]
    alongside layouts.

    Attributes:
        num_servers: Number of physical servers for this model.
        total_gpus: Total GPU count across all servers.
        gpus_per_replica: GPUs per model replica.
        gpus_per_server_list: GPU count per server (last may be partial).
        phase_list: Phase assignment per server (0=A, 1=B, 2=C).
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
    stagger_offsets: np.ndarray
    amplitude_scales: np.ndarray
    noise_fraction: float
