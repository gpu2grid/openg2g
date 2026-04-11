"""Server layout primitives.

Provides the topology building blocks used by datacenter backends.
Power augmentation (scaling per-GPU power to three-phase datacenter
power) lives in `openg2g.datacenter.workloads.inference`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ServerLayout:
    """Per-model server layout describing how GPUs are organized and activated.

    Holds the physical topology (phase assignments, stagger offsets,
    amplitude scales) and a priority ordering that determines which
    servers are active for a given replica count.

    Attributes:
        num_servers: Number of physical servers for this model.
        total_gpus: Total GPU count across all servers.
        gpus_per_replica: GPUs per model replica.
        gpus_per_server: Nominal GPUs per server (for replica-to-server conversion).
        gpus_per_server_list: GPU count per server (last may be partial).
        phase_list: Phase assignment per server (0=A, 1=B, 2=C).
        priority: Random permutation of server indices that determines
            activation order. The top-k servers by priority are active.
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
    gpus_per_server: int
    gpus_per_server_list: np.ndarray
    phase_list: np.ndarray
    priority: np.ndarray
    stagger_offsets: np.ndarray
    amplitude_scales: np.ndarray
    noise_fraction: float

    def active_indices(self, replica_count: int) -> np.ndarray:
        """Top-k server indices in priority order for the given replica count.

        Args:
            replica_count: Number of active replicas.

        Returns:
            1-D int array of active server indices in priority order.
        """
        gpus_needed = max(0, replica_count) * self.gpus_per_replica
        k = max(0, min(self.num_servers, math.ceil(gpus_needed / self.gpus_per_server)))
        return self.priority[:k].copy()

    def active_mask(self, replica_count: int) -> np.ndarray:
        """Boolean mask of active servers for the given replica count.

        Args:
            replica_count: Number of active replicas.

        Returns:
            Array of shape `(num_servers,)` with `True` for active servers.
        """
        indices = self.active_indices(replica_count)
        mask = np.zeros(self.num_servers, dtype=bool)
        mask[indices] = True
        return mask
