"""Server pool primitives.

Provides the shared server pool used by datacenter backends.
Power augmentation (scaling per-GPU power to three-phase datacenter
power) lives in `openg2g.datacenter.workloads.inference`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ServerPool:
    """Shared pool of physical servers for a datacenter.

    All models draw servers from this single pool based on their current
    GPU demand. Each server has model-independent properties (phase
    assignment, stagger offset, amplitude scale). The pool handles
    server-to-model allocation at each timestep via per-model priority
    orderings.

    Attributes:
        num_servers: Total number of physical servers in the pool.
        gpus_per_server: GPUs per physical server.
        phase_list: Phase assignment per server (0=A, 1=B, 2=C).
        stagger_offsets: Per-server offsets for desynchronization.
        amplitude_scales: Per-server power multiplier for inter-server variation.
        noise_fraction: Gaussian noise standard deviation as a fraction of
            per-server power.
        model_priorities: Per-model priority ordering over all servers.
            Each model gets a deterministic permutation of `[0..num_servers)`.
    """

    num_servers: int
    gpus_per_server: int
    phase_list: np.ndarray
    stagger_offsets: np.ndarray
    amplitude_scales: np.ndarray
    noise_fraction: float
    model_priorities: dict[str, np.ndarray]

    def allocate(self, gpu_demands: dict[str, int]) -> dict[str, np.ndarray]:
        """Allocate servers to models with phase-balanced round-robin.

        Each model gets k = ceil(gpu_demand / gpus_per_server) servers.
        Servers are picked by cycling through phases (A, B, C, A, B, C, ...)
        and within each phase, picking the highest-priority unclaimed server.
        This ensures each model's allocation is as phase-balanced as possible.

        Args:
            gpu_demands: Mapping of model label to total GPUs needed.

        Returns:
            Mapping of model label to array of allocated server indices.
        """
        claimed = np.zeros(self.num_servers, dtype=bool)
        result: dict[str, np.ndarray] = {}

        # Pre-group servers by phase, sorted by priority per model
        phase_groups: dict[str, list[list[int]]] = {}
        for label in sorted(gpu_demands):
            priority = self.model_priorities[label]
            groups: list[list[int]] = [[], [], []]
            for idx in priority:
                groups[self.phase_list[idx]].append(idx)
            phase_groups[label] = groups

        for label in sorted(gpu_demands):
            demand = gpu_demands[label]
            if demand <= 0:
                result[label] = np.array([], dtype=int)
                continue
            k = math.ceil(demand / self.gpus_per_server)
            groups = phase_groups[label]
            cursors = [0, 0, 0]
            allocated: list[int] = []
            phase = 0

            while len(allocated) < k:
                # Try each phase starting from current, wrapping around
                found = False
                for attempt in range(3):
                    p = (phase + attempt) % 3
                    while cursors[p] < len(groups[p]):
                        idx = groups[p][cursors[p]]
                        cursors[p] += 1
                        if not claimed[idx]:
                            allocated.append(idx)
                            claimed[idx] = True
                            phase = (p + 1) % 3
                            found = True
                            break
                    if found:
                        break
                if not found:
                    raise RuntimeError(
                        f"ServerPool over-subscribed: model {label!r} requested "
                        f"{k} servers but pool has no more unclaimed servers."
                    )

            result[label] = np.array(allocated, dtype=int)

        return result
