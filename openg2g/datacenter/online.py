"""Online (live GPU) datacenter backend using Zeus power monitoring.

Requires ``pip install zeus`` for GPU power measurement.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from openg2g.clock import SimulationClock
from openg2g.datacenter.base import DatacenterBackend
from openg2g.types import Command, OnlineDatacenterState, ThreePhase

logger = logging.getLogger(__name__)


class OnlineDatacenter(DatacenterBackend):
    """Live GPU power measurement backend using Zeus.

    Polls GPU power at the configured rate and aggregates per-phase power
    based on GPU-to-phase assignments.

    Args:
        gpu_indices: List of GPU device indices to monitor.
        dt_s: Polling interval in seconds.
        phase_assignment: Mapping from GPU index to phase (0=A, 1=B, 2=C).
            GPUs not in this mapping are assigned round-robin.
        batch_control_callback: Callable invoked with
            ``{model_label: batch_size}`` when ``apply_control`` receives new
            batch sizes.  Typically sends an HTTP request to the inference
            server.
        power_tolerance_s: Maximum age (seconds) of a power reading before a
            warning is issued.  Only relevant in live mode.
    """

    def __init__(
        self,
        *,
        gpu_indices: list[int],
        dt_s: float = 0.1,
        phase_assignment: dict[int, int] | None = None,
        batch_control_callback: Callable[[dict[str, int]], None] | None = None,
        replica_count_provider: Callable[[], dict[str, int]] | None = None,
        power_tolerance_s: float = 0.5,
    ):
        try:
            from zeus.device import get_gpus
        except ImportError as exc:
            raise ImportError(
                "OnlineDatacenter requires the Zeus library. Install it with: pip install zeus"
            ) from exc

        self._dt = float(dt_s)
        self._gpu_indices = list(gpu_indices)
        self._batch_callback = batch_control_callback
        # Optional hook for HIL setups (e.g., vLLM replicas) to report
        # active replica counts per model each step.
        self._replica_provider = replica_count_provider
        self._power_tolerance = float(power_tolerance_s)

        # Phase assignment: default round-robin
        if phase_assignment is None:
            phase_assignment = {idx: i % 3 for i, idx in enumerate(gpu_indices)}
        self._phase_map = {int(k): int(v) for k, v in phase_assignment.items()}

        # Zeus GPU handles
        self._gpus = get_gpus()

        # Internal state
        self._last_power: dict[int, float] = {idx: 0.0 for idx in gpu_indices}
        self._last_read_time: float = 0.0
        self._batch_by_model: dict[str, int] = {}
        self._step_count: int = 0

    @property
    def dt_s(self) -> float:
        return self._dt

    def step(self, clock: SimulationClock) -> OnlineDatacenterState:
        """Read current GPU power and return per-phase aggregated state."""
        gpu_power: dict[int, float] = {}
        phase_power = [0.0, 0.0, 0.0]

        for idx in self._gpu_indices:
            try:
                power_w = self._gpus[idx].getInstantPowerUsage()  # type: ignore[index]
                gpu_power[idx] = float(power_w)
                self._last_power[idx] = float(power_w)
            except Exception as exc:
                logger.warning("Failed to read GPU %d power: %s", idx, exc)
                gpu_power[idx] = self._last_power.get(idx, 0.0)

            ph = self._phase_map.get(idx, idx % 3)
            phase_power[ph] += gpu_power[idx]

        self._last_read_time = time.monotonic()
        self._step_count += 1

        active_replicas: dict[str, int] = {}
        if self._replica_provider is not None:
            try:
                raw = self._replica_provider()
                active_replicas = {str(k): int(v) for k, v in raw.items()}
            except Exception as exc:
                logger.warning("Replica count provider failed: %s", exc)

        return OnlineDatacenterState(
            time_s=clock.time_s,
            power_w=ThreePhase(a=phase_power[0], b=phase_power[1], c=phase_power[2]),
            gpu_power_readings=gpu_power,
            batch_size_by_model=dict(self._batch_by_model),
            active_replicas_by_model=active_replicas,
        )

    def apply_control(self, command: Command) -> None:
        """Apply batch size command via the control callback."""
        if command.kind != "set_batch_size":
            raise ValueError(f"OnlineDatacenter does not support command kind={command.kind!r}")
        batch_map = command.payload.get("batch_size_by_model", {})
        if not isinstance(batch_map, dict):
            raise ValueError("set_batch_size requires payload['batch_size_by_model'] as a dict.")
        self._batch_by_model.update({str(k): int(v) for k, v in batch_map.items()})
        if self._batch_callback is not None:
            try:
                self._batch_callback(dict(self._batch_by_model))
            except Exception as exc:
                logger.error("Batch control callback failed: %s", exc)
