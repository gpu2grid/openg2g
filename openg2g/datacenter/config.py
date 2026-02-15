"""Datacenter facility and workload configuration."""

from __future__ import annotations

from dataclasses import dataclass

from openg2g.models.spec import LLMInferenceWorkload
from openg2g.types import ServerRamp, ServerRampSchedule, TrainingRun, TrainingSchedule


@dataclass(frozen=True)
class DatacenterConfig:
    """Physical datacenter facility configuration.

    Attributes:
        gpus_per_server: Number of GPUs per physical server rack.
        base_kW_per_phase: Constant base load per phase (kW).
    """

    gpus_per_server: int = 8
    base_kW_per_phase: float = 0.0

    def __post_init__(self) -> None:
        if self.gpus_per_server < 1:
            raise ValueError(f"gpus_per_server must be >= 1, got {self.gpus_per_server}.")


@dataclass(frozen=True)
class WorkloadConfig:
    """What runs in the datacenter: inference, training, and ramp events.

    Attributes:
        inference: LLM inference workload specification.
        training: Training workload window(s).  `None` disables training overlay.
        server_ramps: Server ramp event(s).  `None` keeps all servers active.
    """

    inference: LLMInferenceWorkload
    training: TrainingRun | TrainingSchedule | None = None
    server_ramps: ServerRamp | ServerRampSchedule | None = None
