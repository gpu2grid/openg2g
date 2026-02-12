"""Model specification dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Specification for one LLM model served in the datacenter.

    Attributes:
        model_label: Human-readable model identifier (e.g. ``"Llama-3.1-70B"``).
        replicas: Total number of replicas of this model across the datacenter.
        gpus_per_replica: GPUs allocated to each replica (determines model
            parallelism and per-replica power draw).
    """

    model_label: str
    replicas: int
    gpus_per_replica: int
