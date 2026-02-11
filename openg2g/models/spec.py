"""Model specification dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Specification for one LLM model served in the datacenter."""

    model_label: str
    replicas: int
    gpus_per_replica: int
