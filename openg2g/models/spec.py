"""Model specification and workload dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMInferenceModelSpec:
    """Specification for one LLM model served in the datacenter.

    Attributes:
        model_label: Human-readable model identifier (e.g. `"Llama-3.1-70B"`).
        num_replicas: Total number of replicas of this model across the datacenter.
        gpus_per_replica: GPUs allocated to each replica (determines model
            parallelism and per-replica power draw).
        initial_batch_size: Initial batch size for this model.
        feasible_batch_sizes: Allowed batch sizes for OFO control.  Baseline
            mode only uses the first (or only) entry.
        itl_deadline_s: Per-model inter-token latency deadline for the OFO
            latency dual (seconds).  Ignored in baseline mode.
    """

    model_label: str
    num_replicas: int
    gpus_per_replica: int
    initial_batch_size: int
    feasible_batch_sizes: tuple[int, ...] = (128,)
    itl_deadline_s: float | None = None

    def __post_init__(self) -> None:
        if not self.feasible_batch_sizes:
            raise ValueError("feasible_batch_sizes must not be empty.")
        if self.num_replicas < 0:
            raise ValueError(f"num_replicas must be >= 0, got {self.num_replicas}.")
        if self.gpus_per_replica < 1:
            raise ValueError(f"gpus_per_replica must be >= 1, got {self.gpus_per_replica}.")
        if self.initial_batch_size <= 0:
            raise ValueError(f"initial_batch_size must be > 0, got {self.initial_batch_size}.")


@dataclass(frozen=True)
class LLMInferenceWorkload:
    """Aggregation of model specs into a workload description.

    Attributes:
        models: Tuple of model specifications served in the datacenter.
    """

    models: tuple[LLMInferenceModelSpec, ...]

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("LLMInferenceWorkload requires at least one model.")
        labels = [m.model_label for m in self.models]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate model labels: {labels}")

    @property
    def model_labels(self) -> list[str]:
        """Ordered list of model labels."""
        return [m.model_label for m in self.models]

    @property
    def total_gpus(self) -> int:
        """Total GPUs consumed by all inference models."""
        return sum(m.num_replicas * m.gpus_per_replica for m in self.models)

    @property
    def initial_batch_size_by_model(self) -> dict[str, int]:
        """Per-model initial batch sizes."""
        return {m.model_label: m.initial_batch_size for m in self.models}

    @property
    def itl_deadline_by_model(self) -> dict[str, float]:
        """Per-model ITL deadlines (seconds).

        Raises:
            ValueError: If any model has `itl_deadline_s = None`.
        """
        result: dict[str, float] = {}
        for m in self.models:
            if m.itl_deadline_s is None:
                raise ValueError(f"Model {m.model_label!r} has no itl_deadline_s set.")
            result[m.model_label] = m.itl_deadline_s
        return result

    @property
    def required_measured_gpus(self) -> dict[str, int]:
        """Per-model measured GPU count (= gpus_per_replica)."""
        return {m.model_label: m.gpus_per_replica for m in self.models}

    @property
    def feasible_batch_sizes_union(self) -> list[int]:
        """Sorted union of all models' feasible batch sizes."""
        all_bs: set[int] = set()
        for m in self.models:
            all_bs.update(m.feasible_batch_sizes)
        return sorted(all_bs)
