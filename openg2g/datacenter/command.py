"""Command types targeting datacenter backends."""

from __future__ import annotations

from dataclasses import dataclass, field


class DatacenterCommand:
    """Base for commands targeting the datacenter backend.

    Subclass this for each concrete datacenter command kind.
    The coordinator routes commands to backends based on this type hierarchy.
    """

    def __init__(self) -> None:
        if type(self) is DatacenterCommand:
            raise TypeError("DatacenterCommand cannot be instantiated directly; subclass it.")


@dataclass(frozen=True)
class SetBatchSize(DatacenterCommand):
    """Set batch sizes for one or more models.

    Attributes:
        batch_size_by_model: Mapping of model label to target batch size.
        ramp_up_rate_by_model: Per-model requests/second ramp-up rate.
            Models not present get immediate changes (rate 0).
    """

    batch_size_by_model: dict[str, int]
    ramp_up_rate_by_model: dict[str, float] = field(default_factory=dict)
