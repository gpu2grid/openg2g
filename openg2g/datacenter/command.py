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
        target_site_id: Site this command targets. The coordinator uses
            this to route the command to the correct datacenter.
    """

    batch_size_by_model: dict[str, int]
    ramp_up_rate_by_model: dict[str, float] = field(default_factory=dict)
    target_site_id: str | None = None


@dataclass(frozen=True)
class ShiftReplicas(DatacenterCommand):
    """Shift replicas for a model at this datacenter.

    Positive ``replica_delta`` adds replicas (receiving site);
    negative removes them (sending site).

    Attributes:
        model_label: Which model to shift.
        replica_delta: Number of replicas to add (>0) or remove (<0).
        target_site_id: Site this command targets.  The coordinator uses
            this to route the command to the correct datacenter.
    """

    model_label: str
    replica_delta: int
    target_site_id: str | None = None
