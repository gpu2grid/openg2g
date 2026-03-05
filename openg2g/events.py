"""Clock-aligned simulation event primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from openg2g.clock import SimulationClock
    from openg2g.coordinator import SimulationLog

EventSource = Literal["coordinator", "controller", "datacenter", "grid", "custom"]


@dataclass(frozen=True)
class SimEvent:
    """Structured simulation event with canonical clock metadata.

    Attributes:
        tick: Integer tick at which the event was emitted.
        t_s: Simulation time in seconds.
        source: Component family that emitted the event.
        topic: Dot-separated event topic string.
        data: Arbitrary key-value payload.
    """

    tick: int
    t_s: float
    source: EventSource
    topic: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventEmitter:
    """Source-bound event helper that stamps [`SimEvent`][..SimEvent]
    instances with clock metadata.

    Attributes:
        clock: Simulation clock for timestamping events.
        log: `SimulationLog` that receives emitted events.
        source: Component family label attached to all events.
    """

    clock: SimulationClock
    log: SimulationLog
    source: EventSource

    def emit(self, topic: str, data: dict[str, Any] | None = None) -> None:
        """Emit one event with current clock metadata."""
        t_s = float(self.clock.time_s)
        self.log.emit(
            SimEvent(
                tick=int(self.clock.step),
                t_s=t_s,
                source=self.source,
                topic=str(topic),
                data={} if data is None else dict(data),
            )
        )
