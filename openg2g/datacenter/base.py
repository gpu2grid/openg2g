"""Abstract base class for datacenter backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from fractions import Fraction
from typing import Generic

from openg2g.clock import SimulationClock
from openg2g.events import EventEmitter
from openg2g.types import DatacenterCommand, DCStateT


class DatacenterBackend(Generic[DCStateT], ABC):
    """Interface for datacenter power simulation backends."""

    @property
    @abstractmethod
    def dt_s(self) -> Fraction:
        """Native timestep as a Fraction (seconds)."""

    @property
    @abstractmethod
    def state(self) -> DCStateT:
        """Latest emitted state.

        Raises:
            RuntimeError: If accessed before the first `step()` call.
        """

    @abstractmethod
    def history(self, n: int | None = None) -> Sequence[DCStateT]:
        """Return emitted state history (all, or latest `n`)."""

    @abstractmethod
    def step(self, clock: SimulationClock) -> DCStateT:
        """Advance one native timestep. Return state for this step."""

    @abstractmethod
    def apply_control(self, command: DatacenterCommand) -> None:
        """Apply one command. Takes effect on next step() call."""

    def start(self) -> None:
        """Acquire resources before simulation. No-op by default."""

    def stop(self) -> None:
        """Release resources after simulation. No-op by default."""

    def bind_event_emitter(self, emitter: EventEmitter) -> None:
        """Attach a clock-bound emitter for backend-originated events."""


class LLMBatchSizeControlledDatacenter(DatacenterBackend[DCStateT]):
    """Datacenter that serves LLM inference and supports batch-size control.

    Marker layer between `DatacenterBackend` and concrete implementations.
    Controllers that issue `set_batch_size` commands or read
    `active_replicas_by_model` / `observed_itl_s_by_model` from state
    should bind their generic to this class.
    """
