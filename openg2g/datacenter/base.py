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

    @abstractmethod
    def reset(self) -> None:
        """Reset simulation state to initial conditions.

        Called by the coordinator before each `start()`. Must clear all
        simulation state: history, counters, RNG seeds, cached values.
        Configuration (dt_s, models, templates) is not affected.

        Abstract so every implementation explicitly enumerates its state.
        A forgotten field is a bug -- not clearing it silently corrupts
        the second run.
        """

    def start(self) -> None:
        """Acquire per-run resources (threads, solver circuits).

        Called after `reset()`, before the simulation loop. Override for
        backends that need resource acquisition (e.g., `OpenDSSGrid`
        compiles its DSS circuit here). No-op by default because most
        offline components have no resources to acquire.
        """

    def stop(self) -> None:
        """Release per-run resources. Simulation state is preserved.

        Called after the simulation loop in LIFO order. Override for
        backends that acquired resources in `start()`. No-op by default.
        """

    def bind_event_emitter(self, emitter: EventEmitter) -> None:
        """Attach a clock-bound emitter for backend-originated events."""


class LLMBatchSizeControlledDatacenter(DatacenterBackend[DCStateT]):
    """Datacenter that serves LLM inference and supports batch-size control.

    Marker layer between `DatacenterBackend` and concrete implementations.
    Controllers that issue `set_batch_size` commands or read
    `active_replicas_by_model` / `observed_itl_s_by_model` from state
    should bind their generic to this class.
    """
