"""Abstract base class for datacenter backends and base state types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Generic, TypeVar

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.events import EventEmitter
from openg2g.types import DatacenterCommand, ThreePhase


@dataclass(frozen=True)
class DatacenterState:
    """State emitted by a datacenter backend each timestep.

    Contains only universally applicable fields. LLM-inference-specific
    fields (batch sizes, replicas, latency) live on `LLMDatacenterState`.
    """

    time_s: float
    power_w: ThreePhase


@dataclass(frozen=True)
class LLMDatacenterState(DatacenterState):
    """State from a datacenter serving LLM inference workloads.

    Extends `DatacenterState` with per-model batch size, replica count,
    and observed inter-token latency fields used by LLM controllers.
    """

    batch_size_by_model: dict[str, int] = field(default_factory=dict)
    active_replicas_by_model: dict[str, int] = field(default_factory=dict)
    observed_itl_s_by_model: dict[str, float] = field(default_factory=dict)


DCStateT = TypeVar("DCStateT", bound=DatacenterState)


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

    @property
    def phase_share_by_model(self) -> dict[str, np.ndarray]:
        """Per-model phase share vectors `[frac_A, frac_B, frac_C]`.

        Returns an empty dict by default. Consumers treat missing keys
        as uniform `[1/3, 1/3, 1/3]`. Override in subclasses that know
        actual server-to-phase placement.
        """
        return {}
