"""Abstract base class for datacenter backends and base state types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Generic, TypeVar, final

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
from openg2g.datacenter.command import DatacenterCommand
from openg2g.events import EventEmitter


@dataclass(frozen=True)
class DatacenterState:
    """State emitted by a datacenter backend each timestep.

    Contains only universally applicable fields. LLM-inference-specific
    fields (batch sizes, replicas, latency) live on child classes like
    [`LLMDatacenterState`][..LLMDatacenterState].

    Attributes:
        time_s: Simulation time in seconds.
        power_w: Three-phase power in watts.
    """

    time_s: float
    power_w: ThreePhase


@dataclass(frozen=True)
class LLMDatacenterState(DatacenterState):
    """State from a datacenter serving LLM workloads.

    Extends [`DatacenterState`][..DatacenterState] with per-model batch
    size, replica count, and observed inter-token latency fields used
    by LLM controllers.

    Attributes:
        batch_size_by_model: Current batch size per model label.
        active_replicas_by_model: Number of active replicas per model.
        observed_itl_s_by_model: Observed average inter-token latency
            (seconds) per model. `NaN` if unavailable.
    """

    batch_size_by_model: dict[str, int] = field(default_factory=dict)
    active_replicas_by_model: dict[str, int] = field(default_factory=dict)
    observed_itl_s_by_model: dict[str, float] = field(default_factory=dict)


DCStateT = TypeVar("DCStateT", bound=DatacenterState)


class DatacenterBackend(Generic[DCStateT], ABC):
    """Interface for datacenter power simulation backends."""

    _INIT_SENTINEL = object()

    def __init__(self) -> None:
        self._state: DCStateT | None = None
        self._history: list[DCStateT] = []
        self._dc_base_init = DatacenterBackend._INIT_SENTINEL

    def _check_base_init(self) -> None:
        if getattr(self, "_dc_base_init", None) is not DatacenterBackend._INIT_SENTINEL:
            raise TypeError(f"{type(self).__name__}.__init__ must call super().__init__() ")

    @property
    @abstractmethod
    def dt_s(self) -> Fraction:
        """Native timestep as a Fraction (seconds)."""

    @final
    @property
    def state(self) -> DCStateT:
        """Latest emitted state.

        Raises:
            RuntimeError: If accessed before the first `step()` call.
        """
        self._check_base_init()
        if self._state is None:
            raise RuntimeError(f"{type(self).__name__}.state accessed before first step().")
        return self._state

    @final
    def history(self, n: int | None = None) -> list[DCStateT]:
        """Return emitted state history (all, or latest `n`)."""
        self._check_base_init()
        if n is None:
            return list(self._history)
        if n <= 0:
            return []
        return list(self._history[-int(n) :])

    @final
    def do_step(self, clock: SimulationClock, events: EventEmitter) -> DCStateT:
        """Call `step`, record the state, and return it.

        Called by the coordinator. Subclasses should not override this.
        """
        self._check_base_init()
        state = self.step(clock, events)
        self._state = state
        self._history.append(state)
        return state

    @abstractmethod
    def step(self, clock: SimulationClock, events: EventEmitter) -> DCStateT:
        """Advance one native timestep. Return state for this step."""

    @abstractmethod
    def apply_control(self, command: DatacenterCommand, events: EventEmitter) -> None:
        """Apply one command. Takes effect on next step() call."""

    @final
    def do_reset(self) -> None:
        """Clear history and call `reset`.

        Called by the coordinator. Subclasses should not override this.
        """
        self._check_base_init()
        self._state = None
        self._history.clear()
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Reset simulation state to initial conditions.

        Called by the coordinator (via `do_reset`) before each
        [`start`][..start]. Must clear all simulation state: counters,
        RNG seeds, cached values. Configuration (dt_s, models,
        templates) is not affected. History is cleared automatically
        by `do_reset`.

        Abstract so every implementation explicitly enumerates its state.
        A forgotten field is a bug -- not clearing it silently corrupts
        the second run.
        """

    def start(self) -> None:
        """Acquire per-run resources (threads, solver circuits).

        Called after [`reset`][..reset], before the simulation loop.
        Override for backends that need resource acquisition (e.g.,
        [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid] compiles its
        DSS circuit here). No-op by default because most offline
        components have no resources to acquire.
        """

    def stop(self) -> None:
        """Release per-run resources. Simulation state is preserved.

        Called after the simulation loop in LIFO order. Override for
        backends that acquired resources in [`start`][..start]. No-op
        by default.
        """


class LLMBatchSizeControlledDatacenter(DatacenterBackend[DCStateT]):
    """Datacenter that serves LLM workloads and supports batch-size control.

    Marker layer between [`DatacenterBackend`][..DatacenterBackend] and
    concrete implementations. Controllers that issue
    [`SetBatchSize`][openg2g.datacenter.command.SetBatchSize] commands or read
    `active_replicas_by_model` / `observed_itl_s_by_model`
    from state should bind their generic to this class.
    """

    @property
    def phase_share_by_model(self) -> dict[str, np.ndarray]:
        """Per-model phase share vectors `[frac_A, frac_B, frac_C]`.

        Returns an empty dict by default. Consumers treat missing keys
        as uniform `[1/3, 1/3, 1/3]`. Override in subclasses that know
        actual server-to-phase placement.
        """
        return {}
