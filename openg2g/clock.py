"""Simulation clock with multi-rate support and optional live-mode wall-clock sync."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field


@dataclass
class SimulationClock:
    """Integer-tick clock that avoids floating-point drift.

    Components run at different rates (DC=0.1s, Grid=1.0s, Controller=1.0s or 60s).
    The coordinator computes `tick_s` as the GCD of all component periods.

    In live mode (`live=True`), the clock synchronizes with wall-clock time.
    If computation falls behind, a warning is issued.
    """

    tick_s: float
    live: bool = False
    _step: int = field(default=0, init=False, repr=False)
    _wall_t0: float | None = field(default=None, init=False, repr=False)

    @property
    def time_s(self) -> float:
        return self._step * self.tick_s

    @property
    def step(self) -> int:
        return self._step

    @property
    def step_index(self) -> int:
        """Global simulation tick index (alias for `step`)."""
        return self._step

    def advance(self) -> float:
        """Advance one tick. Returns new simulation time in seconds."""
        self._step += 1
        if self.live:
            if self._wall_t0 is None:
                self._wall_t0 = time.monotonic()
            expected_wall = self._wall_t0 + self.time_s
            now = time.monotonic()
            if now < expected_wall:
                time.sleep(expected_wall - now)
            elif now - expected_wall > self.tick_s:
                lag = now - expected_wall
                warnings.warn(
                    f"Clock lag: {lag:.3f}s behind wall time at sim t={self.time_s:.1f}s. "
                    f"Control loop cannot keep up with real-time.",
                    stacklevel=2,
                )
        return self.time_s

    def is_due(self, period_s: float) -> bool:
        """Check if an event with the given period should fire on this tick."""
        period_ticks = round(period_s / self.tick_s)
        if period_ticks <= 0:
            return True
        return self._step % period_ticks == 0
