"""Feature interfaces provided through the simulation context."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VoltageSnapshotFeature(Protocol):
    """Provides voltage vector snapshots in a stable bus-phase ordering."""

    @property
    def v_index(self) -> list[tuple[str, int]]:
        """Fixed (bus, phase) ordering used by ``voltages_vector``."""

    def voltages_vector(self) -> np.ndarray:
        """Return voltage magnitudes in ``v_index`` order."""


@runtime_checkable
class SensitivityFeature(Protocol):
    """Provides voltage sensitivity estimation."""

    def estimate_H(self, dp_kw: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
        """Estimate ``H = dv/dp`` and return ``(H, v0)``."""

