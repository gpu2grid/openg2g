"""External load types for grid attachment.

An [`ExternalLoad`][openg2g.grid.load.ExternalLoad] consumes real power (kW)
as a function of time. Loads are attached to grid buses via
[`OpenDSSGrid.attach_load`][openg2g.grid.opendss.OpenDSSGrid.attach_load].
The grid handles reactive power, bus voltage, and DSS element management.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from openg2g.utils import smooth_bump


class ExternalLoad(ABC):
    """Abstract time-varying load attached to a grid bus.

    Subclass this to define how real power consumption varies over time.
    The grid calls [`power_kw`][..power_kw] at each simulation timestep.
    """

    @abstractmethod
    def power_kw(self, t: float) -> float:
        """Return load consumption in kW at simulation time *t* (seconds)."""


class ConstantLoad(ExternalLoad):
    """Fixed power consumption at all times.

    Args:
        peak_kw: Constant consumption in kW.
    """

    def __init__(self, peak_kw: float) -> None:
        self._peak_kw = float(peak_kw)

    def power_kw(self, t: float) -> float:
        return self._peak_kw


class CSVProfileLoad(ExternalLoad):
    """Power consumption interpolated from a CSV time series.

    The CSV file must have two columns: time (seconds) and power (kW).
    The first row is treated as a header and skipped.

    Args:
        csv_path: Path to the CSV file.
    """

    def __init__(self, csv_path: Path) -> None:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        self._time = data[:, 0]
        self._power = data[:, 1]

    def power_kw(self, t: float) -> float:
        return float(np.interp(t, self._time, self._power))


class SyntheticLoad(ExternalLoad):
    """Synthetic load profile with bumps and fluctuation.

    Each `site_idx` produces a visually distinct curve with different
    diurnal patterns. These are synthetic profiles for demonstration.

    Args:
        peak_kw: Peak load consumption in kW.
        site_idx: Site index for distinct per-site profiles (0 to 4).
    """

    def __init__(self, peak_kw: float, site_idx: int = 0) -> None:
        self._peak_kw = float(peak_kw)
        self._site_idx = int(site_idx)

    def power_kw(self, t: float) -> float:
        idx = self._site_idx
        fluct_period = 130.0 + idx * 37
        fluct = 1.0 + 0.06 * math.sin(2 * math.pi * t / fluct_period + idx * 1.4)

        if idx == 0:
            base = 0.15 + 0.85 * smooth_bump(t, 2280, 1400)
            surge = 0.20 * smooth_bump(t, 2280, 180)
            return max(0.0, self._peak_kw * (base + surge) * fluct)
        elif idx == 1:
            base = 0.10
            base += 0.50 * smooth_bump(t, 1500, 600)
            base += 0.80 * smooth_bump(t, 2880, 500)
            return max(0.0, self._peak_kw * base * fluct)
        elif idx == 2:
            base = 0.80 - 0.55 * smooth_bump(t, 1800, 1200)
            surge = 0.70 * smooth_bump(t, 2520, 400)
            return max(0.0, self._peak_kw * (base + surge) * fluct)
        elif idx == 3:
            base = 0.10 + 0.90 * smooth_bump(t, 3120, 800)
            return max(0.0, self._peak_kw * base * fluct)
        elif idx == 4:
            base = 0.10
            base += 0.60 * smooth_bump(t, 1080, 300)
            base += 0.75 * smooth_bump(t, 2100, 350)
            base += 0.90 * smooth_bump(t, 3300, 300)
            return max(0.0, self._peak_kw * base * fluct)
        raise ValueError(f"Invalid site_idx {idx} for SyntheticLoad; must be 0 to 4.")
