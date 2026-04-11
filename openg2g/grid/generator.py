"""Generator (power source) types for grid attachment.

A [`Generator`][openg2g.grid.generator.Generator] produces real power (kW)
as a function of time. Generators are attached to grid buses via
[`OpenDSSGrid.attach_generator`][openg2g.grid.opendss.OpenDSSGrid.attach_generator].
The grid handles reactive power, bus voltage, and DSS element management.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from openg2g.utils import smooth_bump


def irregular_fluct(t: float, seed: float = 0.0) -> float:
    """Irregular fluctuation via superposition of incommensurate frequencies.

    Returns a value centred around 1.0 with ~+-15% variation.
    """
    s = seed
    f1 = 0.06 * math.sin(2 * math.pi * t / 173.0 + s)
    f2 = 0.05 * math.sin(2 * math.pi * t / 97.3 + s * 2.3)
    f3 = 0.04 * math.sin(2 * math.pi * t / 251.7 + s * 0.7)
    f4 = 0.03 * math.sin(2 * math.pi * t / 41.9 + s * 4.1)
    f5 = 0.02 * math.sin(2 * math.pi * t / 317.3 + s * 1.9)
    return 1.0 + f1 + f2 + f3 + f4 + f5


class Generator(ABC):
    """Abstract power source attached to a grid bus.

    Subclass this to define how real power output varies over time.
    The grid calls [`power_kw`][..power_kw] at each simulation timestep.
    """

    @abstractmethod
    def power_kw(self, t: float) -> float:
        """Return generator output in kW at simulation time *t* (seconds)."""


class ConstantGenerator(Generator):
    """Fixed power output at all times.

    Args:
        peak_kw: Constant output in kW.
    """

    def __init__(self, peak_kw: float) -> None:
        self._peak_kw = float(peak_kw)

    def power_kw(self, t: float) -> float:
        return self._peak_kw


class CSVProfileGenerator(Generator):
    """Power output interpolated from a CSV time series.

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


class SyntheticPV(Generator):
    """Synthetic PV profile with cloud dips, trends, and fluctuation.

    Each `site_idx` produces a visually distinct curve. These are
    synthetic "lorem ipsum" profiles for demonstration, not based on
    real irradiance data.

    Args:
        peak_kw: Peak PV output in kW.
        site_idx: Site index for distinct per-site profiles (0 to 2).
    """

    _T = 3600  # assumed total simulation duration for trend shaping

    def __init__(self, peak_kw: float, site_idx: int = 0) -> None:
        self._peak_kw = float(peak_kw)
        self._site_idx = int(site_idx)

    def power_kw(self, t: float) -> float:
        T = self._T
        idx = self._site_idx
        if idx == 0:
            trend = 0.85 - 0.30 * (t / T)
            cloud = 1.0
            cloud -= 0.55 * smooth_bump(t, 600, 120)
            cloud -= 0.40 * smooth_bump(t, 2100, 180)
            fluct = irregular_fluct(t, seed=0.3)
            return max(0.0, self._peak_kw * trend * max(cloud, 0.05) * fluct)
        elif idx == 1:
            ramp = 0.55 + 0.40 * smooth_bump(t, 1200, 900)
            cloud = 1.0
            cloud -= 0.60 * smooth_bump(t, 1680, 240)
            cloud -= 0.25 * smooth_bump(t, 2400, 150)
            fluct = irregular_fluct(t, seed=2.1)
            return max(0.0, self._peak_kw * ramp * max(cloud, 0.05) * fluct)
        elif idx == 2:
            ramp = 0.30 + 0.65 * min(1.0, t / 900.0)
            cloud = 1.0
            cloud -= 0.70 * smooth_bump(t, 2700, 300)
            cloud -= 0.30 * smooth_bump(t, 1200, 100)
            fluct = irregular_fluct(t, seed=2.0 + idx * 3.7)
            return max(0.0, self._peak_kw * ramp * max(cloud, 0.05) * fluct)
        raise ValueError(f"Unsupported site_idx {idx} for SyntheticPV; valid range is 0-2.")
