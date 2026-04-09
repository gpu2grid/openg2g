"""Shared utility functions."""

from __future__ import annotations

import math


def split_integer_evenly(n: int, k: int) -> list[int]:
    """Split integer *n* into *k* non-negative integers whose sum is *n*,
    differing by at most 1.

    Example:

    ```python
    split_integer_evenly(10, 3)  # -> [4, 3, 3]
    split_integer_evenly(2, 5)   # -> [1, 1, 0, 0, 0]
    ```
    """
    q, r = divmod(int(n), int(k))
    return [q + (1 if i < r else 0) for i in range(k)]


def smooth_bump(t: float, t_center: float, half_width: float) -> float:
    """Smooth bump function: 1 at center, 0 outside half_width."""
    dt = abs(t - t_center)
    if dt >= half_width:
        return 0.0
    x = dt / half_width
    return (1 - x * x) ** 2


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
