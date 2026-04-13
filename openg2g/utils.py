"""Shared utility functions."""

from __future__ import annotations


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
