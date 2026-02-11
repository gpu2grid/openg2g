"""Shared utility functions."""

from __future__ import annotations

import numpy as np


def split_integer_evenly(n: int, k: int) -> list[int]:
    """Split integer *n* into *k* non-negative integers whose sum is *n*,
    differing by at most 1.

    Example::

        split_integer_evenly(10, 3) -> [4, 3, 3]
        split_integer_evenly(2, 5)  -> [1, 1, 0, 0, 0]
    """
    q, r = divmod(int(n), int(k))
    return [q + (1 if i < r else 0) for i in range(k)]


def resample_to_uniform_grid(
    t_src: np.ndarray,
    y_src: np.ndarray,
    t_dst: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate *(t_src, y_src)* onto the target grid *t_dst*."""
    t_src = np.asarray(t_src, float)
    y_src = np.asarray(y_src, float)
    t_dst = np.asarray(t_dst, float)
    if len(t_src) < 2:
        return np.full_like(t_dst, float(y_src[0] if len(y_src) else np.nan), dtype=float)
    return np.interp(t_dst, t_src, y_src, left=float(y_src[0]), right=float(y_src[-1]))
