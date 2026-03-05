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
