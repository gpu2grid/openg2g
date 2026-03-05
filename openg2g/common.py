"""Cross-cutting types shared across component families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThreePhase:
    """Three-phase quantity. Access via `.a`, `.b`, `.c`.

    Attributes:
        a: Phase A value.
        b: Phase B value.
        c: Phase C value.
    """

    a: float
    b: float
    c: float
