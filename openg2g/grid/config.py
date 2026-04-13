"""Grid configuration and schedule types."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

_PHASE_KEYS = ("a", "b", "c")


@dataclass(frozen=True)
class TapPosition:
    """Regulator tap position as a mapping of regulator names to tap ratios.

    All regulators are stored in a single `regulators` dict.  For
    convenience, per-phase keyword arguments `a`, `b`, `c` are
    accepted and stored under those keys:

    ```python
    # These are equivalent:
    TapPosition(a=1.075, b=1.05, c=1.075)
    TapPosition(regulators={"a": 1.075, "b": 1.05, "c": 1.075})
    ```

    Named regulators for multi-bank systems:

    ```python
    TapPosition(regulators={"creg1a": 1.075, "creg1b": 1.05, "creg2a": 1.0})
    ```

    Attributes:
        regulators: Mapping of regulator name to tap ratio (pu).
    """

    regulators: dict[str, float] = field(default_factory=dict)

    def __init__(
        self,
        *,
        a: float | None = None,
        b: float | None = None,
        c: float | None = None,
        regulators: dict[str, float] | None = None,
    ) -> None:
        merged = dict(regulators) if regulators else {}
        for key, val in zip(_PHASE_KEYS, (a, b, c), strict=True):
            if val is not None:
                merged[key] = val
        if not merged:
            raise ValueError("TapPosition requires at least one regulator tap value.")
        object.__setattr__(self, "regulators", merged)

    @property
    def a(self) -> float | None:
        """Phase A tap ratio, or `None` if not set."""
        return self.regulators.get("a")

    @property
    def b(self) -> float | None:
        """Phase B tap ratio, or `None` if not set."""
        return self.regulators.get("b")

    @property
    def c(self) -> float | None:
        """Phase C tap ratio, or `None` if not set."""
        return self.regulators.get("c")

    def at(self, t: float) -> TapSchedule:
        """Schedule this position at time `t` seconds."""
        return TapSchedule(((t, self),))


class TapSchedule:
    """Ordered sequence of scheduled tap positions.

    Build using [`TapPosition.at`][..TapPosition.at] and the `|` operator:

    ```python
    TAP_STEP = 0.00625  # standard 5/8% tap step
    schedule = (
        TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP).at(t=0)
        | TapPosition(a=1.0 + 16 * TAP_STEP).at(t=25 * 60)
    )
    ```

    Raises:
        ValueError: If two entries share the same timestamp.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: tuple[tuple[float, TapPosition], ...]) -> None:
        self._entries = tuple(sorted(entries, key=lambda e: e[0]))
        times = [t for t, _ in self._entries]
        if len(times) != len(set(times)):
            seen: set[float] = set()
            dupes = sorted({t for t in times if t in seen or seen.add(t)})
            raise ValueError(f"TapSchedule has duplicate timestamps: {dupes}")

    def __or__(self, other: TapSchedule) -> TapSchedule:
        return TapSchedule(self._entries + other._entries)

    def __iter__(self) -> Iterator[tuple[float, TapPosition]]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __repr__(self) -> str:
        parts: list[str] = []
        for t, p in self._entries:
            fields = []
            if p.a is not None:
                fields.append(f"a={p.a}")
            if p.b is not None:
                fields.append(f"b={p.b}")
            if p.c is not None:
                fields.append(f"c={p.c}")
            # Show non-phase regulators
            for name, val in p.regulators.items():
                if name not in _PHASE_KEYS:
                    fields.append(f"{name}={val}")
            parts.append(f"TapPosition({', '.join(fields)}).at(t={t})")
        return " | ".join(parts)
