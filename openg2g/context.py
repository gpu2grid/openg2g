"""Shared simulation context passed to controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openg2g.capabilities import SensitivityFeature, VoltageSnapshotFeature


@dataclass(frozen=True)
class SimulationContext:
    """Runtime feature handles exposed by the coordinator."""

    voltage: VoltageSnapshotFeature | None = None
    sensitivity: SensitivityFeature | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def validate_required_features(
    *,
    controller_name: str,
    required: set[str],
    context: SimulationContext,
) -> None:
    """Validate that required context features are available."""
    missing: list[str] = []
    for key in sorted(required):
        if key == "voltage" and context.voltage is None:
            missing.append("voltage")
        elif key == "sensitivity" and context.sensitivity is None:
            missing.append("sensitivity")
        elif key not in {"voltage", "sensitivity"}:
            missing.append(key)

    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Controller {controller_name!r} requires missing feature(s): {names}. "
            "Check controller requirements and coordinator context wiring."
        )
