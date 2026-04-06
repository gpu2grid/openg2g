"""Rule-based batch-size controller for voltage regulation.

A proportional controller that adjusts LLM batch sizes based on observed
voltage violations.  Unlike the OFO controller, it requires no sensitivity
matrix, no logistic curve fits, and no dual variables — making it a natural
"simple baseline" for comparison.

Algorithm (each control step):
    1. Read all bus-phase voltages from the grid.
    2. Find worst voltage violation magnitude.
    3. Compute a signed "pressure" signal:
       - positive  (undervoltage) → reduce batch (less power draw, less voltage drop)
       - negative  (overvoltage)  → increase batch (more power draw, more voltage drop)
       - zero                     → no action (all voltages within bounds)
    4. Adjust each model's batch size proportionally in log2-space.
    5. Snap to the nearest feasible batch size.
"""

from __future__ import annotations

import logging
import math
from fractions import Fraction

from pydantic import BaseModel

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter, LLMDatacenterState
from openg2g.datacenter.command import DatacenterCommand, SetBatchSize
from openg2g.datacenter.config import InferenceModelSpec
from openg2g.events import EventEmitter
from openg2g.grid.command import GridCommand
from openg2g.grid.opendss import OpenDSSGrid

logger = logging.getLogger(__name__)


class RuleBasedConfig(BaseModel):
    """Configuration for the rule-based batch-size controller."""

    step_size: float = 10.0
    """Proportional gain: log2(batch) change per pu of voltage violation.
    With feasible batches spaced ~1 log2 unit apart, a violation of 0.01 pu
    needs step_size ~10 to produce a 0.1 log2 shift, enough to eventually
    change the discrete batch level."""

    v_min: float = 0.95
    """Lower voltage limit (pu)."""

    v_max: float = 1.05
    """Upper voltage limit (pu)."""

    deadband: float = 0.001
    """Ignore violations smaller than this (pu).  Prevents chattering."""

    latency_guard: bool = True
    """If True, prevent batch size increase when ITL exceeds deadline."""


class RuleBasedBatchSizeController(
    Controller[LLMBatchSizeControlledDatacenter[LLMDatacenterState], OpenDSSGrid],
):
    """Proportional rule-based controller for LLM batch-size regulation.

    Reads grid voltages, computes a signed pressure signal from the worst
    violation, and adjusts batch sizes proportionally.  No model fits or
    sensitivity matrices required.
    """

    def __init__(
        self,
        inference_models: tuple[InferenceModelSpec, ...] | list[InferenceModelSpec],
        *,
        config: RuleBasedConfig,
        dt_s: Fraction = Fraction(1),
        site_id: str | None = None,
        exclude_buses: tuple[str, ...] = (),
        initial_batch_sizes: dict[str, int] | None = None,
    ) -> None:
        model_specs = list(inference_models)
        self._dt_s = dt_s
        self._site_id = site_id
        self._config = config
        self._models = model_specs
        self._exclude_lower = {b.lower() for b in exclude_buses}
        self._initial_batch_sizes = initial_batch_sizes or {}

        # Build per-model feasible batch list (sorted ascending)
        self._feasible: dict[str, list[int]] = {}
        self._itl_deadline: dict[str, float] = {}
        for ms in model_specs:
            self._feasible[ms.model_label] = sorted(ms.feasible_batch_sizes)
            self._itl_deadline[ms.model_label] = ms.itl_deadline_s

        # Continuous state: log2(batch) per model (for smooth proportional control)
        self._log2_batch: dict[str, float] = {
            ms.model_label: math.log2(self._initial_batch_sizes.get(ms.model_label, ms.feasible_batch_sizes[0]))
            for ms in model_specs
        }

        logger.info(
            "RuleBasedBatchSizeController: %d models, dt=%s s, step_size=%.2f, deadband=%.4f, v=[%.2f, %.2f]",
            len(model_specs),
            dt_s,
            config.step_size,
            config.deadband,
            config.v_min,
            config.v_max,
        )

    @property
    def site_id(self) -> str | None:
        return self._site_id

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    def reset(self) -> None:
        self._log2_batch = {
            ms.model_label: math.log2(self._initial_batch_sizes.get(ms.model_label, ms.feasible_batch_sizes[0]))
            for ms in self._models
        }

    def step(
        self,
        clock: SimulationClock,
        datacenter: LLMBatchSizeControlledDatacenter[LLMDatacenterState],
        grid: OpenDSSGrid,
        events: EventEmitter,
    ) -> list[DatacenterCommand | GridCommand]:
        if grid.state is None:
            return []

        cfg = self._config
        voltages = grid.state.voltages

        # ── 1. Find worst voltage violation ──
        worst_under = 0.0  # magnitude of worst undervoltage (positive)
        worst_over = 0.0  # magnitude of worst overvoltage (positive)

        for bus in voltages.buses():
            if bus.lower() in self._exclude_lower:
                continue
            pv = voltages[bus]
            for v in (pv.a, pv.b, pv.c):
                if math.isnan(v):
                    continue
                if v < cfg.v_min:
                    worst_under = max(worst_under, cfg.v_min - v)
                elif v > cfg.v_max:
                    worst_over = max(worst_over, v - cfg.v_max)

        # ── 2. Compute pressure signal ──
        # Positive pressure → reduce batch (undervoltage: DC draws too much power)
        # Negative pressure → increase batch (overvoltage: DC draws too little)
        if worst_under > cfg.deadband:
            pressure = worst_under
        elif worst_over > cfg.deadband:
            pressure = -worst_over
        else:
            pressure = 0.0

        if pressure == 0.0:
            return []

        # ── 3. Read latency state for guard ──
        dc_state = datacenter.state
        itl_by_model = dc_state.observed_itl_s_by_model if dc_state else {}

        # ── 4. Adjust batch sizes ──
        new_batches: dict[str, int] = {}
        changed = False

        for ms in self._models:
            label = ms.model_label
            log2_b = self._log2_batch[label]
            feasible = self._feasible[label]

            # Proportional adjustment in log2-space
            # pressure > 0 (undervoltage) → reduce batch → log2_b decreases
            delta = -cfg.step_size * pressure
            new_log2 = log2_b + delta

            # Clamp to feasible range
            new_log2 = max(math.log2(feasible[0]), min(math.log2(feasible[-1]), new_log2))

            # Latency guard: don't increase batch if ITL already exceeds deadline
            if cfg.latency_guard and delta > 0:
                itl = itl_by_model.get(label, 0.0)
                if not math.isnan(itl) and itl > self._itl_deadline[label]:
                    new_log2 = log2_b  # revert

            # Snap to nearest feasible batch size
            target = 2.0**new_log2
            best = min(feasible, key=lambda b: abs(b - target))

            # Keep continuous state for accumulation; only snap for the command
            self._log2_batch[label] = new_log2
            new_batches[label] = best

            if best != dc_state.batch_size_by_model.get(
                label, self._initial_batch_sizes.get(label, ms.feasible_batch_sizes[0])
            ):
                changed = True

        if not changed:
            return []

        events.emit(
            "rule_based.step",
            {"time_s": clock.time_s, "pressure": pressure, "batch": dict(new_batches)},
        )

        return [SetBatchSize(batch_size_by_model=new_batches)]
