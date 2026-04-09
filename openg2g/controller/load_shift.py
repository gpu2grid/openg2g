"""Cross-site LLM load shifting controller.

Shifts replicas between datacenters when batch-size control (OFO) is
exhausted and voltage violations persist.  Runs after all per-site OFO
controllers in the coordinator loop.
"""

from __future__ import annotations

import logging
from fractions import Fraction

from pydantic import BaseModel

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter
from openg2g.datacenter.command import DatacenterCommand, ShiftReplicas
from openg2g.events import EventEmitter
from openg2g.grid.command import GridCommand
from openg2g.grid.opendss import OpenDSSGrid

logger = logging.getLogger(__name__)


class LoadShiftConfig(BaseModel):
    """Configuration for cross-site load shifting."""

    enabled: bool = False
    gpus_per_shift: int = 8
    headroom: float = 0.3
    """Fraction of extra server capacity to pre-allocate at each DC
    so incoming replicas have room (e.g. 0.3 = 30% headroom)."""


class LoadShiftController(Controller[LLMBatchSizeControlledDatacenter, OpenDSSGrid]):
    """Shift LLM replicas between datacenters to resolve voltage violations.

    Rules:
    1. Only shift models already running at both source and destination.
    2. Only act when batch sizes are saturated AND violation persists.
    3. For undervoltage: shift load OUT of violated site -> highest-voltage site.
       For overvoltage: shift load INTO violated site <- lowest-voltage site.
    4. Shift `gpus_per_shift` GPUs worth of replicas per time step.
    5. Repeat until violation resolves or no candidates remain.
    """

    def __init__(
        self,
        *,
        config: LoadShiftConfig,
        dt_s: Fraction,
        datacenters: list[LLMBatchSizeControlledDatacenter],
        dc_bus_map: dict[LLMBatchSizeControlledDatacenter, str],
        models_by_dc: dict[LLMBatchSizeControlledDatacenter, list[str]],
        gpus_per_replica_by_model: dict[str, int],
        feasible_batch_sizes_by_model: dict[str, list[int]],
        v_min: float = 0.95,
        v_max: float = 1.05,
    ) -> None:
        self._config = config
        self._dt_s = dt_s
        self._datacenters = list(datacenters)
        self._dc_bus_map = dict(dc_bus_map)
        self._models_by_dc = dict(models_by_dc)
        self._gpus_per_replica = gpus_per_replica_by_model
        self._feasible_bs = feasible_batch_sizes_by_model
        self._v_min = v_min
        self._v_max = v_max
        self._step_count = 0

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    @property
    def datacenters(self) -> list:
        return list(self._datacenters)

    def step(
        self,
        clock: SimulationClock,
        grid: OpenDSSGrid,
        events: EventEmitter,
    ) -> list[DatacenterCommand | GridCommand]:
        self._step_count += 1
        if not self._config.enabled:
            return []

        # Build bus -> min voltage mapping from grid
        voltages = grid.voltages_vector()
        v_index = grid.v_index
        bus_voltages: dict[str, list[float]] = {}
        for (bus, _phase), v in zip(v_index, voltages, strict=False):
            bus_voltages.setdefault(bus.lower(), []).append(float(v))

        # Per-DC min/max voltage
        dc_vmin: dict[LLMBatchSizeControlledDatacenter, float] = {}
        dc_vmax: dict[LLMBatchSizeControlledDatacenter, float] = {}
        for dc, bus in self._dc_bus_map.items():
            vs = bus_voltages.get(bus.lower(), [])
            if vs:
                dc_vmin[dc] = min(vs)
                dc_vmax[dc] = max(vs)

        commands: list[DatacenterCommand | GridCommand] = []

        for dc in self._datacenters:
            vmin = dc_vmin.get(dc, 1.0)
            vmax = dc_vmax.get(dc, 1.0)

            is_undervoltage = vmin < self._v_min
            is_overvoltage = vmax > self._v_max

            if not is_undervoltage and not is_overvoltage:
                continue

            if not self._is_batch_saturated(dc, is_undervoltage):
                continue

            dc_models = set(self._models_by_dc.get(dc, []))

            if is_undervoltage:
                best_dest = None
                best_v = -1.0
                for other_dc in self._datacenters:
                    if other_dc is dc:
                        continue
                    other_models = set(self._models_by_dc.get(other_dc, []))
                    shared = dc_models & other_models
                    if not shared:
                        continue
                    if other_dc.available_gpu_capacity() < self._config.gpus_per_shift:
                        continue
                    ov = dc_vmin.get(other_dc, 0.0)
                    if ov > best_v:
                        best_v = ov
                        best_dest = other_dc

                if best_dest is None:
                    continue

                shared_models = dc_models & set(self._models_by_dc.get(best_dest, []))
                model = self._pick_model(shared_models)
                if model is None:
                    continue

                replicas = max(1, self._config.gpus_per_shift // self._gpus_per_replica[model])
                commands.append(ShiftReplicas(model_label=model, replica_delta=-replicas, target=dc))
                commands.append(ShiftReplicas(model_label=model, replica_delta=+replicas, target=best_dest))
                logger.info(
                    "LoadShift: undervoltage at %s (Vmin=%.4f), shift %s x%d replicas -> %s (Vmin=%.4f, free=%d GPUs)",
                    dc.name,
                    vmin,
                    model,
                    replicas,
                    best_dest.name,
                    best_v,
                    best_dest.available_gpu_capacity(),
                )

            elif is_overvoltage:
                if dc.available_gpu_capacity() < self._config.gpus_per_shift:
                    continue

                best_src = None
                best_v = 2.0
                for other_dc in self._datacenters:
                    if other_dc is dc:
                        continue
                    other_models = set(self._models_by_dc.get(other_dc, []))
                    shared = dc_models & other_models
                    if not shared:
                        continue
                    ov = dc_vmax.get(other_dc, 2.0)
                    if ov < best_v:
                        best_v = ov
                        best_src = other_dc

                if best_src is None:
                    continue

                shared_models = dc_models & set(self._models_by_dc.get(best_src, []))
                model = self._pick_model(shared_models)
                if model is None:
                    continue

                replicas = max(1, self._config.gpus_per_shift // self._gpus_per_replica[model])
                commands.append(ShiftReplicas(model_label=model, replica_delta=-replicas, target=best_src))
                commands.append(ShiftReplicas(model_label=model, replica_delta=+replicas, target=dc))
                logger.info(
                    "LoadShift: overvoltage at %s (Vmax=%.4f), shift %s x%d replicas <- %s (Vmax=%.4f)",
                    dc.name,
                    vmax,
                    model,
                    replicas,
                    best_src.name,
                    best_v,
                )

        return commands

    def _is_batch_saturated(
        self,
        dc: LLMBatchSizeControlledDatacenter,
        is_undervoltage: bool,
    ) -> bool:
        """Check if all models at DC have batch sizes at their limit."""
        state = dc.state
        if state is None:
            return False

        dc_models = self._models_by_dc.get(dc, [])
        for model_label in dc_models:
            current_bs = state.batch_size_by_model.get(model_label)
            feasible = self._feasible_bs.get(model_label, [])
            if not feasible or current_bs is None:
                continue
            if is_undervoltage:
                if current_bs > min(feasible):
                    return False
            else:
                if current_bs < max(feasible):
                    return False
        return True

    def _pick_model(self, shared_models: set[str]) -> str | None:
        """Pick the model with the most GPUs per replica (largest power impact)."""
        if not shared_models:
            return None
        return max(shared_models, key=lambda m: self._gpus_per_replica.get(m, 1))

    def reset(self) -> None:
        self._step_count = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass
