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
    3. For undervoltage: shift load OUT of violated site → highest-voltage site.
       For overvoltage: shift load INTO violated site ← lowest-voltage site.
    4. Shift ``gpus_per_shift`` GPUs worth of replicas per time step.
    5. Repeat until violation resolves or no candidates remain.
    """

    def __init__(
        self,
        *,
        config: LoadShiftConfig,
        dt_s: Fraction,
        datacenters: dict[str, LLMBatchSizeControlledDatacenter],
        site_bus_map: dict[str, str],
        models_by_site: dict[str, list[str]],
        gpus_per_replica_by_model: dict[str, int],
        feasible_batch_sizes_by_model: dict[str, list[int]],
        v_min: float = 0.95,
        v_max: float = 1.05,
    ) -> None:
        self._config = config
        self._dt_s = dt_s
        self._datacenters = datacenters  # site_id -> datacenter
        self._site_bus_map = site_bus_map  # site_id -> bus name
        self._models_by_site = models_by_site  # site_id -> [model_labels]
        self._gpus_per_replica = gpus_per_replica_by_model
        self._feasible_bs = feasible_batch_sizes_by_model
        self._v_min = v_min
        self._v_max = v_max
        self._step_count = 0

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    @property
    def site_id(self) -> str | None:
        return None  # cross-site controller

    def step(
        self,
        clock: SimulationClock,
        datacenter: LLMBatchSizeControlledDatacenter,
        grid: OpenDSSGrid,
        events: EventEmitter,
    ) -> list[DatacenterCommand | GridCommand]:
        self._step_count += 1
        if not self._config.enabled:
            return []

        # Build bus -> min voltage mapping from grid
        voltages = grid.voltages_vector()
        v_index = grid.v_index  # list of (bus, phase)
        bus_voltages: dict[str, list[float]] = {}
        for (bus, _phase), v in zip(v_index, voltages, strict=False):
            bus_voltages.setdefault(bus.lower(), []).append(float(v))

        # Per-site min/max voltage
        site_vmin: dict[str, float] = {}
        site_vmax: dict[str, float] = {}
        for site_id, bus in self._site_bus_map.items():
            vs = bus_voltages.get(bus.lower(), [])
            if vs:
                site_vmin[site_id] = min(vs)
                site_vmax[site_id] = max(vs)

        commands: list[DatacenterCommand | GridCommand] = []

        # Check each site for violations
        for site_id in list(self._site_bus_map.keys()):
            vmin = site_vmin.get(site_id, 1.0)
            vmax = site_vmax.get(site_id, 1.0)

            is_undervoltage = vmin < self._v_min
            is_overvoltage = vmax > self._v_max

            if not is_undervoltage and not is_overvoltage:
                continue

            # Check if batch sizes are saturated at this site
            if not self._is_batch_saturated(site_id, datacenter, grid, is_undervoltage):
                continue

            # Find best destination and model to shift
            site_models = set(self._models_by_site.get(site_id, []))

            if is_undervoltage:
                # Shift load OUT: pick destination with highest voltage AND available capacity
                best_dest = None
                best_v = -1.0
                for other_id in self._site_bus_map:
                    if other_id == site_id:
                        continue
                    other_models = set(self._models_by_site.get(other_id, []))
                    shared = site_models & other_models
                    if not shared:
                        continue
                    dest_dc = self._datacenters.get(other_id)
                    if dest_dc is not None and dest_dc.available_gpu_capacity() < self._config.gpus_per_shift:
                        continue  # no room at this destination
                    ov = site_vmin.get(other_id, 0.0)
                    if ov > best_v:
                        best_v = ov
                        best_dest = other_id

                if best_dest is None:
                    continue

                shared_models = site_models & set(self._models_by_site.get(best_dest, []))
                model = self._pick_model(shared_models)
                if model is None:
                    continue

                replicas = max(1, self._config.gpus_per_shift // self._gpus_per_replica[model])
                commands.append(
                    ShiftReplicas(
                        model_label=model,
                        replica_delta=-replicas,
                        target_site_id=site_id,
                    )
                )
                commands.append(
                    ShiftReplicas(
                        model_label=model,
                        replica_delta=+replicas,
                        target_site_id=best_dest,
                    )
                )
                logger.info(
                    "LoadShift: undervoltage at %s (Vmin=%.4f), shift %s ×%d replicas -> %s (Vmin=%.4f, free=%d GPUs)",
                    site_id,
                    vmin,
                    model,
                    replicas,
                    best_dest,
                    best_v,
                    self._datacenters[best_dest].available_gpu_capacity() if best_dest in self._datacenters else -1,
                )

            elif is_overvoltage:
                # Shift load IN: check this site has capacity, pick source with lowest voltage
                site_dc = self._datacenters.get(site_id)
                if site_dc is not None and site_dc.available_gpu_capacity() < self._config.gpus_per_shift:
                    continue  # violated site is full, can't accept more load

                best_src = None
                best_v = 2.0
                for other_id in self._site_bus_map:
                    if other_id == site_id:
                        continue
                    other_models = set(self._models_by_site.get(other_id, []))
                    shared = site_models & other_models
                    if not shared:
                        continue
                    ov = site_vmax.get(other_id, 2.0)
                    if ov < best_v:
                        best_v = ov
                        best_src = other_id

                if best_src is None:
                    continue

                shared_models = site_models & set(self._models_by_site.get(best_src, []))
                model = self._pick_model(shared_models)
                if model is None:
                    continue

                replicas = max(1, self._config.gpus_per_shift // self._gpus_per_replica[model])
                commands.append(
                    ShiftReplicas(
                        model_label=model,
                        replica_delta=-replicas,
                        target_site_id=best_src,
                    )
                )
                commands.append(
                    ShiftReplicas(
                        model_label=model,
                        replica_delta=+replicas,
                        target_site_id=site_id,
                    )
                )
                logger.info(
                    "LoadShift: overvoltage at %s (Vmax=%.4f), shift %s ×%d replicas <- %s (Vmax=%.4f)",
                    site_id,
                    vmax,
                    model,
                    replicas,
                    best_src,
                    best_v,
                )

        return commands

    def _is_batch_saturated(
        self,
        site_id: str,
        datacenter: LLMBatchSizeControlledDatacenter,
        grid: OpenDSSGrid,
        is_undervoltage: bool,
    ) -> bool:
        """Check if all models at site have batch sizes at their limit.

        For undervoltage: saturated = all at minimum batch (can't reduce power further
        via OFO — batch is already at min so load can't be lowered).
        For overvoltage: saturated = all at maximum batch (can't increase power further).
        """
        dc = self._datacenters.get(site_id)
        if dc is None:
            return False
        state = dc.state
        if state is None:
            return False

        site_models = self._models_by_site.get(site_id, [])
        for model_label in site_models:
            current_bs = state.batch_size_by_model.get(model_label)
            feasible = self._feasible_bs.get(model_label, [])
            if not feasible or current_bs is None:
                continue
            if is_undervoltage:
                # Saturated = at minimum batch (OFO can't reduce power further)
                if current_bs > min(feasible):
                    return False  # Still room to reduce
            else:
                # Saturated = at maximum batch (OFO can't increase power further)
                if current_bs < max(feasible):
                    return False  # Still room to increase
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
