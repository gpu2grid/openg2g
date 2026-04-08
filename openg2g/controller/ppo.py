"""PPO-trained batch-size controller for voltage regulation.

Loads a trained stable-baselines3 PPO model and uses it for deterministic
inference within the openg2g Controller interface.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter, LLMDatacenterState
from openg2g.datacenter.command import DatacenterCommand, SetBatchSize
from openg2g.datacenter.config import InferenceModelSpec
from openg2g.events import EventEmitter
from openg2g.grid.command import GridCommand
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.rl.env import ObservationConfig, build_observation, compute_zone_mask


def _load_sb3_model(model_path: str | Path):
    """Load an SB3 PPO model, handling .zip extension."""
    from stable_baselines3 import PPO

    resolved = Path(model_path).resolve()
    load_path = str(resolved.with_suffix("")) if resolved.suffix == ".zip" else str(resolved)
    return PPO.load(load_path)


class PPOBatchSizeController(
    Controller[LLMBatchSizeControlledDatacenter[LLMDatacenterState], OpenDSSGrid],
):
    """Batch-size controller using a trained PPO policy (single site).

    Args:
        inference_models: Model specifications served in the datacenter.
        model_path: Path to saved SB3 PPO model (.zip).
        obs_config: Observation space configuration.
        dt_s: Control interval (seconds).
        site_id: Site identifier for multi-datacenter setups.
    """

    def __init__(
        self,
        inference_models: tuple[InferenceModelSpec, ...],
        *,
        model_path: str | Path,
        obs_config: ObservationConfig,
        dt_s: Fraction = Fraction(1),
        site_id: str | None = None,
    ) -> None:
        self._models = inference_models
        self._sb3_model = _load_sb3_model(model_path)
        self._obs_config = obs_config
        self._dt_s = dt_s
        self._site_id = site_id
        self._feasible = {s.model_label: tuple(s.feasible_batch_sizes) for s in inference_models}
        self._prev_batch: dict[str, int] = {}
        self._zone_mask: np.ndarray | None = None
        self._zone_mask_computed = False
        self._init_prev_batch()

    def _init_prev_batch(self) -> None:
        self._prev_batch = {
            s.model_label: s.feasible_batch_sizes[len(s.feasible_batch_sizes) // 2] for s in self._models
        }

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    def reset(self) -> None:
        self._init_prev_batch()
        self._zone_mask_computed = False

    def step(
        self,
        clock: SimulationClock,
        datacenter: LLMBatchSizeControlledDatacenter[LLMDatacenterState],
        grid: OpenDSSGrid,
        events: EventEmitter,
    ) -> list[DatacenterCommand | GridCommand]:
        # Lazily compute zone mask on first step (grid must be started)
        if not self._zone_mask_computed:
            if self._obs_config.zone_buses is not None:
                self._zone_mask = compute_zone_mask(grid.v_index, self._obs_config.zone_buses)
            self._zone_mask_computed = True

        obs = build_observation(grid, datacenter, self._obs_config, self._prev_batch, self._zone_mask)
        action, _ = self._sb3_model.predict(obs, deterministic=True)

        batch_sizes: dict[str, int] = {}
        for i, label in enumerate(self._obs_config.model_labels):
            feasible = self._feasible[label]
            idx = int(action[i])
            idx = max(0, min(idx, len(feasible) - 1))
            batch_sizes[label] = feasible[idx]

        self._prev_batch = batch_sizes
        events.emit("controller.ppo.step", {"batch_size_by_model": batch_sizes})
        return [SetBatchSize(batch_size_by_model=batch_sizes, target_site_id=self._site_id)]


class SharedPPOBatchSizeController(
    Controller[LLMBatchSizeControlledDatacenter[LLMDatacenterState], OpenDSSGrid],
):
    """Shared PPO controller that outputs batch sizes for ALL sites jointly.

    Requires the coordinator to have all datacenter sites registered.
    Outputs one ``SetBatchSize`` command per site.

    Args:
        model_path: Path to saved SB3 PPO model (.zip).
        obs_config: Combined observation config (all models from all sites).
        site_model_mapping: Maps site_id → list of model labels at that site.
        dt_s: Control interval (seconds).
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        obs_config: ObservationConfig,
        site_model_mapping: dict[str, list[str]],
        dt_s: Fraction = Fraction(1),
    ) -> None:
        self._sb3_model = _load_sb3_model(model_path)
        self._obs_config = obs_config
        self._site_model_mapping = site_model_mapping
        self._dt_s = dt_s
        self._feasible = dict(obs_config.feasible_batch_sizes)
        self._prev_batch: dict[str, int] = {}
        self._zone_mask: np.ndarray | None = None
        self._zone_mask_computed = False
        self._init_prev_batch()

    def _init_prev_batch(self) -> None:
        self._prev_batch = {label: fbs[len(fbs) // 2] for label, fbs in self._feasible.items()}

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    def reset(self) -> None:
        self._init_prev_batch()
        self._zone_mask_computed = False

    def step(
        self,
        clock: SimulationClock,
        datacenter: LLMBatchSizeControlledDatacenter[LLMDatacenterState],
        grid: OpenDSSGrid,
        events: EventEmitter,
    ) -> list[DatacenterCommand | GridCommand]:
        if not self._zone_mask_computed:
            if self._obs_config.zone_buses is not None:
                self._zone_mask = compute_zone_mask(grid.v_index, self._obs_config.zone_buses)
            self._zone_mask_computed = True

        # Build obs from all DCs — but we only receive one DC from the coordinator.
        # Use it for the observation (coordinator passes the first DC).
        obs = build_observation(grid, datacenter, self._obs_config, self._prev_batch, self._zone_mask)
        action, _ = self._sb3_model.predict(obs, deterministic=True)

        # Map action to per-site batch sizes
        all_batch: dict[str, int] = {}
        for i, label in enumerate(self._obs_config.model_labels):
            feasible = self._feasible[label]
            idx = int(action[i])
            idx = max(0, min(idx, len(feasible) - 1))
            all_batch[label] = feasible[idx]

        self._prev_batch = all_batch
        events.emit("controller.ppo.step", {"batch_size_by_model": all_batch})

        # Emit one SetBatchSize per site
        commands: list[DatacenterCommand | GridCommand] = []
        for sid, labels in self._site_model_mapping.items():
            site_batch = {label: all_batch[label] for label in labels if label in all_batch}
            if site_batch:
                commands.append(SetBatchSize(batch_size_by_model=site_batch, target_site_id=sid))
        return commands
