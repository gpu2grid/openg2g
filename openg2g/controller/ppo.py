"""PPO-trained batch-size controller for voltage regulation.

Loads a trained stable-baselines3 PPO model and uses it for deterministic
inference within the openg2g Controller interface.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter, LLMDatacenterState
from openg2g.datacenter.command import DatacenterCommand, SetBatchSize
from openg2g.datacenter.config import InferenceModelSpec
from openg2g.events import EventEmitter
from openg2g.grid.command import GridCommand
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.rl.env import ObservationConfig, build_observation


class PPOBatchSizeController(
    Controller[LLMBatchSizeControlledDatacenter[LLMDatacenterState], OpenDSSGrid],
):
    """Batch-size controller using a trained PPO policy.

    Loads a stable-baselines3 PPO checkpoint and calls
    ``model.predict(obs, deterministic=True)`` each control step.

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
        from stable_baselines3 import PPO

        self._models = inference_models
        resolved = Path(model_path).resolve()
        # PPO.load() auto-appends .zip, so strip it if present
        load_path = str(resolved.with_suffix("")) if resolved.suffix == ".zip" else str(resolved)
        self._sb3_model = PPO.load(load_path)
        self._obs_config = obs_config
        self._dt_s = dt_s
        self._site_id = site_id
        self._feasible = {s.model_label: tuple(s.feasible_batch_sizes) for s in inference_models}
        self._prev_batch: dict[str, int] = {}
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

    def step(
        self,
        clock: SimulationClock,
        datacenter: LLMBatchSizeControlledDatacenter[LLMDatacenterState],
        grid: OpenDSSGrid,
        events: EventEmitter,
    ) -> list[DatacenterCommand | GridCommand]:
        obs = build_observation(grid, datacenter, self._obs_config, self._prev_batch)
        action, _ = self._sb3_model.predict(obs, deterministic=True)

        batch_sizes: dict[str, int] = {}
        for i, label in enumerate(self._obs_config.model_labels):
            feasible = self._feasible[label]
            idx = int(action[i])
            idx = max(0, min(idx, len(feasible) - 1))
            batch_sizes[label] = feasible[idx]

        self._prev_batch = batch_sizes

        events.emit(
            "controller.ppo.step",
            {"batch_size_by_model": batch_sizes},
        )

        return [SetBatchSize(batch_size_by_model=batch_sizes, target_site_id=self._site_id)]
