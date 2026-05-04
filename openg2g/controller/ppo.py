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
from openg2g.rl.env import ObservationConfig, build_observation, compute_zone_mask, decode_action


def _load_sb3_model(model_path: str | Path):
    """Load an SB3 PPO model, handling .zip extension."""
    from stable_baselines3 import PPO

    resolved = Path(model_path).resolve()
    load_path = str(resolved.with_suffix("")) if resolved.suffix == ".zip" else str(resolved)
    return PPO.load(load_path)


def _load_vecnormalize(vecnormalize_path: str | Path, observation_space, action_space):
    """Load a saved VecNormalize wrapper, restoring its running obs/reward statistics.

    Returns a VecNormalize whose ``normalize_obs`` reproduces what the policy
    saw during training. We attach a 1-env DummyVecEnv whose obs/action spaces
    match the trained model so SB3's loader is happy.
    """
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class _DummyEnv(gym.Env):
        metadata = {"render_modes": []}  # noqa: RUF012

        def __init__(self, obs_space, act_space):
            super().__init__()
            self.observation_space = obs_space
            self.action_space = act_space
            self.render_mode = None

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

        def close(self):
            pass

    venv = DummyVecEnv([lambda: _DummyEnv(observation_space, action_space)])
    vn = VecNormalize.load(str(vecnormalize_path), venv)
    vn.training = False
    vn.norm_reward = False
    return vn


def _detect_action_mode(action_space, n_models: int) -> str:
    """Infer the action mode from a saved model's action space shape."""
    from gymnasium import spaces as gspaces

    if isinstance(action_space, gspaces.Discrete):
        return "coupled"
    if isinstance(action_space, gspaces.MultiDiscrete):
        nvec = tuple(int(d) for d in action_space.nvec)
        if all(d == 3 for d in nvec):
            return "delta"
    raise ValueError(
        f"Unrecognised action space {action_space!r} — expected Discrete (coupled) or MultiDiscrete([3]*N) (delta)"
    )


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
        vecnormalize_path: Optional path to a saved VecNormalize stats pickle
            (``*_vecnormalize.pkl``). If provided, observations are normalized
            with the saved running mean/var before being passed to the policy
            — this MUST match the wrapper used during training, otherwise the
            policy sees out-of-distribution input.
    """

    def __init__(
        self,
        inference_models: tuple[InferenceModelSpec, ...],
        *,
        datacenter: LLMBatchSizeControlledDatacenter[LLMDatacenterState],
        grid: OpenDSSGrid,
        model_path: str | Path,
        obs_config: ObservationConfig,
        dt_s: Fraction = Fraction(1),
        vecnormalize_path: str | Path | None = None,
    ) -> None:
        self._models = inference_models
        self._datacenter = datacenter
        self._grid = grid
        self._sb3_model = _load_sb3_model(model_path)
        self._vecnormalize = (
            _load_vecnormalize(vecnormalize_path, self._sb3_model.observation_space, self._sb3_model.action_space)
            if vecnormalize_path is not None
            else None
        )
        self._obs_config = obs_config
        self._dt_s = dt_s
        self._feasible = {s.model_label: tuple(s.feasible_batch_sizes) for s in inference_models}
        self._prev_batch: dict[str, int] = {}
        self._zone_mask: np.ndarray | None = None
        self._zone_mask_computed = False
        # Detect action mode from model's action space
        self._action_mode = _detect_action_mode(self._sb3_model.action_space, len(inference_models))
        n_feasible = min(len(f) for f in self._feasible.values())
        self._coupled_max_shift = n_feasible - 1
        self._init_prev_batch()

    def _init_prev_batch(self) -> None:
        self._prev_batch = {s.model_label: self._obs_config.get_initial_batch(s.model_label) for s in self._models}

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    def reset(self) -> None:
        self._init_prev_batch()
        self._zone_mask_computed = False

    def step(
        self,
        clock: SimulationClock,
        events: EventEmitter,
    ) -> list[DatacenterCommand | GridCommand]:
        datacenter = self._datacenter
        grid = self._grid
        # Lazily compute zone mask on first step (grid must be started)
        if not self._zone_mask_computed:
            if self._obs_config.zone_buses is not None:
                self._zone_mask = compute_zone_mask(grid.v_index, self._obs_config.zone_buses)
            self._zone_mask_computed = True

        obs = build_observation(grid, datacenter, self._obs_config, self._prev_batch, self._zone_mask)
        if self._vecnormalize is not None:
            obs = self._vecnormalize.normalize_obs(obs)
        action, _ = self._sb3_model.predict(obs, deterministic=True)

        batch_sizes = decode_action(
            action,
            self._action_mode,
            self._obs_config.model_labels,
            self._feasible,
            self._prev_batch,
            self._coupled_max_shift,
        )

        self._prev_batch = batch_sizes
        events.emit("controller.ppo.step", {"batch_size_by_model": batch_sizes})
        return [SetBatchSize(batch_size_by_model=batch_sizes, target=datacenter)]


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
        datacenter: LLMBatchSizeControlledDatacenter[LLMDatacenterState],
        grid: OpenDSSGrid,
        model_path: str | Path,
        obs_config: ObservationConfig,
        site_model_mapping: dict[str, list[str]],
        dt_s: Fraction = Fraction(1),
        vecnormalize_path: str | Path | None = None,
    ) -> None:
        self._datacenter = datacenter
        self._grid = grid
        self._sb3_model = _load_sb3_model(model_path)
        self._vecnormalize = (
            _load_vecnormalize(vecnormalize_path, self._sb3_model.observation_space, self._sb3_model.action_space)
            if vecnormalize_path is not None
            else None
        )
        self._obs_config = obs_config
        self._site_model_mapping = site_model_mapping
        self._dt_s = dt_s
        self._feasible = dict(obs_config.feasible_batch_sizes)
        self._prev_batch: dict[str, int] = {}
        self._zone_mask: np.ndarray | None = None
        self._zone_mask_computed = False
        # Detect action mode from saved model so policies trained with
        # Detect action mode so coupled vs delta decodes correctly.
        n_models_total = len(self._obs_config.model_labels)
        self._action_mode = _detect_action_mode(self._sb3_model.action_space, n_models_total)
        n_feasible = min(len(f) for f in self._feasible.values())
        self._coupled_max_shift = n_feasible - 1
        # Optional list of all datacenters (set by attach_datacenters() before
        # coord.run()). The Coordinator's step loop only passes ONE DC per
        # controller (resolved by _site_id), but a shared policy trained on
        # observations from ALL sites needs every DC's batch/itl/replicas/power
        # state. When set, step() uses this list and ignores the single DC the
        # coordinator hands in.
        self._all_datacenters: list = []
        self._init_prev_batch()

    def attach_datacenters(self, datacenters: list) -> None:
        """Register all datacenter backends so the shared policy sees the
        joint multi-site observation. Call once after the Coordinator is
        constructed and before ``coord.run()``."""
        self._all_datacenters = list(datacenters)

    def _init_prev_batch(self) -> None:
        self._prev_batch = {label: self._obs_config.get_initial_batch(label) for label in self._obs_config.model_labels}

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    def reset(self) -> None:
        self._init_prev_batch()
        self._zone_mask_computed = False

    def step(
        self,
        clock: SimulationClock,
        events: EventEmitter,
    ) -> list[DatacenterCommand | GridCommand]:
        datacenter = self._datacenter
        grid = self._grid
        if not self._zone_mask_computed:
            if self._obs_config.zone_buses is not None:
                self._zone_mask = compute_zone_mask(grid.v_index, self._obs_config.zone_buses)
            self._zone_mask_computed = True

        # Build obs from all DCs. The Coordinator's step loop hands controllers
        # only their own DC, but a shared policy needs the joint state.
        # attach_datacenters() must be called once before coord.run() to
        # register the full multi-site list; falls back to the single agent
        # DC for legacy single-site behaviour.
        dc_arg = self._all_datacenters if self._all_datacenters else datacenter
        obs = build_observation(grid, dc_arg, self._obs_config, self._prev_batch, self._zone_mask)
        if self._vecnormalize is not None:
            obs = self._vecnormalize.normalize_obs(obs)
        action, _ = self._sb3_model.predict(obs, deterministic=True)

        all_batch = decode_action(
            action,
            self._action_mode,
            self._obs_config.model_labels,
            self._feasible,
            self._prev_batch,
            self._coupled_max_shift,
        )

        self._prev_batch = all_batch
        events.emit("controller.ppo.step", {"batch_size_by_model": all_batch})

        # Emit one SetBatchSize per site. Master's command-routing requires a
        # target= DC object per command. SharedPPOBatchSizeController is a
        # multi-DC controller; routing each site's command to the right DC
        # needs the controller to track per-sid DC references (added below
        # only if/when the shared-PPO inference path is exercised). For now
        # this raises so callers see a clear migration boundary instead of
        # silently routing every site to the same DC.
        if len(self._site_model_mapping) > 1:
            raise NotImplementedError(
                "SharedPPOBatchSizeController multi-DC routing needs migration: "
                "the controller must accept a dict[str, DatacenterBackend] in "
                "__init__ to route per-site SetBatchSize commands. Single-site "
                "shared mode still works."
            )
        commands: list[DatacenterCommand | GridCommand] = []
        for _sid, labels in self._site_model_mapping.items():
            site_batch = {label: all_batch[label] for label in labels if label in all_batch}
            if site_batch:
                commands.append(SetBatchSize(batch_size_by_model=site_batch, target=datacenter))
        return commands
