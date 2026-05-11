"""Gymnasium environment wrapping the openg2g simulation for RL training."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from scenarios import (
    EXPERIMENTS,
    PVSystemSpec,
    ScenarioRecord,
    TimeVaryingLoadSpec,
    load_library_data,
    materialize_scenario,
)

from openg2g.controller.ofo import LogisticModelStore
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.base import DatacenterBackend, LLMDatacenterState
from openg2g.datacenter.command import DatacenterCommand, SetBatchSize
from openg2g.datacenter.config import InferenceModelSpec
from openg2g.grid.base import GridBackend
from openg2g.grid.command import GridCommand


@dataclass(frozen=True)
class ObservationConfig:
    """Fixed observation space configuration.

    The observation vector layout is:

    ```
    [0 .. M-1]              voltage vector (M bus-phase magnitudes, pu)
    ```

    Then, if `zone_summary` is set (per-zone voltage summary mode):

    ```
    [M + 3*z + 0]           zone z: worst undervoltage magnitude
    [M + 3*z + 1]           zone z: worst overvoltage magnitude
    [M + 3*z + 2]           zone z: fraction of bus-phases in violation
    [M + 3*n_zones + 5*i]   model i features (see below)
    ```

    Otherwise (global summary mode):

    ```
    [M]                     worst undervoltage magnitude
    [M+1]                   worst overvoltage magnitude
    [M+2]                   fraction of bus-phases in violation
    [M+3 + 5*i + 0]         model i features (see below)
    ```

    Per-model features (5 values each):

    ```
    + 0   normalized batch size (log2 scale) [0,1]
    + 1   ITL / deadline  [0,3]
    + 2   active_replicas / max_replicas  [0,1]
    + 3   total 3-phase power (MW)
    + 4   delta batch from prev step (log2 norm) [-1,1]
    ```

    Where M = `n_bus_phases`. Set M = 0 for summary-only observations. When
    `zone_buses` is set, M is the number of bus-phases within those buses
    (subset of the full grid), and violation summaries are computed over
    that zone only. When `zone_summary` is set, the 3 global violation
    scalars are replaced by 3 scalars per zone: use with M = 0 for a
    compact multi-zone observation. When `bus_phase_groups` is set, M must
    equal `2 * n_buses` and the first M slots contain [min_phase_voltage,
    max_phase_voltage] per bus instead of raw per-phase voltages; each
    entry is a tuple of indices into the full `v_vec` for that bus's phases.
    """

    model_labels: tuple[str, ...]
    feasible_batch_sizes: dict[str, tuple[int, ...]]
    itl_deadlines: dict[str, float]
    max_replicas: dict[str, int]
    n_bus_phases: int
    initial_batch_sizes: dict[str, int] | None = None
    zone_buses: tuple[str, ...] | None = None
    zone_summary: dict[str, tuple[str, ...]] | None = None
    bus_phase_groups: tuple[tuple[int, ...], ...] | None = None
    v_min: float = 0.95
    v_max: float = 1.05

    def get_initial_batch(self, label: str) -> int:
        """Return the initial batch size for *label*, falling back to midpoint of feasible sizes."""
        if self.initial_batch_sizes is not None and label in self.initial_batch_sizes:
            return self.initial_batch_sizes[label]
        fbs = self.feasible_batch_sizes[label]
        return fbs[len(fbs) // 2]

    @property
    def n_models(self) -> int:
        return len(self.model_labels)

    @property
    def n_zone_summary_slots(self) -> int:
        return len(self.zone_summary) * 3 if self.zone_summary else 0

    @property
    def obs_dim(self) -> int:
        if self.zone_summary:
            return self.n_bus_phases + self.n_zone_summary_slots + 5 * self.n_models
        return self.n_bus_phases + 3 + 5 * self.n_models

    @classmethod
    def from_model_specs(
        cls,
        specs: tuple[InferenceModelSpec, ...],
        replica_counts: dict[str, int],
        n_bus_phases: int,
        initial_batch_sizes: dict[str, int] | None = None,
        zone_buses: tuple[str, ...] | None = None,
        v_min: float = 0.95,
        v_max: float = 1.05,
    ) -> ObservationConfig:
        return cls(
            model_labels=tuple(s.model_label for s in specs),
            feasible_batch_sizes={s.model_label: tuple(s.feasible_batch_sizes) for s in specs},
            itl_deadlines={s.model_label: s.itl_deadline_s for s in specs},
            max_replicas={s.model_label: replica_counts.get(s.model_label, 1) for s in specs},
            n_bus_phases=n_bus_phases,
            initial_batch_sizes=initial_batch_sizes,
            zone_buses=zone_buses,
            v_min=v_min,
            v_max=v_max,
        )

    @classmethod
    def from_multi_site(
        cls,
        site_specs: dict[str, tuple[InferenceModelSpec, ...]],
        site_replica_counts: dict[str, dict[str, int]],
        n_bus_phases: int,
        initial_batch_sizes: dict[str, int] | None = None,
        zone_buses: tuple[str, ...] | None = None,
        zone_summary: dict[str, tuple[str, ...]] | None = None,
        bus_phase_groups: tuple[tuple[int, ...], ...] | None = None,
        v_min: float = 0.95,
        v_max: float = 1.05,
    ) -> ObservationConfig:
        """Build config combining models from ALL sites."""
        all_labels: list[str] = []
        all_feasible: dict[str, tuple[int, ...]] = {}
        all_deadlines: dict[str, float] = {}
        all_max_rep: dict[str, int] = {}
        for sid in site_specs:
            for spec in site_specs[sid]:
                label = spec.model_label
                all_labels.append(label)
                all_feasible[label] = tuple(spec.feasible_batch_sizes)
                all_deadlines[label] = spec.itl_deadline_s
                all_max_rep[label] = site_replica_counts.get(sid, {}).get(label, 1)
        return cls(
            model_labels=tuple(all_labels),
            feasible_batch_sizes=all_feasible,
            itl_deadlines=all_deadlines,
            max_replicas=all_max_rep,
            n_bus_phases=n_bus_phases,
            initial_batch_sizes=initial_batch_sizes,
            zone_buses=zone_buses,
            zone_summary=zone_summary,
            bus_phase_groups=bus_phase_groups,
            v_min=v_min,
            v_max=v_max,
        )


@dataclass(frozen=True)
class RewardConfig:
    """Reward function weights."""

    w_voltage: float = 1000.0
    w_throughput: float = 0.01
    w_latency: float = 10.0
    w_switch: float = 0.1
    v_min: float = 0.95
    v_max: float = 1.05
    reward_clip: float = 0.0
    """If > 0, clip per-step reward to [-reward_clip, +inf). Prevents
    catastrophic scenarios from dominating PPO updates. Recommended: 1.0."""
    switch_mode: str = "magnitude"
    """Switch penalty mode:
    - `"magnitude"`: `-w_switch * |log2(b_t) - log2(b_{t-1})|` (original).
    - `"binary"`: `-w_switch` per model whenever batch size changes.
    - `"cooldown"`: `-w_switch * exp(-steps_since_last_change / switch_cooldown_tau)`
      per model whenever batch size changes (recent changes are expensive).
    """
    w_safe: float = 0.0
    """Small positive reward for keeping voltages in range. Each step the agent
    receives `+w_safe * (fraction of bus-phases within [v_min, v_max])`.
    Default 0 (disabled). Recommended starting value: 0.01."""
    switch_cooldown_tau: float = 30.0
    """Time constant (in steps) for the cooldown switch penalty.
    Only used when `switch_mode="cooldown"`."""


def compute_zone_mask(v_index: list[tuple[str, int]], zone_buses: tuple[str, ...]) -> np.ndarray:
    """Boolean mask selecting bus-phases belonging to *zone_buses*."""
    bus_set = {b.lower() for b in zone_buses}
    return np.array([bus.lower() in bus_set for bus, _ph in v_index], dtype=bool)


def compute_bus_phase_groups(v_index: list[tuple[str, int]]) -> tuple[tuple[int, ...], ...]:
    """Group v_index positions by bus name.

    Returns one tuple of indices per unique bus, in the order buses first
    appear in *v_index*.  Used to compute per-bus min/max voltages.
    """
    groups: dict[str, list[int]] = {}
    for i, (bus, _ph) in enumerate(v_index):
        groups.setdefault(bus, []).append(i)
    return tuple(tuple(idx) for idx in groups.values())


def decode_action(
    action: np.ndarray,
    action_mode: str,
    model_labels: tuple[str, ...],
    feasible_batch_sizes: dict[str, tuple[int, ...]],
    prev_batch: dict[str, int],
    coupled_max_shift: int = 6,
) -> dict[str, int]:
    """Decode a raw action array into batch-size assignments.

    This is the single source of truth for action → batch-size mapping,
    shared between the training env and the inference controller.
    """

    def _apply_delta(label: str, delta: int) -> int:
        feasible = feasible_batch_sizes[label]
        prev_b = prev_batch.get(label, feasible[len(feasible) // 2])
        try:
            cur_idx = feasible.index(prev_b)
        except ValueError:
            cur_idx = len(feasible) // 2
        new_idx = max(0, min(len(feasible) - 1, cur_idx + delta))
        return feasible[new_idx]

    if action_mode == "coupled":
        delta = int(action) - coupled_max_shift
        return {label: _apply_delta(label, delta) for label in model_labels}

    if action_mode == "delta":
        return {label: _apply_delta(label, int(action[i]) - 1) for i, label in enumerate(model_labels)}

    raise ValueError(f"Unknown action_mode: {action_mode!r}")


def build_observation(
    grid: GridBackend,
    datacenter: DatacenterBackend[LLMDatacenterState] | list[DatacenterBackend[LLMDatacenterState]],
    obs_config: ObservationConfig,
    prev_batch: dict[str, int],
    zone_mask: np.ndarray | None = None,
    zone_masks: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build a fixed-size observation vector from grid and datacenter state.

    *datacenter* may be a single backend or a list (for shared multi-site).
    When a list, per-model features are gathered from all DCs in order.
    *zone_mask* filters the voltage vector to the agent's zone (single-zone mode).
    *zone_masks* provides per-zone boolean masks for per-zone summary obs
    (used when obs_config.zone_summary is set).
    """
    obs = np.zeros(obs_config.obs_dim, dtype=np.float32)
    M = obs_config.n_bus_phases

    # Voltage vector (optionally filtered by zone)
    v_vec_full = grid.voltages_vector()
    v_vec = v_vec_full[zone_mask] if zone_mask is not None else v_vec_full

    if obs_config.bus_phase_groups is not None:
        # Per-bus min/max: [min_phase, max_phase] for each bus
        for k, indices in enumerate(obs_config.bus_phase_groups):
            ph_v = v_vec_full[list(indices)]
            obs[2 * k] = float(np.min(ph_v))
            obs[2 * k + 1] = float(np.max(ph_v))
    elif M > 0:
        n = min(len(v_vec), M)
        obs[:n] = v_vec[:n].astype(np.float32)

    v_min_cfg, v_max_cfg = obs_config.v_min, obs_config.v_max

    if obs_config.zone_summary:
        # Per-zone violation summary: replaces the 3 global scalars
        base = M
        for zone_name, _zone_bus_list in obs_config.zone_summary.items():
            mask_z = (zone_masks or {}).get(zone_name)
            v_z = v_vec_full[mask_z] if mask_z is not None else v_vec_full
            under_z = np.maximum(v_min_cfg - v_z, 0.0)
            over_z = np.maximum(v_z - v_max_cfg, 0.0)
            n_z = len(v_z)
            obs[base + 0] = float(np.max(under_z)) if n_z > 0 else 0.0
            obs[base + 1] = float(np.max(over_z)) if n_z > 0 else 0.0
            obs[base + 2] = float(np.count_nonzero(under_z > 0) + np.count_nonzero(over_z > 0)) / max(n_z, 1)
            base += 3
    else:
        # Global violation summary
        under = np.maximum(v_min_cfg - v_vec, 0.0)
        over = np.maximum(v_vec - v_max_cfg, 0.0)
        n_total = len(v_vec)
        obs[M] = float(np.max(under)) if n_total > 0 else 0.0
        obs[M + 1] = float(np.max(over)) if n_total > 0 else 0.0
        obs[M + 2] = float(np.count_nonzero(under > 0) + np.count_nonzero(over > 0)) / max(n_total, 1)

    # Per-model features: gather DC states
    dcs = datacenter if isinstance(datacenter, list) else [datacenter]
    # Build a merged state dict from all DCs
    batch_by_model: dict[str, int] = {}
    itl_by_model: dict[str, float] = {}
    replicas_by_model: dict[str, int] = {}
    total_power_w = 0.0
    for dc in dcs:
        st = dc.state
        batch_by_model.update(st.batch_size_by_model)
        itl_by_model.update(st.observed_itl_s_by_model)
        replicas_by_model.update(st.active_replicas_by_model)
        total_power_w += st.power_w.a + st.power_w.b + st.power_w.c

    base_offset = M + (obs_config.n_zone_summary_slots if obs_config.zone_summary else 3)
    for i, label in enumerate(obs_config.model_labels):
        base = base_offset + 5 * i
        feasible = obs_config.feasible_batch_sizes[label]
        log2_min = math.log2(feasible[0])
        log2_max = math.log2(feasible[-1])
        log2_range = log2_max - log2_min if log2_max > log2_min else 1.0

        batch = batch_by_model.get(label, feasible[len(feasible) // 2])
        obs[base + 0] = (math.log2(max(batch, 1)) - log2_min) / log2_range

        itl = itl_by_model.get(label, float("nan"))
        deadline = obs_config.itl_deadlines[label]
        obs[base + 1] = float(np.clip(itl / deadline, 0.0, 3.0)) if not math.isnan(itl) else 0.0

        replicas = replicas_by_model.get(label, 0)
        max_rep = obs_config.max_replicas[label]
        obs[base + 2] = replicas / max(max_rep, 1)

        obs[base + 3] = total_power_w / 1e6  # MW

        prev_b = prev_batch.get(label, batch)
        if prev_b > 0 and batch > 0:
            delta = (math.log2(batch) - math.log2(prev_b)) / log2_range
            obs[base + 4] = float(np.clip(delta, -1.0, 1.0))

    return obs


def compute_reward(
    grid: GridBackend,
    datacenter: DatacenterBackend[LLMDatacenterState] | list[DatacenterBackend[LLMDatacenterState]],
    obs_config: ObservationConfig,
    reward_config: RewardConfig,
    prev_batch: dict[str, int],
    curr_batch: dict[str, int],
    logistic_models: LogisticModelStore | None = None,
    steps_since_change: dict[str, int] | None = None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Compute per-step scalar reward.

    Returns:
        `(total_reward, reward_components, voltage_stats)`.

        `reward_components` is a signed breakdown by source: keys
        `"voltage"`, `"throughput"`, `"latency"`, `"switch"`,
        `"safe"`. Penalties are negative, bonuses positive; the sum
        equals `total_reward`.

        `voltage_stats` reports per-step grid health: `"max_under"`
        (worst undervoltage magnitude in pu), `"max_over"` (worst
        overvoltage), and `"violation_frac"` (fraction of bus-phases
        currently in violation). All computed over the full grid.
    """
    # Voltage violation penalty (over ALL buses, not just zone)
    v_vec = grid.voltages_vector()
    v_min_cfg, v_max_cfg = reward_config.v_min, reward_config.v_max
    under = np.maximum(v_min_cfg - v_vec, 0.0)
    over = np.maximum(v_vec - v_max_cfg, 0.0)
    voltage_penalty = float(np.sum(under**2) + np.sum(over**2))
    voltage_term = -reward_config.w_voltage * voltage_penalty

    n_total = len(v_vec)
    n_violating = int(np.count_nonzero(under > 0) + np.count_nonzero(over > 0))
    safe_frac = (n_total - n_violating) / max(n_total, 1)
    safe_term = reward_config.w_safe * safe_frac

    voltage_stats = {
        "max_under": float(np.max(under)) if n_total > 0 else 0.0,
        "max_over": float(np.max(over)) if n_total > 0 else 0.0,
        "violation_frac": float(n_violating) / max(n_total, 1),
    }

    # Gather DC states
    dcs = datacenter if isinstance(datacenter, list) else [datacenter]
    itl_by_model: dict[str, float] = {}
    for dc in dcs:
        itl_by_model.update(dc.state.observed_itl_s_by_model)

    throughput_term = 0.0
    latency_term = 0.0
    switch_term = 0.0
    switch_mode = reward_config.switch_mode
    for label in obs_config.model_labels:
        feasible = obs_config.feasible_batch_sizes[label]
        batch = curr_batch.get(label, feasible[len(feasible) // 2])

        if logistic_models is not None:
            th_fit = logistic_models.throughput(label)
            th_max = th_fit.eval(feasible[-1])
            th_max = max(th_max, 1e-9)
            throughput_term += reward_config.w_throughput * th_fit.eval(batch) / th_max
        else:
            log2_max = math.log2(feasible[-1])
            throughput_term += reward_config.w_throughput * math.log2(max(batch, 1)) / log2_max

        itl = itl_by_model.get(label, float("nan"))
        deadline = obs_config.itl_deadlines[label]
        if not math.isnan(itl) and itl > deadline:
            latency_term -= reward_config.w_latency * (itl - deadline) / deadline

        prev_b = prev_batch.get(label, batch)
        if prev_b > 0 and batch > 0 and batch != prev_b:
            if switch_mode == "magnitude":
                switch_term -= reward_config.w_switch * abs(math.log2(batch) - math.log2(prev_b))
            elif switch_mode == "binary":
                switch_term -= reward_config.w_switch
            elif switch_mode == "cooldown":
                ssc = (steps_since_change or {}).get(label, 999)
                switch_term -= reward_config.w_switch * math.exp(-ssc / reward_config.switch_cooldown_tau)

    reward = voltage_term + throughput_term + latency_term + switch_term + safe_term
    components = {
        "voltage": voltage_term,
        "throughput": throughput_term,
        "latency": latency_term,
        "switch": switch_term,
        "safe": safe_term,
    }
    return reward, components, voltage_stats


SimComponents = tuple[
    dict[str, DatacenterBackend],  # datacenters keyed by site_id
    GridBackend,  # grid
    TapScheduleController | None,  # tap controller (optional)
]

MakeSimFn = Callable[..., SimComponents]


class ScenarioLibrary:
    """Pre-screened scenario bank for PPO training and evaluation.

    Loaded from the directory produced by `build_library.py`:

    - `metadata.json`: build-time config + per-scenario scalar fields.
    - `traces.npz`: per-scenario voltage penalty arrays
      (`ofo_<i>` / `baseline_<i>`).

    `materialize(rec)` replays `randomize_scenario(seed=rec.seed, ...)` to
    rebuild the full per-episode scenario configuration on demand. Replay is
    bit-identical to the build-time output because `randomize_scenario` is
    fully seeded: there is no on-disk cache of the resolved configs to keep
    in sync.
    """

    def __init__(
        self,
        path: str,
        *,
        training_trace=None,
    ) -> None:
        lib_dir = Path(path)
        self.scenarios, self.config = load_library_data(lib_dir)
        if not self.scenarios:
            raise ValueError(f"Scenario library at {path} is empty.")

        # Materialization base components, reconstructed from the experiment
        # factory + library config. randomize_scenario needs these to replay.
        base_exp = EXPERIMENTS[self.config["system"]](training_trace=training_trace)
        self._dc_sites_base: dict[str, Any] = base_exp["dc_sites"]
        self._pv_systems_base = [PVSystemSpec(**p) for p in self.config["pv_systems_base"]]
        self._tvl_base = [TimeVaryingLoadSpec(**t) for t in self.config["tvl_base"]]
        overlay_cfg = self.config.get("training_base")
        if overlay_cfg is not None and training_trace is not None:
            self._training_base: dict | None = {**overlay_cfg, "trace": training_trace}
        else:
            self._training_base = None

        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.scenarios)

    def sample(self) -> Any:
        """Return a random scenario record."""
        idx = int(self._rng.integers(0, len(self.scenarios)))
        return self.scenarios[idx]

    def materialize(self, rec: ScenarioRecord) -> dict:
        """Re-derive the full per-episode scenario dict for `rec`.

        Returns the same shape as `randomize_scenario` (keys `dc_sites`,
        `pv_systems`, `tvl`, `training_run`, `params`, ...).
        """
        return materialize_scenario(
            rec,
            dc_sites_base=self._dc_sites_base,
            pv_systems_base=self._pv_systems_base,
            tvl_base=self._tvl_base,
            training_base=self._training_base,
            randomize_kwargs=self.config["randomize_kwargs"],
        )


class BatchSizeEnv(gymnasium.Env):
    """Gymnasium environment for batch-size voltage regulation.

    Each `step(action)` advances the simulation by one control interval.
    For multi-DC setups, `agent_site_id` specifies which site the RL agent
    controls.  Other sites run with fixed batch sizes.

    When `scenario_library` is provided, each `reset()` samples a
    pre-screened scenario and replays it.  When `ofo_baseline=True` the
    voltage reward term becomes the per-step difference between PPO's
    voltage penalty and the OFO oracle's (stored in the library), so the
    agent is rewarded for *improving on OFO*.  When `truncate_episode=True`
    the episode fast-forwards through the initial quiet period (before the
    first baseline violation) and terminates after the last violation clears.
    """

    metadata = {"render_modes": []}  # noqa: RUF012  # gym Env.metadata override; not annotated to avoid ty's invalid-attribute-override

    def __init__(
        self,
        make_sim_fn: MakeSimFn,
        obs_config: ObservationConfig,
        agent_site_id: str = "_default",
        reward_config: RewardConfig | None = None,
        action_mode: str = "delta",
        logistic_models: LogisticModelStore | None = None,
        dt_ctrl: Fraction = Fraction(1),
        total_duration_s: int = 3600,
        scenario_library: ScenarioLibrary | None = None,
        ofo_baseline: bool = False,
        truncate_episode: bool = False,
    ) -> None:
        super().__init__()
        self._make_sim = make_sim_fn
        self._obs_config = obs_config
        self._agent_site_id = agent_site_id
        self._reward_config = reward_config or RewardConfig()
        self._logistic_models = logistic_models
        self._dt_ctrl = dt_ctrl
        self._total_duration_s = total_duration_s
        self._scenario_library = scenario_library
        self._ofo_baseline = ofo_baseline
        self._truncate_episode = truncate_episode

        self._action_mode = action_mode

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_config.obs_dim,), dtype=np.float32)
        n_feasible = min(len(obs_config.feasible_batch_sizes[m]) for m in obs_config.model_labels)
        n_models = len(obs_config.model_labels)
        self._coupled_max_shift = n_feasible - 1
        n_coupled = 2 * self._coupled_max_shift + 1

        if self._action_mode == "coupled":
            self.action_space = spaces.Discrete(n_coupled)
        elif self._action_mode == "delta":
            self.action_space = spaces.MultiDiscrete([3] * n_models)
        else:
            raise ValueError(f"Unknown action_mode: {self._action_mode!r}")

        self._datacenters: dict[str, DatacenterBackend] = {}
        self._coord: Coordinator | None = None
        self._prev_batch: dict[str, int] = {}
        self._steps_since_change: dict[str, int] = {}
        self._steps_done: int = 0
        self._max_steps: int = 0
        self._zone_mask: np.ndarray | None = None
        self._zone_masks: dict[str, np.ndarray] | None = None
        self._ofo_voltage_trace: np.ndarray | None = None
        self._sim_step_offset: int = 0  # base-tick offset of the first PPO control step

    def _action_to_batch_sizes(self, action: np.ndarray) -> dict[str, int]:
        return decode_action(
            action,
            self._action_mode,
            self._obs_config.model_labels,
            self._obs_config.feasible_batch_sizes,
            self._prev_batch,
            self._coupled_max_shift,
        )

    def _action_to_commands(self, action: np.ndarray) -> tuple[list[DatacenterCommand | GridCommand], dict[str, int]]:
        """Decode an action into the commands to dispatch this step.

        Returns `(commands, applied_batch_sizes)`: the `applied_batch_sizes`
        dict is what `compute_reward` and the steps-since-change tracking
        consume. Subclasses (e.g. `SharedBatchSizeEnv`) override to dispatch
        per-site commands when one policy controls multiple datacenters.
        """
        batch_sizes = self._action_to_batch_sizes(action)
        commands: list[DatacenterCommand | GridCommand] = [
            SetBatchSize(batch_size_by_model=batch_sizes, target=self._datacenters[self._agent_site_id])
        ]
        return commands, batch_sizes

    def _obs_target(self) -> DatacenterBackend | list[DatacenterBackend]:
        """Datacenter(s) feeding `build_observation` / `compute_reward`.

        Single-policy default: just the agent's DC. `SharedBatchSizeEnv`
        overrides to return all DCs jointly.
        """
        return self._datacenters[self._agent_site_id]

    def _advance_one_control_interval(self) -> None:
        coord = self._coord
        if coord is None:
            raise RuntimeError("Coordinator not started; call reset() first.")
        n_ticks = int(self._dt_ctrl / coord.clock.tick_s)
        for _ in range(n_ticks):
            coord.step()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        if self._coord is not None:
            self._coord.stop()
            self._coord = None

        self._ofo_voltage_trace = None
        self._sim_step_offset = 0
        t_control_start = 0
        t_control_end = self._total_duration_s

        if self._scenario_library is not None:
            scenario_record = self._scenario_library.sample()
            if self._ofo_baseline:
                self._ofo_voltage_trace = scenario_record.ofo_voltage_pen_per_step
            if self._truncate_episode:
                t_control_start = scenario_record.t_control_start
                t_control_end = scenario_record.t_control_end
            scenario_dict = self._scenario_library.materialize(scenario_record)
            datacenters, grid, tap_ctrl = self._make_sim(scenario_override=scenario_dict)
        else:
            datacenters, grid, tap_ctrl = self._make_sim()
        self._datacenters = datacenters

        controllers = [tap_ctrl] if tap_ctrl is not None else []
        self._coord = Coordinator(
            datacenters=list(datacenters.values()),
            grid=grid,
            controllers=controllers,
            total_duration_s=self._total_duration_s,
        )
        self._coord.reset()
        self._coord.start()

        zone_buses = self._obs_config.zone_buses
        self._zone_mask = compute_zone_mask(grid.v_index, zone_buses) if zone_buses is not None else None
        if self._obs_config.zone_summary:
            self._zone_masks = {
                zname: compute_zone_mask(grid.v_index, tuple(zbuses))
                for zname, zbuses in self._obs_config.zone_summary.items()
            }
        else:
            self._zone_masks = None

        # Fast-forward through the quiet pre-violation window with fixed initial
        # batch sizes; the sim still runs so grid/DC states evolve.
        self._sim_step_offset = t_control_start
        for _ in range(t_control_start):
            self._advance_one_control_interval()

        # One more interval so the first observation reflects t_control_start.
        # That consumes one tick of the control window, so the agent gets
        # `(t_control_end - t_control_start - 1)` steps to act on; without the
        # -1, the final step's sim time would be t_control_end and overshoot
        # the OFO trace (which covers [0, total_duration_s - 1]).
        self._advance_one_control_interval()
        self._max_steps = t_control_end - t_control_start - 1
        self._steps_done = 0

        self._prev_batch = {label: self._obs_config.get_initial_batch(label) for label in self._obs_config.model_labels}
        self._steps_since_change = {label: 999 for label in self._obs_config.model_labels}
        obs = build_observation(
            self._coord.grid, self._obs_target(), self._obs_config, self._prev_batch, self._zone_mask, self._zone_masks
        )
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        coord = self._coord
        if coord is None:
            raise RuntimeError("Coordinator not started; call reset() first.")

        commands, applied_batch = self._action_to_commands(action)
        coord.dispatch_commands(commands)
        self._advance_one_control_interval()
        self._steps_done += 1

        target = self._obs_target()
        reward, reward_components, voltage_stats = compute_reward(
            coord.grid,
            target,
            self._obs_config,
            self._reward_config,
            self._prev_batch,
            applied_batch,
            self._logistic_models,
            steps_since_change=self._steps_since_change,
        )

        for label in self._obs_config.model_labels:
            if applied_batch.get(label) != self._prev_batch.get(label):
                self._steps_since_change[label] = 0
            else:
                self._steps_since_change[label] = self._steps_since_change.get(label, 999) + 1

        # OFO-difference reward: add back the OFO oracle's voltage penalty at
        # this simulation timestep so the agent is rewarded for beating OFO.
        if self._ofo_voltage_trace is not None:
            sim_t = self._sim_step_offset + self._steps_done
            if sim_t >= len(self._ofo_voltage_trace):
                raise IndexError(
                    f"OFO trace lookup at sim_t={sim_t} but trace length is "
                    f"{len(self._ofo_voltage_trace)}; episode is running past the trace. "
                    "Check t_control_end vs. the scenario library's total_duration_s."
                )
            ofo_pen = float(self._ofo_voltage_trace[sim_t])
            ofo_voltage_baseline = self._reward_config.w_voltage * ofo_pen
            reward += ofo_voltage_baseline
            reward_components = dict(reward_components)
            reward_components["voltage"] += ofo_voltage_baseline
            reward_components["ofo_baseline"] = ofo_voltage_baseline

        if self._reward_config.reward_clip > 0:
            reward = max(reward, -self._reward_config.reward_clip)

        self._prev_batch = dict(applied_batch)
        obs = build_observation(
            coord.grid, target, self._obs_config, self._prev_batch, self._zone_mask, self._zone_masks
        )

        truncated = self._steps_done >= self._max_steps
        info = {"reward_components": reward_components, "voltage_stats": voltage_stats}
        return obs, reward, False, truncated, info

    def close(self) -> None:
        if self._coord is not None:
            self._coord.stop()
            self._coord = None
        super().close()


class SharedBatchSizeEnv(BatchSizeEnv):
    """Controls ALL datacenter sites jointly with a single policy.

    The observation includes per-model features from all sites; the action
    space covers all models across all sites. `site_model_mapping` maps each
    site_id to the list of model labels served at that site.
    """

    def __init__(
        self,
        make_sim_fn: MakeSimFn,
        obs_config: ObservationConfig,
        site_model_mapping: dict[str, list[str]],
        reward_config: RewardConfig | None = None,
        action_mode: str = "delta",
        logistic_models: LogisticModelStore | None = None,
        dt_ctrl: Fraction = Fraction(1),
        total_duration_s: int = 3600,
        scenario_library: ScenarioLibrary | None = None,
        ofo_baseline: bool = False,
        truncate_episode: bool = False,
    ) -> None:
        super().__init__(
            make_sim_fn=make_sim_fn,
            obs_config=obs_config,
            reward_config=reward_config,
            action_mode=action_mode,
            logistic_models=logistic_models,
            dt_ctrl=dt_ctrl,
            total_duration_s=total_duration_s,
            scenario_library=scenario_library,
            ofo_baseline=ofo_baseline,
            truncate_episode=truncate_episode,
        )
        self._site_model_mapping = site_model_mapping

    def _action_to_commands(self, action: np.ndarray) -> tuple[list[DatacenterCommand | GridCommand], dict[str, int]]:
        flat = self._action_to_batch_sizes(action)
        commands: list[DatacenterCommand | GridCommand] = []
        applied_batch: dict[str, int] = {}
        for sid, labels in self._site_model_mapping.items():
            if sid not in self._datacenters:
                continue
            site_batch = {label: flat[label] for label in labels if label in flat}
            if not site_batch:
                continue
            commands.append(SetBatchSize(batch_size_by_model=site_batch, target=self._datacenters[sid]))
            applied_batch.update(site_batch)
        return commands, applied_batch

    def _obs_target(self) -> list[DatacenterBackend]:
        return list(self._datacenters.values())
