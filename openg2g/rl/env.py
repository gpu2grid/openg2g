"""Gymnasium environment wrapping the openg2g simulation for RL training."""

from __future__ import annotations

import math
import typing
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
from openg2g.controller.ofo import LogisticModelStore
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import SimulationLog
from openg2g.datacenter.base import DatacenterBackend, LLMDatacenterState
from openg2g.datacenter.command import SetBatchSize
from openg2g.datacenter.config import InferenceModelSpec
from openg2g.events import EventEmitter
from openg2g.grid.base import GridBackend

# ── Observation & reward config ─────────────────────────────────────────────


@dataclass(frozen=True)
class ObservationConfig:
    """Fixed observation space configuration.

    The observation vector layout is::

        [0 .. M-1]              voltage vector (M bus-phase magnitudes, pu)
        [M]                     worst undervoltage magnitude
        [M+1]                   worst overvoltage magnitude
        [M+2]                   fraction of bus-phases in violation
        [M+3 + 5*i + 0]        model i: normalized batch size (log2 scale) [0,1]
        [M+3 + 5*i + 1]        model i: ITL / deadline  [0,3]
        [M+3 + 5*i + 2]        model i: active_replicas / max_replicas  [0,1]
        [M+3 + 5*i + 3]        model i: total 3-phase power (MW)
        [M+3 + 5*i + 4]        model i: delta batch from prev step (log2 norm) [-1,1]

    Where M = ``n_bus_phases``.  Set M = 0 for summary-only observations.
    When ``zone_buses`` is set, M is the number of bus-phases within those
    buses (subset of the full grid), and violation summaries are computed
    over that zone only.
    """

    model_labels: tuple[str, ...]
    feasible_batch_sizes: dict[str, tuple[int, ...]]
    itl_deadlines: dict[str, float]
    max_replicas: dict[str, int]
    n_bus_phases: int
    zone_buses: tuple[str, ...] | None = None
    v_min: float = 0.95
    v_max: float = 1.05

    @property
    def n_models(self) -> int:
        return len(self.model_labels)

    @property
    def obs_dim(self) -> int:
        return self.n_bus_phases + 3 + 5 * self.n_models

    @classmethod
    def from_model_specs(
        cls,
        specs: tuple[InferenceModelSpec, ...],
        replica_counts: dict[str, int],
        n_bus_phases: int,
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
        zone_buses: tuple[str, ...] | None = None,
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
            zone_buses=zone_buses,
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


# ── Zone voltage filtering ──────────────────────────────────────────────────


def compute_zone_mask(v_index: list[tuple[str, int]], zone_buses: tuple[str, ...]) -> np.ndarray:
    """Boolean mask selecting bus-phases belonging to *zone_buses*."""
    bus_set = {b.lower() for b in zone_buses}
    return np.array([bus.lower() in bus_set for bus, _ph in v_index], dtype=bool)


# ── Observation & reward builders ───────────────────────────────────────────


def build_observation(
    grid: GridBackend,
    datacenter: DatacenterBackend[LLMDatacenterState] | list[DatacenterBackend[LLMDatacenterState]],
    obs_config: ObservationConfig,
    prev_batch: dict[str, int],
    zone_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build a fixed-size observation vector from grid and datacenter state.

    *datacenter* may be a single backend or a list (for shared multi-site).
    When a list, per-model features are gathered from all DCs in order.
    *zone_mask* filters the voltage vector to the agent's zone.
    """
    obs = np.zeros(obs_config.obs_dim, dtype=np.float32)
    M = obs_config.n_bus_phases

    # Voltage vector (optionally filtered by zone)
    v_vec_full = grid.voltages_vector()
    v_vec = v_vec_full[zone_mask] if zone_mask is not None else v_vec_full

    if M > 0:
        n = min(len(v_vec), M)
        obs[:n] = v_vec[:n].astype(np.float32)

    # Violation summary (over zone voltages, or all if no zone)
    v_min_cfg, v_max_cfg = obs_config.v_min, obs_config.v_max
    under = np.maximum(v_min_cfg - v_vec, 0.0)
    over = np.maximum(v_vec - v_max_cfg, 0.0)
    n_total = len(v_vec)
    obs[M] = float(np.max(under)) if n_total > 0 else 0.0
    obs[M + 1] = float(np.max(over)) if n_total > 0 else 0.0
    obs[M + 2] = float(np.count_nonzero(under > 0) + np.count_nonzero(over > 0)) / max(n_total, 1)

    # Per-model features — gather DC states
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

    base_offset = M + 3
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
) -> float:
    """Compute per-step scalar reward."""
    reward = 0.0

    # Voltage violation penalty (over ALL buses, not just zone)
    v_vec = grid.voltages_vector()
    v_min_cfg, v_max_cfg = reward_config.v_min, reward_config.v_max
    under = np.maximum(v_min_cfg - v_vec, 0.0)
    over = np.maximum(v_vec - v_max_cfg, 0.0)
    voltage_penalty = float(np.sum(under**2) + np.sum(over**2))
    reward -= reward_config.w_voltage * voltage_penalty

    # Gather DC states
    dcs = datacenter if isinstance(datacenter, list) else [datacenter]
    itl_by_model: dict[str, float] = {}
    for dc in dcs:
        itl_by_model.update(dc.state.observed_itl_s_by_model)

    for label in obs_config.model_labels:
        feasible = obs_config.feasible_batch_sizes[label]
        batch = curr_batch.get(label, feasible[len(feasible) // 2])

        if logistic_models is not None:
            th_fit = logistic_models.throughput(label)
            th_max = th_fit.eval(feasible[-1])
            th_max = max(th_max, 1e-9)
            reward += reward_config.w_throughput * th_fit.eval(batch) / th_max
        else:
            log2_max = math.log2(feasible[-1])
            reward += reward_config.w_throughput * math.log2(max(batch, 1)) / log2_max

        itl = itl_by_model.get(label, float("nan"))
        deadline = obs_config.itl_deadlines[label]
        if not math.isnan(itl) and itl > deadline:
            reward -= reward_config.w_latency * (itl - deadline) / deadline

        prev_b = prev_batch.get(label, batch)
        if prev_b > 0 and batch > 0:
            reward -= reward_config.w_switch * abs(math.log2(batch) - math.log2(prev_b))

    return reward


# ── Simulation factory type ─────────────────────────────────────────────────

SimComponents = tuple[
    dict[str, DatacenterBackend],  # datacenters keyed by site_id
    GridBackend,  # grid
    TapScheduleController | None,  # tap controller (optional)
]

MakeSimFn = Callable[[], SimComponents]


# ── Per-site Gymnasium environment ──────────────────────────────────────────


class BatchSizeEnv(gymnasium.Env):
    """Gymnasium environment for batch-size voltage regulation.

    Each ``step(action)`` advances the simulation by one control interval.
    For multi-DC setups, ``agent_site_id`` specifies which site the RL agent
    controls.  Other sites run with fixed batch sizes.
    """

    metadata: typing.ClassVar[dict[str, list]] = {"render_modes": []}

    def __init__(
        self,
        make_sim_fn: MakeSimFn,
        obs_config: ObservationConfig,
        agent_site_id: str = "_default",
        reward_config: RewardConfig | None = None,
        logistic_models: LogisticModelStore | None = None,
        dt_ctrl: Fraction = Fraction(1),
        total_duration_s: int = 3600,
    ) -> None:
        super().__init__()
        self._make_sim = make_sim_fn
        self._obs_config = obs_config
        self._agent_site_id = agent_site_id
        self._reward_config = reward_config or RewardConfig()
        self._logistic_models = logistic_models
        self._dt_ctrl = dt_ctrl
        self._total_duration_s = total_duration_s

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_config.obs_dim,), dtype=np.float32)
        action_dims = [len(obs_config.feasible_batch_sizes[m]) for m in obs_config.model_labels]
        self.action_space = spaces.MultiDiscrete(action_dims)

        self._datacenters: dict[str, DatacenterBackend] = {}
        self._grid: GridBackend | None = None
        self._tap_ctrl: TapScheduleController | None = None
        self._clock: SimulationClock | None = None
        self._log: SimulationLog | None = None
        self._dc_events: EventEmitter | None = None
        self._grid_events: EventEmitter | None = None
        self._ctrl_events: EventEmitter | None = None
        self._dc_buffers: dict[str, list[ThreePhase]] = {}
        self._prev_batch: dict[str, int] = {}
        self._steps_done: int = 0
        self._max_steps: int = 0
        self._zone_mask: np.ndarray | None = None

    def _action_to_batch_sizes(self, action: np.ndarray) -> dict[str, int]:
        return {
            label: self._obs_config.feasible_batch_sizes[label][int(action[i])]
            for i, label in enumerate(self._obs_config.model_labels)
        }

    @property
    def _agent_dc(self) -> DatacenterBackend:
        return self._datacenters[self._agent_site_id]

    def _advance_ticks(self, n_ticks: int) -> None:
        """Advance all components by *n_ticks* base ticks."""
        datacenters = self._datacenters
        grid = self._grid
        tap_ctrl = self._tap_ctrl
        clock = self._clock
        assert grid is not None and clock is not None
        for _ in range(n_ticks):
            for sid, dc in datacenters.items():
                if clock.is_due(dc.dt_s):
                    dc_state = dc.do_step(clock, self._dc_events)
                    self._dc_buffers[sid].append(dc_state.power_w)
            if clock.is_due(grid.dt_s):
                power_arg = {sid: list(buf) for sid, buf in self._dc_buffers.items()}
                grid.do_step(clock, power_arg, self._grid_events)
                for buf in self._dc_buffers.values():
                    buf.clear()
            if tap_ctrl is not None and clock.is_due(tap_ctrl.dt_s):
                first_dc = next(iter(datacenters.values()))
                cmds = tap_ctrl.step(clock, first_dc, grid, self._ctrl_events)
                for cmd in cmds:
                    grid.apply_control(cmd, self._grid_events)
            clock.advance()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        datacenters, grid, tap_ctrl = self._make_sim()
        self._datacenters = datacenters
        self._grid = grid
        self._tap_ctrl = tap_ctrl

        all_dcs = list(datacenters.values())
        periods = [dc.dt_s for dc in all_dcs] + [grid.dt_s]
        if tap_ctrl is not None:
            periods.append(tap_ctrl.dt_s)
        periods.append(self._dt_ctrl)
        tick = periods[0]
        for p in periods[1:]:
            tick = _gcd_fraction(tick, p)

        self._clock = SimulationClock(tick_s=tick)
        self._log = SimulationLog()
        self._dc_events = EventEmitter(self._clock, self._log, "datacenter")
        self._grid_events = EventEmitter(self._clock, self._log, "grid")
        self._ctrl_events = EventEmitter(self._clock, self._log, "controller")
        self._dc_buffers = {sid: [] for sid in datacenters}
        self._max_steps = int(Fraction(self._total_duration_s) / self._dt_ctrl)
        self._steps_done = 0

        for dc in all_dcs:
            dc.do_reset()
        grid.do_reset()
        if tap_ctrl is not None:
            tap_ctrl.reset()
        for dc in all_dcs:
            dc.start()
        grid.start()
        if tap_ctrl is not None:
            tap_ctrl.start()

        # Compute zone mask if configured
        zone_buses = self._obs_config.zone_buses
        if zone_buses is not None:
            self._zone_mask = compute_zone_mask(grid.v_index, zone_buses)
        else:
            self._zone_mask = None

        # Run initial ticks
        n_ticks_per_ctrl = int(self._dt_ctrl / tick)
        self._advance_ticks(n_ticks_per_ctrl)

        self._prev_batch = {label: fbs[len(fbs) // 2] for label, fbs in self._obs_config.feasible_batch_sizes.items()}
        obs = build_observation(grid, self._agent_dc, self._obs_config, self._prev_batch, self._zone_mask)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._grid is not None and self._clock is not None

        batch_sizes = self._action_to_batch_sizes(action)
        cmd = SetBatchSize(batch_size_by_model=batch_sizes)
        self._agent_dc.apply_control(cmd, self._dc_events)

        n_ticks_per_ctrl = int(self._dt_ctrl / self._clock.tick_s)
        self._advance_ticks(n_ticks_per_ctrl)
        self._steps_done += 1

        reward = compute_reward(
            self._grid,
            self._agent_dc,
            self._obs_config,
            self._reward_config,
            self._prev_batch,
            batch_sizes,
            self._logistic_models,
        )
        self._prev_batch = dict(batch_sizes)
        obs = build_observation(self._grid, self._agent_dc, self._obs_config, self._prev_batch, self._zone_mask)

        truncated = self._steps_done >= self._max_steps
        if truncated:
            self._stop_components()
        return obs, reward, False, truncated, {}

    def _stop_components(self) -> None:
        if self._tap_ctrl is not None:
            self._tap_ctrl.stop()
        if self._grid is not None:
            self._grid.stop()
        for dc in self._datacenters.values():
            dc.stop()

    def close(self) -> None:
        self._stop_components()
        super().close()


# ── Shared multi-site Gymnasium environment ─────────────────────────────────


class SharedBatchSizeEnv(BatchSizeEnv):
    """Controls ALL datacenter sites jointly with a single policy.

    The observation includes per-model features from all sites.
    The action space covers all models across all sites.
    ``site_model_mapping`` maps site_id → list of model labels at that site.
    """

    def __init__(
        self,
        make_sim_fn: MakeSimFn,
        obs_config: ObservationConfig,
        site_model_mapping: dict[str, list[str]],
        reward_config: RewardConfig | None = None,
        logistic_models: LogisticModelStore | None = None,
        dt_ctrl: Fraction = Fraction(1),
        total_duration_s: int = 3600,
    ) -> None:
        # Use the first site as agent_site_id (unused for shared, but needed by parent)
        first_site = next(iter(site_model_mapping))
        super().__init__(
            make_sim_fn=make_sim_fn,
            obs_config=obs_config,
            agent_site_id=first_site,
            reward_config=reward_config,
            logistic_models=logistic_models,
            dt_ctrl=dt_ctrl,
            total_duration_s=total_duration_s,
        )
        self._site_model_mapping = site_model_mapping

    def _action_to_site_batch_sizes(self, action: np.ndarray) -> dict[str, dict[str, int]]:
        """Convert action indices to per-site batch size dicts."""
        flat = super()._action_to_batch_sizes(action)
        result: dict[str, dict[str, int]] = {}
        for sid, labels in self._site_model_mapping.items():
            result[sid] = {label: flat[label] for label in labels if label in flat}
        return result

    @property
    def _all_dcs_list(self) -> list[DatacenterBackend]:
        return list(self._datacenters.values())

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        # Call parent reset (sets up sim, runs initial ticks)
        obs, info = super().reset(seed=seed, options=options)
        # Rebuild obs using ALL DCs
        obs = build_observation(self._grid, self._all_dcs_list, self._obs_config, self._prev_batch, self._zone_mask)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._grid is not None and self._clock is not None

        # Apply batch sizes to each site's DC
        site_batches = self._action_to_site_batch_sizes(action)
        all_batch_sizes: dict[str, int] = {}
        for sid, batches in site_batches.items():
            if sid in self._datacenters:
                cmd = SetBatchSize(batch_size_by_model=batches)
                self._datacenters[sid].apply_control(cmd, self._dc_events)
                all_batch_sizes.update(batches)

        n_ticks_per_ctrl = int(self._dt_ctrl / self._clock.tick_s)
        self._advance_ticks(n_ticks_per_ctrl)
        self._steps_done += 1

        reward = compute_reward(
            self._grid,
            self._all_dcs_list,
            self._obs_config,
            self._reward_config,
            self._prev_batch,
            all_batch_sizes,
            self._logistic_models,
        )
        self._prev_batch = dict(all_batch_sizes)
        obs = build_observation(self._grid, self._all_dcs_list, self._obs_config, self._prev_batch, self._zone_mask)

        truncated = self._steps_done >= self._max_steps
        if truncated:
            self._stop_components()
        return obs, reward, False, truncated, {}


# ── Utilities ───────────────────────────────────────────────────────────────


def _gcd_fraction(a: Fraction, b: Fraction) -> Fraction:
    """GCD of two Fractions."""
    from math import gcd

    num = gcd(a.numerator * b.denominator, b.numerator * a.denominator)
    den = a.denominator * b.denominator
    return Fraction(num, den)
