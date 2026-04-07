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


@dataclass(frozen=True)
class ObservationConfig:
    """Fixed observation space configuration.

    The observation vector layout is::

        [0 .. M-1]              full voltage vector (M bus-phase magnitudes, pu)
        [M]                     worst undervoltage magnitude
        [M+1]                   worst overvoltage magnitude
        [M+2]                   fraction of bus-phases in violation
        [M+3 + 5*i + 0]        model i: normalized batch size (log2 scale) [0,1]
        [M+3 + 5*i + 1]        model i: ITL / deadline  [0,3]
        [M+3 + 5*i + 2]        model i: active_replicas / max_replicas  [0,1]
        [M+3 + 5*i + 3]        model i: total 3-phase power (MW)
        [M+3 + 5*i + 4]        model i: delta batch from prev step (log2 norm) [-1,1]

    Where M = ``n_bus_phases`` (number of monitored bus-phase pairs).
    """

    model_labels: tuple[str, ...]
    feasible_batch_sizes: dict[str, tuple[int, ...]]
    itl_deadlines: dict[str, float]
    max_replicas: dict[str, int]
    n_bus_phases: int
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
        v_min: float = 0.95,
        v_max: float = 1.05,
    ) -> ObservationConfig:
        return cls(
            model_labels=tuple(s.model_label for s in specs),
            feasible_batch_sizes={s.model_label: tuple(s.feasible_batch_sizes) for s in specs},
            itl_deadlines={s.model_label: s.itl_deadline_s for s in specs},
            max_replicas={s.model_label: replica_counts.get(s.model_label, 1) for s in specs},
            n_bus_phases=n_bus_phases,
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


def build_observation(
    grid: GridBackend,
    datacenter: DatacenterBackend[LLMDatacenterState],
    obs_config: ObservationConfig,
    prev_batch: dict[str, int],
) -> np.ndarray:
    """Build a fixed-size observation vector from grid and datacenter state."""
    obs = np.zeros(obs_config.obs_dim, dtype=np.float32)
    M = obs_config.n_bus_phases

    # Full voltage vector (when M > 0)
    v_vec = grid.voltages_vector()
    if M > 0:
        n = min(len(v_vec), M)
        obs[:n] = v_vec[:n].astype(np.float32)

    # Violation summary features (always computed over all bus-phases)
    v_min_cfg, v_max_cfg = obs_config.v_min, obs_config.v_max
    under = np.maximum(v_min_cfg - v_vec, 0.0)
    over = np.maximum(v_vec - v_max_cfg, 0.0)
    n_total = len(v_vec)
    obs[M] = float(np.max(under)) if n_total > 0 else 0.0
    obs[M + 1] = float(np.max(over)) if n_total > 0 else 0.0
    obs[M + 2] = float(np.count_nonzero(under > 0) + np.count_nonzero(over > 0)) / max(n_total, 1)

    # Per-model features
    dc_state = datacenter.state
    base_offset = M + 3
    for i, label in enumerate(obs_config.model_labels):
        base = base_offset + 5 * i
        feasible = obs_config.feasible_batch_sizes[label]
        log2_min = math.log2(feasible[0])
        log2_max = math.log2(feasible[-1])
        log2_range = log2_max - log2_min if log2_max > log2_min else 1.0

        # Normalized batch size
        batch = dc_state.batch_size_by_model.get(label, feasible[len(feasible) // 2])
        obs[base + 0] = (math.log2(max(batch, 1)) - log2_min) / log2_range

        # Normalized ITL
        itl = dc_state.observed_itl_s_by_model.get(label, float("nan"))
        deadline = obs_config.itl_deadlines[label]
        obs[base + 1] = float(np.clip(itl / deadline, 0.0, 3.0)) if not math.isnan(itl) else 0.0

        # Replica fraction
        replicas = dc_state.active_replicas_by_model.get(label, 0)
        max_rep = obs_config.max_replicas[label]
        obs[base + 2] = replicas / max(max_rep, 1)

        # Power (total three-phase MW)
        power = dc_state.power_w
        obs[base + 3] = (power.a + power.b + power.c) / 1e6

        # Delta batch from previous step
        prev_b = prev_batch.get(label, batch)
        if prev_b > 0 and batch > 0:
            delta = (math.log2(batch) - math.log2(prev_b)) / log2_range
            obs[base + 4] = float(np.clip(delta, -1.0, 1.0))

    return obs


def compute_reward(
    grid: GridBackend,
    datacenter: DatacenterBackend[LLMDatacenterState],
    obs_config: ObservationConfig,
    reward_config: RewardConfig,
    prev_batch: dict[str, int],
    curr_batch: dict[str, int],
    logistic_models: LogisticModelStore | None = None,
) -> float:
    """Compute per-step scalar reward.

    When *logistic_models* is provided, the throughput term uses the
    fitted ``throughput.eval(batch)`` curve (normalised by throughput at
    max batch).  Otherwise falls back to a ``log2`` proxy.
    """
    reward = 0.0

    # Voltage violation penalty (squared hinge over full voltage vector)
    v_vec = grid.voltages_vector()
    v_min_cfg, v_max_cfg = reward_config.v_min, reward_config.v_max
    under = np.maximum(v_min_cfg - v_vec, 0.0)
    over = np.maximum(v_vec - v_max_cfg, 0.0)
    voltage_penalty = float(np.sum(under**2) + np.sum(over**2))
    reward -= reward_config.w_voltage * voltage_penalty

    dc_state = datacenter.state
    for label in obs_config.model_labels:
        feasible = obs_config.feasible_batch_sizes[label]
        batch = curr_batch.get(label, feasible[len(feasible) // 2])

        # Throughput: use logistic fit when available
        if logistic_models is not None:
            th_fit = logistic_models.throughput(label)
            th_max = th_fit.eval(feasible[-1])
            th_max = max(th_max, 1e-9)
            reward += reward_config.w_throughput * th_fit.eval(batch) / th_max
        else:
            log2_max = math.log2(feasible[-1])
            reward += reward_config.w_throughput * math.log2(max(batch, 1)) / log2_max

        # Latency penalty (from observed ITL)
        itl = dc_state.observed_itl_s_by_model.get(label, float("nan"))
        deadline = obs_config.itl_deadlines[label]
        if not math.isnan(itl) and itl > deadline:
            reward -= reward_config.w_latency * (itl - deadline) / deadline

        # Switching cost
        prev_b = prev_batch.get(label, batch)
        if prev_b > 0 and batch > 0:
            reward -= reward_config.w_switch * abs(math.log2(batch) - math.log2(prev_b))

    return reward


# ── Simulation factory type ─────────────────────────────────────────────────

SimComponents = tuple[
    DatacenterBackend,  # datacenter
    GridBackend,  # grid
    TapScheduleController | None,  # tap controller (optional)
]

MakeSimFn = Callable[[], SimComponents]


# ── Gymnasium environment ───────────────────────────────────────────────────


class BatchSizeEnv(gymnasium.Env):
    """Gymnasium environment for batch-size voltage regulation.

    Each ``step(action)`` advances the simulation by one control interval
    (``dt_ctrl`` seconds) using the coordinator's multi-rate logic.
    """

    metadata: typing.ClassVar[dict[str, list]] = {"render_modes": []}

    def __init__(
        self,
        make_sim_fn: MakeSimFn,
        obs_config: ObservationConfig,
        reward_config: RewardConfig | None = None,
        logistic_models: LogisticModelStore | None = None,
        dt_ctrl: Fraction = Fraction(1),
        total_duration_s: int = 3600,
    ) -> None:
        super().__init__()
        self._make_sim = make_sim_fn
        self._obs_config = obs_config
        self._reward_config = reward_config or RewardConfig()
        self._logistic_models = logistic_models
        self._dt_ctrl = dt_ctrl
        self._total_duration_s = total_duration_s

        # Spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_config.obs_dim,), dtype=np.float32)
        action_dims = [len(obs_config.feasible_batch_sizes[m]) for m in obs_config.model_labels]
        self.action_space = spaces.MultiDiscrete(action_dims)

        # Will be initialized in reset()
        self._dc: DatacenterBackend | None = None
        self._grid: GridBackend | None = None
        self._tap_ctrl: TapScheduleController | None = None
        self._clock: SimulationClock | None = None
        self._log: SimulationLog | None = None
        self._dc_events: EventEmitter | None = None
        self._grid_events: EventEmitter | None = None
        self._ctrl_events: EventEmitter | None = None
        self._dc_buffer: list[ThreePhase] = []
        self._prev_batch: dict[str, int] = {}
        self._steps_done: int = 0
        self._max_steps: int = 0

    def _action_to_batch_sizes(self, action: np.ndarray) -> dict[str, int]:
        """Convert action indices to batch size dict."""
        return {
            label: self._obs_config.feasible_batch_sizes[label][int(action[i])]
            for i, label in enumerate(self._obs_config.model_labels)
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        # Build fresh simulation components
        dc, grid, tap_ctrl = self._make_sim()
        self._dc = dc
        self._grid = grid
        self._tap_ctrl = tap_ctrl

        # Compute tick as GCD of component periods
        periods = [dc.dt_s, grid.dt_s]
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
        self._dc_buffer = []

        # Max RL steps per episode
        self._max_steps = int(Fraction(self._total_duration_s) / self._dt_ctrl)
        self._steps_done = 0

        # Reset and start components
        dc.do_reset()
        grid.do_reset()
        if tap_ctrl is not None:
            tap_ctrl.reset()

        dc.start()
        grid.start()
        if tap_ctrl is not None:
            tap_ctrl.start()

        # Run initial ticks until first control point
        n_ticks_per_ctrl = int(self._dt_ctrl / tick)
        for _ in range(n_ticks_per_ctrl):
            if self._clock.is_due(dc.dt_s):
                dc_state = dc.do_step(self._clock, self._dc_events)
                self._dc_buffer.append(dc_state.power_w)
            if self._clock.is_due(grid.dt_s):
                grid.do_step(self._clock, {"_default": list(self._dc_buffer)}, self._grid_events)
                self._dc_buffer.clear()
            if tap_ctrl is not None and self._clock.is_due(tap_ctrl.dt_s):
                cmds = tap_ctrl.step(self._clock, dc, grid, self._ctrl_events)
                for cmd in cmds:
                    grid.apply_control(cmd, self._grid_events)
            self._clock.advance()

        # Initial batch sizes (mid-range)
        self._prev_batch = {label: fbs[len(fbs) // 2] for label, fbs in self._obs_config.feasible_batch_sizes.items()}

        obs = build_observation(grid, dc, self._obs_config, self._prev_batch)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._dc is not None and self._grid is not None and self._clock is not None

        dc = self._dc
        grid = self._grid
        tap_ctrl = self._tap_ctrl
        clock = self._clock

        # Convert action to batch sizes and apply
        batch_sizes = self._action_to_batch_sizes(action)
        cmd = SetBatchSize(batch_size_by_model=batch_sizes)
        dc.apply_control(cmd, self._dc_events)

        # Advance simulation by one control interval
        n_ticks_per_ctrl = int(self._dt_ctrl / clock.tick_s)
        for _ in range(n_ticks_per_ctrl):
            if clock.is_due(dc.dt_s):
                dc_state = dc.do_step(clock, self._dc_events)
                self._dc_buffer.append(dc_state.power_w)
            if clock.is_due(grid.dt_s):
                grid.do_step(clock, {"_default": list(self._dc_buffer)}, self._grid_events)
                self._dc_buffer.clear()
            if tap_ctrl is not None and clock.is_due(tap_ctrl.dt_s):
                cmds = tap_ctrl.step(clock, dc, grid, self._ctrl_events)
                for c in cmds:
                    grid.apply_control(c, self._grid_events)
            clock.advance()

        self._steps_done += 1

        # Compute reward and observation
        reward = compute_reward(
            grid, dc, self._obs_config, self._reward_config, self._prev_batch, batch_sizes, self._logistic_models
        )
        self._prev_batch = dict(batch_sizes)
        obs = build_observation(grid, dc, self._obs_config, self._prev_batch)

        truncated = self._steps_done >= self._max_steps
        terminated = False

        info: dict[str, Any] = {}
        if truncated:
            self._stop_components()

        return obs, reward, terminated, truncated, info

    def _stop_components(self) -> None:
        if self._tap_ctrl is not None:
            self._tap_ctrl.stop()
        if self._grid is not None:
            self._grid.stop()
        if self._dc is not None:
            self._dc.stop()

    def close(self) -> None:
        self._stop_components()
        super().close()


def _gcd_fraction(a: Fraction, b: Fraction) -> Fraction:
    """GCD of two Fractions."""
    from math import gcd

    num = gcd(a.numerator * b.denominator, b.numerator * a.denominator)
    den = a.denominator * b.denominator
    return Fraction(num, den)
