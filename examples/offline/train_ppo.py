"""Train a PPO controller for batch-size voltage regulation.

Trains one PPO model per datacenter site.  For multi-DC systems (ieee34,
ieee123), each site gets its own policy while other sites use fixed
mid-range batch sizes during that site's training.

Usage:
    python train_ppo.py --system ieee13 --total-timesteps 200000
    python train_ppo.py --system ieee13 --obs-mode system-summary-only   # summary-only obs
    python train_ppo.py --system ieee13 --hidden-dims 256 256 256 --n-envs 8
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.datacenter.config import (
    DatacenterConfig,
    InferenceModelSpec,
    ReplicaSchedule,
    TrainingRun,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.config import TapSchedule
from openg2g.rl.env import (
    BatchSizeEnv,
    ObservationConfig,
    RewardConfig,
    SharedBatchSizeEnv,
    compute_bus_phase_groups,
    compute_zone_mask,
    resolve_action_mode,
)

from systems import (
    DT_CTRL,
    DT_DC,
    DT_GRID,
    EXPERIMENTS,
    POWER_AUG,
    SPECS_CACHE_DIR,
    TRAINING_TRACE_PATH,
    V_MAX,
    V_MIN,
    DCSite,
    ScenarioOpenDSSGrid,
    materialize_scenario,
)

logger = logging.getLogger(__name__)

# ── Simulation factory ──────────────────────────────────────────────────────


def make_sim_factory(
    exp: dict,
    inference_data: InferenceData,
):
    """Return a callable that builds fresh simulation components.

    Returns ``(make_sim, all_site_specs, all_replica_counts)`` where
    ``make_sim()`` produces ``(dict[str, DatacenterBackend], grid, tap_ctrl)``.

    Library scenarios are replayed by reading the resolved fields stored on
    each ``ScenarioRecord``; the rng-replay path is gone. Old libraries must
    be upgraded via ``examples/offline/migrate_scenario_library.py``.
    """
    sys = exp["sys"]
    dc_sites: dict[str, DCSite] = exp["dc_sites"]
    pv_systems_base = exp.get("pv_systems", [])
    tvl_base = exp.get("time_varying_loads", [])
    training_base = exp.get("training_base")

    is_single_dc = len(dc_sites) == 1
    if is_single_dc:
        orig_sid = next(iter(dc_sites))
        dc_sites = {"_default": dc_sites[orig_sid]}

    all_site_specs: dict[str, tuple[InferenceModelSpec, ...]] = {}
    all_replica_counts: dict[str, dict[str, int]] = {}
    all_initial_batch_sizes: dict[str, dict[str, int]] = {}
    site_inference: dict[str, InferenceData] = {}
    for sid, site in dc_sites.items():
        specs = tuple(md.spec for md, _ in site.models)
        all_site_specs[sid] = specs
        all_replica_counts[sid] = {md.spec.model_label: sched.initial for md, sched in site.models}
        all_initial_batch_sizes[sid] = {md.spec.model_label: md.initial_batch_size for md, _ in site.models}
        site_inference[sid] = inference_data.filter_models(specs)

    _episode_counter = [0]

    def make_sim(scenario_override=None):
        _episode_counter[0] += 1

        if scenario_override is not None:
            sc = materialize_scenario(scenario_override, training_base=training_base)
            sites = sc["dc_sites"]
            # Library was built keyed by the experiment's DC site id (e.g. "default");
            # the grid expects "_default" for single-DC. Remap once here.
            if is_single_dc and "_default" not in sites:
                orig = next(iter(sites))
                sites = {"_default": sites[orig]}
            pv_systems = sc["pv_systems"]
            tvl = sc["tvl"]
            training = sc["training_run"]
        else:
            sites = dc_sites
            pv_systems = pv_systems_base
            tvl = tvl_base
            if training_base is not None:
                training = TrainingRun(
                    n_gpus=training_base["n_gpus"],
                    trace=training_base["trace"],
                    target_peak_W_per_gpu=training_base["target_peak_W_per_gpu"],
                ).at(t_start=training_base["t_start"], t_end=training_base["t_end"])
            else:
                training = None

        # Build all datacenters. Per-model schedules combine the (initial,
        # ramps) pair stored on site.models — replaces the old dict[str, int]
        # replica_counts + separate inference_ramps fields.
        datacenters: dict[str, OfflineDatacenter] = {}
        for sid, site in sites.items():
            dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=site.base_kw_per_phase)
            replica_schedules: dict[str, ReplicaSchedule] = {md.spec.model_label: sched for md, sched in site.models}
            initial_bs = {md.spec.model_label: md.initial_batch_size for md, _ in site.models}
            wl_kwargs: dict = {
                "inference_data": site_inference[sid],
                "replica_schedules": replica_schedules,
                "initial_batch_sizes": initial_bs,
            }
            if training is not None:
                wl_kwargs["training"] = training
            workload = OfflineWorkload(**wl_kwargs)
            datacenters[sid] = OfflineDatacenter(
                dc_config,
                workload,
                name=sid,
                dt_s=DT_DC,
                seed=site.seed,
                power_augmentation=POWER_AUG,
                total_gpu_capacity=site.total_gpu_capacity,
            )

        # Build grid then attach DCs (master's imperative pattern; old
        # dc_loads / dc_bus kwargs no longer accepted by OpenDSSGrid).
        dc_config_pf = DatacenterConfig(base_kw_per_phase=0).power_factor
        exclude = tuple(sys.get("exclude_buses", ()))
        grid = ScenarioOpenDSSGrid(
            pv_systems=pv_systems,
            time_varying_loads=tvl,
            source_pu=sys["source_pu"],
            dss_case_dir=sys["dss_case_dir"],
            dss_master_file=sys["dss_master_file"],
            dt_s=DT_GRID,
            initial_tap_position=sys["initial_taps"],
            exclude_buses=exclude,
        )
        for sid, dc in datacenters.items():
            site = sites[sid]
            grid.attach_dc(
                dc,
                bus=site.bus,
                connection_type=site.connection_type,
                power_factor=dc_config_pf,
            )

        tap_ctrl = TapScheduleController(schedule=TapSchedule(()), dt_s=DT_CTRL)
        return datacenters, grid, tap_ctrl

    return make_sim, all_site_specs, all_replica_counts, all_initial_batch_sizes


# ── Training metrics callback ───────────────────────────────────────────────


def _new_episode_acc() -> dict:
    return {
        "voltage": 0.0,
        "throughput": 0.0,
        "latency": 0.0,
        "switch": 0.0,
        "safe": 0.0,
        "max_under": 0.0,
        "max_over": 0.0,
        "viol_frac_sum": 0.0,
        "n_steps": 0,
    }


class TrainingMetricsCallback:
    """SB3 BaseCallback that aggregates per-episode reward components and voltage stats.

    Writes one CSV row per completed episode and mirrors the same metrics to
    the SB3 TensorBoard logger so they show up alongside built-in PPO metrics.
    Imported lazily inside ``main`` so SB3 isn't a hard import for tooling that
    only wants the experiment definitions.
    """

    def __new__(cls, csv_path: Path):
        # Late-bind to BaseCallback so this module is importable without SB3.
        from stable_baselines3.common.callbacks import BaseCallback

        class _Impl(BaseCallback):
            def __init__(self, csv_path: Path):
                super().__init__(verbose=0)
                self.csv_path = csv_path
                self._per_env: dict[int, dict] = {}
                self._ep_count = 0
                self._fp = None
                self._writer = None

            def _on_training_start(self) -> None:
                self._fp = open(self.csv_path, "w", buffering=1, newline="")  # noqa: SIM115
                self._writer = csv.writer(self._fp)
                self._writer.writerow(
                    [
                        "episode",
                        "timestep",
                        "ep_reward",
                        "ep_length",
                        "voltage",
                        "throughput",
                        "latency",
                        "switch",
                        "safe",
                        "max_undervoltage",
                        "max_overvoltage",
                        "mean_violation_frac",
                    ]
                )

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", []) or []
                for env_idx, info in enumerate(infos):
                    acc = self._per_env.setdefault(env_idx, _new_episode_acc())
                    rc = info.get("reward_components")
                    if rc is not None:
                        acc["voltage"] += rc.get("voltage", 0.0)
                        acc["throughput"] += rc.get("throughput", 0.0)
                        acc["latency"] += rc.get("latency", 0.0)
                        acc["switch"] += rc.get("switch", 0.0)
                        acc["safe"] += rc.get("safe", 0.0)
                    vs = info.get("voltage_stats")
                    if vs is not None:
                        if vs.get("max_under", 0.0) > acc["max_under"]:
                            acc["max_under"] = vs["max_under"]
                        if vs.get("max_over", 0.0) > acc["max_over"]:
                            acc["max_over"] = vs["max_over"]
                        acc["viol_frac_sum"] += vs.get("violation_frac", 0.0)
                    acc["n_steps"] += 1

                    # Monitor wrapper injects an "episode" key on done
                    ep = info.get("episode")
                    if ep is not None:
                        self._ep_count += 1
                        n = max(acc["n_steps"], 1)
                        row = [
                            self._ep_count,
                            self.num_timesteps,
                            float(ep["r"]),
                            int(ep["l"]),
                            acc["voltage"],
                            acc["throughput"],
                            acc["latency"],
                            acc["switch"],
                            acc["safe"],
                            acc["max_under"],
                            acc["max_over"],
                            acc["viol_frac_sum"] / n,
                        ]
                        self._writer.writerow(row)
                        # Mirror to TB
                        self.logger.record("custom/voltage_pen", acc["voltage"])
                        self.logger.record("custom/throughput_bonus", acc["throughput"])
                        self.logger.record("custom/latency_pen", acc["latency"])
                        self.logger.record("custom/switch_pen", acc["switch"])
                        self.logger.record("custom/safe_bonus", acc["safe"])
                        self.logger.record("custom/max_undervoltage", acc["max_under"])
                        self.logger.record("custom/max_overvoltage", acc["max_over"])
                        self.logger.record("custom/violation_frac", acc["viol_frac_sum"] / n)
                        self._per_env[env_idx] = _new_episode_acc()
                return True

            def _on_training_end(self) -> None:
                if self._fp is not None:
                    self._fp.close()
                    self._fp = None

        return _Impl(csv_path)


# ── Plotting ────────────────────────────────────────────────────────────────


def plot_training_progress(csv_path: Path, output_path: Path, label: str) -> Path | None:
    """Read the per-episode metrics CSV and emit a 2x2 PNG dashboard.

    Returns the output path on success, or ``None`` if the CSV is empty.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict] = []
    with open(csv_path, newline="") as fp:
        for r in csv.DictReader(fp):
            if any(v is None for v in r.values()):
                continue  # skip partial rows from interrupted buffered writes
            rows.append({k: float(v) if k not in ("episode", "ep_length") else int(float(v)) for k, v in r.items()})
    if not rows:
        return None

    eps = np.array([r["episode"] for r in rows])
    ep_reward = np.array([r["ep_reward"] for r in rows])
    voltage = np.array([r["voltage"] for r in rows])
    throughput = np.array([r["throughput"] for r in rows])
    latency = np.array([r["latency"] for r in rows])
    switch = np.array([r["switch"] for r in rows])
    max_under = np.array([r["max_undervoltage"] for r in rows])
    max_over = np.array([r["max_overvoltage"] for r in rows])
    viol_frac = np.array([r["mean_violation_frac"] for r in rows])

    def smooth(arr: np.ndarray, window: int) -> np.ndarray:
        if window <= 1 or len(arr) < 2:
            return arr
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="same")

    window = max(1, len(rows) // 20)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(eps, ep_reward, alpha=0.3, label="raw")
    ax.plot(eps, smooth(ep_reward, window), label=f"smooth (w={window})", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title(f"Learning curve — {label}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(eps, smooth(voltage, window), label="voltage", color="C3")
    ax.plot(eps, smooth(throughput, window), label="throughput", color="C2")
    ax.plot(eps, smooth(latency, window), label="latency", color="C1")
    ax.plot(eps, smooth(switch, window), label="switch", color="C0")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Component reward (per episode)")
    ax.set_title("Reward decomposition")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(eps, smooth(max_under, window), label="max undervoltage", color="C0")
    ax.plot(eps, smooth(max_over, window), label="max overvoltage", color="C3")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Worst per-step deviation (pu)")
    ax.set_title("Voltage violation magnitude")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(eps, smooth(viol_frac, window), color="C4")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean fraction of bus-phases violating")
    ax.set_title("Violation prevalence")
    ax.set_ylim(0, max(0.05, float(viol_frac.max()) * 1.1))
    ax.grid(alpha=0.3)

    fig.suptitle(f"PPO training progress — {label}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return output_path


# ── Main ────────────────────────────────────────────────────────────────────


@dataclass
class Args:
    system: str = "ieee13"
    """System name (ieee13, ieee34, ieee123)."""
    total_timesteps: int = 200_000
    """Total environment timesteps for training (per site). Counted across ALL parallel envs."""
    learning_rate: float = 1e-4
    """PPO learning rate (initial value if lr_schedule != 'constant')."""
    lr_schedule: str = "constant"
    """Learning rate schedule: 'constant' or 'linear' (decays to 0 over training)."""
    n_steps: int = 3600
    """Rollout length per environment (one full simulated hour)."""
    batch_size: int = 128
    """Minibatch size for PPO updates."""
    n_epochs: int = 10
    """Number of PPO epochs per update."""
    gamma: float = 0.99
    """Discount factor."""
    gae_lambda: float = 0.95
    """GAE lambda."""
    clip_range: float = 0.2
    """PPO clipping range."""
    ent_coef: float = 0.01
    """Entropy coefficient."""
    hidden_dims: tuple[int, ...] = (128, 128)
    """Hidden layer widths for the MLP policy/value network. Pass multiple values for a deeper net, e.g. --hidden-dims 256 256 256."""  # noqa: E501
    w_voltage: float = 1000.0
    """Reward weight for voltage violations."""
    w_throughput: float = 0.0
    """Reward weight for throughput. Default 0 to isolate the voltage-control objective."""
    w_latency: float = 0.0
    """Reward weight for latency violations. Default 0 to isolate the voltage-control objective."""
    w_switch: float = 0.01
    """Reward weight for switching cost (penalizes |log2(batch_t) - log2(batch_{t-1})| summed over models). Without this, randomized-scenario runs converge to a near-uniform action distribution and the deterministic eval policy ends up flipping batch sizes on every step. 0.01 is a gentle prior — much smaller than voltage_pen so it acts as a tie-breaker, not a co-equal objective."""  # noqa: E501
    w_safe: float = 0.0
    """Small positive reward for staying in the safe voltage range. Each step adds +w_safe * (fraction of bus-phases within [v_min, v_max]). Default 0 (disabled). Recommended: 0.01."""  # noqa: E501
    switch_mode: str = "magnitude"
    """Switch penalty mode: 'magnitude' (original log-ratio), 'binary' (fixed cost per change), or 'cooldown' (decaying cost, recent changes expensive)."""  # noqa: E501
    switch_cooldown_tau: float = 30.0
    """Time constant (steps) for cooldown switch penalty. Only used with --switch-mode cooldown."""
    action_mode: str = "delta"
    """Action space mode: 'delta' (per-model {-1,0,+1}, 3^N actions) or
    'coupled' (all models move by the same delta, 13 actions)."""
    reward_clip: float = 0.0
    """If > 0, clip per-step reward to [-reward_clip, +inf). Prevents catastrophic scenarios from dominating PPO updates. Recommended: 1.0 (affects ~4% of episodes, leaving normal training signal intact)."""  # noqa: E501
    vec_normalize: bool = True
    """Wrap the vec env with SB3 VecNormalize (running obs/reward normalization). Strongly recommended: voltage_pen variance across scenarios is huge and tanks value-function learning without it."""  # noqa: E501
    obs_mode: str = "full-voltage"
    """Voltage observation mode. Choices:
    - "full-voltage": all bus-phase raw voltages + per-system summary (3 global scalars).
    - "per-bus-summary": per-bus [min,max] phase voltage + per-zone-summary (if zones exist) or per-system summary.
    - "per-zone-summary": per-zone summary only (3 scalars/zone, no raw voltages). Requires zones.
    - "system-summary-only": 3 global scalars only (no raw voltages, no zone breakdown).
    """
    shared: bool = True
    """Train one shared PPO for all sites (instead of separate per-site)."""
    total_duration_s: int = 3600
    """Episode length in simulated seconds. Lower for fast smoke tests (e.g. 300 = 5 simulated minutes)."""
    n_envs: int = 1
    """Number of parallel rollout environments. >1 uses SubprocVecEnv (each subprocess builds its own OpenDSS instance to avoid global-state conflicts)."""  # noqa: E501
    tensorboard: bool = True
    """Write TensorBoard logs to <output_dir>/tb. View with `tensorboard --logdir <output_dir>/tb`."""
    plot: bool = True
    """Generate matplotlib training-progress plots after each model finishes."""
    output_dir: str = ""
    """Output directory (default: outputs/<system>/ppo)."""
    log_level: str = "INFO"
    """Logging verbosity."""
    scenario_library: str = ""
    """Path to a scenario library .pkl built by build_scenario_library.py. When set, episodes are sampled from this library."""  # noqa: E501
    ofo_baseline: bool = False
    """Subtract the OFO oracle's per-step voltage penalty from PPO's reward (requires --scenario-library). Disabling gives the raw voltage penalty as reward."""  # noqa: E501
    truncate_episode: bool = True
    """Fast-forward past the initial quiet period and terminate after the last violation (requires --scenario-library with t_control_start/end). Disabling uses full 3600s episodes."""  # noqa: E501
    seed: int = 42
    """Random seed."""
    init_from: str = ""
    """Path to a PPO checkpoint .zip to warm-start from (e.g. ppo_1152000_steps.zip). If a sibling ppo_vecnormalize_<steps>.pkl exists and --vec-normalize is set, its stats are loaded too. Hyperparameters stored in the checkpoint (lr, ent_coef, clip_range, …) are preserved; pass CLI flags only to change the env-side reward weights."""  # noqa: E501


def main() -> None:
    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("openg2g.coordinator").setLevel(logging.WARNING)
    logging.getLogger("openg2g.datacenter").setLevel(logging.WARNING)
    logging.getLogger("openg2g.grid").setLevel(logging.WARNING)

    if args.system not in EXPERIMENTS:
        logger.error("Unknown system: %s. Available: %s", args.system, list(EXPERIMENTS.keys()))
        sys.exit(1)

    # Load training trace from the canonical fixed path (matches master's
    # run_ofo.py pattern; the JSON-driven training_trace_params pipeline is gone).
    training_trace = TrainingTrace.ensure(TRAINING_TRACE_PATH)

    exp = EXPERIMENTS[args.system](training_trace)
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "outputs" / args.system / (args.output_dir or "ppo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all model specs across sites
    dc_sites: dict[str, DCSite] = exp["dc_sites"]
    all_specs: list[InferenceModelSpec] = []
    for site in dc_sites.values():
        all_specs.extend(md.spec for md, _ in site.models)
    all_specs_tuple = tuple({s.model_label: s for s in all_specs}.values())

    # Load data via the per-spec content-addressed cache under SPECS_CACHE_DIR.
    # InferenceData.ensure regenerates only specs whose manifest is missing.
    logger.info("Loading data for %s...", args.system)
    inference_data = InferenceData.ensure(
        SPECS_CACHE_DIR,
        all_specs_tuple,
        plot=False,
        dt_s=float(DT_DC),
    )

    from openg2g.controller.ofo import LogisticModelStore

    logistic_models = LogisticModelStore.ensure(
        SPECS_CACHE_DIR,
        all_specs_tuple,
        plot=False,
    )

    scenario_lib = None
    if args.scenario_library:
        from openg2g.rl.env import ScenarioLibrary

        scenario_lib = ScenarioLibrary(args.scenario_library)
        logger.info(
            "Loaded scenario library with %d scenarios from %s (ofo_baseline=%s, truncate=%s)",
            len(scenario_lib),
            args.scenario_library,
            args.ofo_baseline,
            args.truncate_episode,
        )

    make_sim, all_site_specs, all_replica_counts, all_initial_batch_sizes = make_sim_factory(
        exp,
        inference_data,
    )

    # Probe grid for v_index and n_bus_phases
    probe_dcs, probe_grid, _ = make_sim()
    for dc in probe_dcs.values():
        dc.do_reset()
        dc.start()
    probe_grid.do_reset()
    probe_grid.start()
    v_index = probe_grid.v_index
    n_bus_phases_full = len(v_index)
    probe_grid.stop()
    for dc in probe_dcs.values():
        dc.stop()

    _VALID_OBS_MODES = {"full-voltage", "per-bus-summary", "per-zone-summary", "system-summary-only"}
    if args.obs_mode not in _VALID_OBS_MODES:
        raise ValueError(f"--obs-mode must be one of {sorted(_VALID_OBS_MODES)}, got {args.obs_mode!r}")

    if args.action_mode not in ("delta", "coupled"):
        raise ValueError(f"--action-mode must be 'delta' or 'coupled', got {args.action_mode!r}")

    # Zone info needed early for per-zone-summary validation
    zones: dict[str, list[str]] | None = exp.get("sys", {}).get("zones")

    if args.obs_mode == "per-zone-summary" and zones is None:
        raise ValueError("--obs-mode per-zone-summary requires the system to have zones defined (e.g. ieee123)")

    if args.obs_mode == "full-voltage":
        n_bus_phases = n_bus_phases_full
        bus_phase_groups = None
    elif args.obs_mode == "per-bus-summary":
        bus_phase_groups = compute_bus_phase_groups(v_index)
        n_bus_phases = 2 * len(bus_phase_groups)
    else:  # per-zone-summary or system-summary-only
        n_bus_phases = 0
        bus_phase_groups = None

    logger.info(
        "Grid has %d bus-phase pairs across %d buses; obs_mode=%s, n_bus_phases=%d",
        n_bus_phases_full,
        len(set(b for b, _ in v_index)),
        args.obs_mode,
        n_bus_phases,
    )

    reward_config = RewardConfig(
        w_voltage=args.w_voltage,
        w_throughput=args.w_throughput,
        w_latency=args.w_latency,
        w_switch=args.w_switch,
        w_safe=args.w_safe,
        v_min=V_MIN,
        v_max=V_MAX,
        reward_clip=args.reward_clip,
        switch_mode=args.switch_mode,
        switch_cooldown_tau=args.switch_cooldown_tau,
        action_mode=args.action_mode,
    )

    site_ids = list(all_site_specs.keys())

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    def _train_and_save(env_factory, label: str, save_name: str) -> None:
        """Build a (possibly vectorized) env from ``env_factory`` and train one PPO model.

        ``env_factory`` is a zero-arg callable returning a fresh ``BatchSizeEnv``
        (or subclass). It is invoked once per parallel environment, wrapped with
        ``Monitor``, and stitched into a vec-env. ``SubprocVecEnv`` is used when
        ``args.n_envs > 1`` because each rollout needs its own OpenDSS instance
        (OpenDSS holds global state, so multiple envs in one process collide).
        """

        def _make_one():
            env = env_factory()
            return Monitor(env)

        if args.n_envs > 1:
            vec_env = SubprocVecEnv([_make_one for _ in range(args.n_envs)])
        else:
            vec_env = DummyVecEnv([_make_one])

        if args.vec_normalize:
            vn_init_ckpt = None
            if args.init_from:
                _p = Path(args.init_from)
                _cand = _p.with_name(_p.name.replace("ppo_", "ppo_vecnormalize_", 1).replace(".zip", ".pkl"))
                if _cand.exists():
                    vn_init_ckpt = _cand
            if vn_init_ckpt is not None:
                vec_env = VecNormalize.load(str(vn_init_ckpt), vec_env)
                vec_env.training = True
                vec_env.norm_reward = True
                logger.info("Loaded VecNormalize stats from %s", vn_init_ckpt)
            else:
                if args.init_from:
                    logger.warning(
                        "--init-from set but no VecNormalize sibling pkl found; starting VecNormalize stats fresh."
                    )
                vec_env = VecNormalize(
                    vec_env,
                    norm_obs=True,
                    norm_reward=True,
                    clip_obs=10.0,
                    clip_reward=10.0,
                    gamma=args.gamma,
                )

        obs_dim = int(vec_env.observation_space.shape[0])
        if hasattr(vec_env.action_space, "nvec"):
            n_act = int(len(vec_env.action_space.nvec))
        elif hasattr(vec_env.action_space, "n"):
            n_act = int(vec_env.action_space.n)
        else:
            n_act = int(vec_env.action_space.shape[0])
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training '%s': obs_dim=%d, n_actions=%d", label, obs_dim, n_act)
        logger.info(
            "  shared=%s, n_envs=%d, hidden_dims=%s, vec_normalize=%s",
            args.shared,
            args.n_envs,
            tuple(args.hidden_dims),
            args.vec_normalize,
        )
        logger.info(
            "  reward weights: voltage=%s throughput=%s latency=%s switch=%s safe=%s  reward_clip=%s",
            args.w_voltage,
            args.w_throughput,
            args.w_latency,
            args.w_switch,
            args.w_safe,
            args.reward_clip,
        )
        logger.info("  ofo_baseline=%s", args.ofo_baseline)
        logger.info(
            "  switch_mode=%s  switch_cooldown_tau=%s  action_mode=%s",
            args.switch_mode,
            args.switch_cooldown_tau,
            resolve_action_mode(reward_config),
        )
        logger.info("=" * 60)

        checkpoint_cb = CheckpointCallback(
            save_freq=max(args.n_steps * 10, 1),
            save_path=str(output_dir / "checkpoints" / label),
            name_prefix="ppo",
            save_vecnormalize=args.vec_normalize,
        )
        metrics_csv = output_dir / f"metrics_{label}.csv"
        metrics_cb = TrainingMetricsCallback(metrics_csv)
        callbacks = CallbackList([checkpoint_cb, metrics_cb])

        tb_log = str(output_dir / "tb") if args.tensorboard else None

        if args.lr_schedule == "linear":
            _lr_init = float(args.learning_rate)

            def lr_arg(progress_remaining):
                return progress_remaining * _lr_init

        elif args.lr_schedule == "constant":
            lr_arg = args.learning_rate
        else:
            raise ValueError(f"Unknown --lr-schedule: {args.lr_schedule!r} (expected 'constant' or 'linear')")

        if args.init_from:
            model = PPO.load(
                args.init_from,
                env=vec_env,
                device="auto",
                tensorboard_log=tb_log,
            )
            model.set_env(vec_env)
            logger.info(
                "Warm-started PPO from %s (num_timesteps=%d).",
                args.init_from,
                getattr(model, "num_timesteps", 0),
            )
        else:
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=lr_arg,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                verbose=1,
                seed=args.seed,
                tensorboard_log=tb_log,
                policy_kwargs=dict(net_arch=list(args.hidden_dims)),
            )
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name=label,
            reset_num_timesteps=not bool(args.init_from),
        )
        model_path = output_dir / save_name
        model.save(str(model_path))
        logger.info("Saved '%s' model to %s.zip", label, model_path)
        if args.vec_normalize:
            # VecNormalize running stats MUST be reloaded at inference time,
            # otherwise the policy sees unnormalized obs and acts nonsensically.
            vn_path = output_dir / f"{save_name}_vecnormalize.pkl"
            vec_env.save(str(vn_path))
            logger.info("Saved VecNormalize stats to %s", vn_path)
        vec_env.close()

        if args.plot:
            try:
                plot_path = plot_training_progress(metrics_csv, output_dir / f"training_progress_{label}.png", label)
                if plot_path is not None:
                    logger.info("Wrote training plot to %s", plot_path)
                else:
                    logger.warning("No metrics rows in %s — skipping plot", metrics_csv)
            except Exception as e:
                logger.warning("Plotting failed for '%s': %s", label, e)

    if args.shared and len(site_ids) > 1:
        # ── Shared multi-site PPO ──
        logger.info("Training SHARED PPO for %d sites: %s", len(site_ids), site_ids)

        site_model_mapping = {sid: [s.model_label for s in all_site_specs[sid]] for sid in site_ids}
        all_initial_bs_flat = {label: bs for sid in site_ids for label, bs in all_initial_batch_sizes[sid].items()}
        zone_summary = (
            {zname: tuple(zbuses) for zname, zbuses in zones.items()}
            if zones is not None and args.obs_mode in ("per-zone-summary", "per-bus-summary")
            else None
        )
        obs_config = ObservationConfig.from_multi_site(
            all_site_specs,
            all_replica_counts,
            n_bus_phases=n_bus_phases,
            initial_batch_sizes=all_initial_bs_flat,
            zone_summary=zone_summary,
            bus_phase_groups=bus_phase_groups,
            v_min=V_MIN,
            v_max=V_MAX,
        )

        def shared_env_factory():
            return SharedBatchSizeEnv(
                make_sim_fn=make_sim,
                obs_config=obs_config,
                site_model_mapping=site_model_mapping,
                reward_config=reward_config,
                logistic_models=logistic_models,
                dt_ctrl=DT_CTRL,
                total_duration_s=args.total_duration_s,
                scenario_library=scenario_lib,
                ofo_baseline=args.ofo_baseline and scenario_lib is not None,
                truncate_episode=args.truncate_episode and scenario_lib is not None,
            )

        _train_and_save(shared_env_factory, "shared", "ppo_model_shared")

    else:
        # ── Per-site PPO ──
        logger.info("Training %d separate PPO(s): %s", len(site_ids), site_ids)

        for sid in site_ids:
            specs = all_site_specs[sid]
            replica_counts = all_replica_counts[sid]

            # Zone-local voltage filtering for systems with zone definitions
            zone_buses = None
            n_bp = n_bus_phases
            if zones is not None and sid in zones and args.obs_mode == "full-voltage":
                zone_buses = tuple(zones[sid])
                zone_mask = compute_zone_mask(v_index, zone_buses)
                n_bp = int(np.sum(zone_mask))
                logger.info("Site '%s': using zone-local obs with %d/%d bus-phases", sid, n_bp, n_bus_phases_full)

            site_initial_bs = all_initial_batch_sizes[sid]
            obs_config = ObservationConfig.from_model_specs(
                specs,
                replica_counts,
                n_bus_phases=n_bp,
                initial_batch_sizes=site_initial_bs,
                zone_buses=zone_buses,
                v_min=V_MIN,
                v_max=V_MAX,
            )

            def site_env_factory(_obs_config=obs_config, _sid=sid, _lib=scenario_lib):
                return BatchSizeEnv(
                    make_sim_fn=make_sim,
                    obs_config=_obs_config,
                    agent_site_id=_sid,
                    reward_config=reward_config,
                    logistic_models=logistic_models,
                    dt_ctrl=DT_CTRL,
                    total_duration_s=args.total_duration_s,
                    scenario_library=_lib,
                    ofo_baseline=args.ofo_baseline and _lib is not None,
                    truncate_episode=args.truncate_episode and _lib is not None,
                )

            _train_and_save(site_env_factory, sid, f"ppo_model_{sid}")

    # For single-site, also save without site suffix for backwards compat
    if len(site_ids) == 1:
        import shutil

        src = output_dir / f"ppo_model_{site_ids[0]}.zip"
        dst = output_dir / "ppo_model.zip"
        shutil.copy2(src, dst)
        logger.info("Copied to %s", dst)

    logger.info("All done. Models saved to %s", output_dir)


if __name__ == "__main__":
    main()
