"""Train a PPO controller for batch-size voltage regulation.

Trains one PPO model per datacenter site.  For multi-DC systems (ieee34,
ieee123), each site gets its own policy while other sites use fixed
mid-range batch sizes during that site's training.

Usage:
    python train_ppo.py --system ieee13 --total-timesteps 200000
    python train_ppo.py --system ieee34 --total-timesteps 200000 --randomize
    python train_ppo.py --system ieee123 --total-timesteps 200000 --randomize
    python train_ppo.py --system ieee13 --no-full-voltage       # summary-only obs
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from sweep_dc_locations import ScenarioOpenDSSGrid
from systems import (
    DT_CTRL,
    DT_DC,
    DT_GRID,
    POWER_AUG,
    SYSTEMS,
    V_MAX,
    V_MIN,
    DCSite,
    PVSystemSpec,
    TimeVaryingLoadSpec,
    deploy,
    load_data_sources,
)

from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.datacenter.config import (
    DatacenterConfig,
    InferenceModelSpec,
    InferenceRamp,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.grid.config import DCLoadSpec, TapSchedule
from openg2g.rl.env import BatchSizeEnv, ObservationConfig, RewardConfig, SharedBatchSizeEnv, compute_zone_mask

logger = logging.getLogger(__name__)

TOTAL_DURATION_S = 3600


# ── Per-system experiment definitions ───────────────────────────────────────


def _ieee13_experiment() -> dict:
    """IEEE 13-bus: single DC at bus 671 with 5 LLM models."""
    sys = SYSTEMS["ieee13"]()
    models = (
        deploy("Llama-3.1-8B", 720),
        deploy("Llama-3.1-70B", 180),
        deploy("Llama-3.1-405B", 90),
        deploy("Qwen3-30B-A3B", 480),
        deploy("Qwen3-235B-A22B", 210),
    )
    ramps = (
        InferenceRamp(target=144, model="Llama-3.1-8B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=36, model="Llama-3.1-70B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=18, model="Llama-3.1-405B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=96, model="Qwen3-30B-A3B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=42, model="Qwen3-235B-A22B").at(t_start=2500, t_end=3000)
    )
    return dict(
        sys=sys,
        dc_sites={
            "default": DCSite(
                bus="671",
                bus_kv=sys["bus_kv"],
                base_kw_per_phase=500.0,
                total_gpu_capacity=7200,
                models=models,
                seed=0,
                inference_ramps=ramps,
            ),
        },
        pv_systems=[PVSystemSpec(bus="675", bus_kv=4.16, peak_kw=10.0)],
        time_varying_loads=[TimeVaryingLoadSpec(bus="680", bus_kv=4.16, peak_kw=10.0)],
    )


def _ieee34_experiment() -> dict:
    """IEEE 34-bus: two DC sites (upstream/downstream)."""
    sys = SYSTEMS["ieee34"]()
    return dict(
        sys=sys,
        dc_sites={
            "upstream": DCSite(
                bus="850",
                bus_kv=24.9,
                base_kw_per_phase=120.0,
                models=(deploy("Llama-3.1-8B", 720), deploy("Llama-3.1-70B", 180), deploy("Llama-3.1-405B", 90)),
                seed=0,
                total_gpu_capacity=2400,
            ),
            "downstream": DCSite(
                bus="834",
                bus_kv=24.9,
                base_kw_per_phase=80.0,
                models=(deploy("Qwen3-30B-A3B", 480), deploy("Qwen3-235B-A22B", 210)),
                seed=42,
                total_gpu_capacity=2880,
            ),
        },
        pv_systems=[
            PVSystemSpec(bus="848", bus_kv=24.9, peak_kw=130.0),
            PVSystemSpec(bus="830", bus_kv=24.9, peak_kw=65.0),
        ],
        time_varying_loads=[
            TimeVaryingLoadSpec(bus="860", bus_kv=24.9, peak_kw=80.0),
            TimeVaryingLoadSpec(bus="844", bus_kv=24.9, peak_kw=120.0),
            TimeVaryingLoadSpec(bus="840", bus_kv=24.9, peak_kw=60.0),
            TimeVaryingLoadSpec(bus="858", bus_kv=24.9, peak_kw=50.0),
            TimeVaryingLoadSpec(bus="854", bus_kv=24.9, peak_kw=40.0),
        ],
    )


def _ieee123_experiment() -> dict:
    """IEEE 123-bus: four DC sites across zones."""
    sys = SYSTEMS["ieee123"]()
    return dict(
        sys=sys,
        dc_sites={
            "z1_sw": DCSite(
                bus="8",
                bus_kv=4.16,
                base_kw_per_phase=310.0,
                models=(deploy("Llama-3.1-8B", 120),),
                seed=0,
                total_gpu_capacity=180,
                inference_ramps=InferenceRamp(target=180, model="Llama-3.1-8B").at(t_start=500, t_end=1000),
            ),
            "z2_nw": DCSite(
                bus="23",
                bus_kv=4.16,
                base_kw_per_phase=265.0,
                models=(deploy("Qwen3-30B-A3B", 80),),
                seed=17,
                total_gpu_capacity=208,
                inference_ramps=InferenceRamp(target=104, model="Qwen3-30B-A3B").at(t_start=1500, t_end=2500),
            ),
            "z3_se": DCSite(
                bus="60",
                bus_kv=4.16,
                base_kw_per_phase=295.0,
                models=(deploy("Llama-3.1-70B", 30), deploy("Llama-3.1-405B", 35)),
                seed=34,
                total_gpu_capacity=460,
                inference_ramps=InferenceRamp(target=45, model="Llama-3.1-70B").at(t_start=700, t_end=1100),
            ),
            "z4_ne": DCSite(
                bus="105",
                bus_kv=4.16,
                base_kw_per_phase=325.0,
                models=(deploy("Qwen3-235B-A22B", 55),),
                seed=51,
                total_gpu_capacity=440,
                inference_ramps=InferenceRamp(target=27, model="Qwen3-235B-A22B").at(t_start=2000, t_end=2500),
            ),
        },
        pv_systems=[
            PVSystemSpec(bus="1", bus_kv=4.16, peak_kw=333.3),
            PVSystemSpec(bus="48", bus_kv=4.16, peak_kw=333.3),
            PVSystemSpec(bus="99", bus_kv=4.16, peak_kw=333.3),
        ],
        time_varying_loads=[],
    )


EXPERIMENTS = {"ieee13": _ieee13_experiment, "ieee34": _ieee34_experiment, "ieee123": _ieee123_experiment}


# ── Simulation factory ──────────────────────────────────────────────────────


def _randomize_ramps(
    dc_sites: dict[str, DCSite],
    rng: np.random.Generator,
) -> dict[str, DCSite]:
    """Return a copy of dc_sites with randomized ramp targets and timing."""
    ramp_frac = rng.uniform(0.1, 0.4)
    ramp_start = rng.uniform(500, 3000)
    ramp_dur = rng.uniform(300, 800)
    ramp_end = ramp_start + ramp_dur

    new_sites: dict[str, DCSite] = {}
    for sid, site in dc_sites.items():
        ramps = None
        for md in site.models:
            target = max(1, int(ramp_frac * md.num_replicas))
            r = InferenceRamp(target=target, model=md.spec.model_label).at(t_start=ramp_start, t_end=ramp_end)
            ramps = r if ramps is None else (ramps | r)
        new_sites[sid] = DCSite(
            bus=site.bus,
            bus_kv=site.bus_kv,
            base_kw_per_phase=site.base_kw_per_phase,
            total_gpu_capacity=site.total_gpu_capacity,
            models=site.models,
            seed=int(rng.integers(0, 10000)),
            connection_type=site.connection_type,
            inference_ramps=ramps,
        )
    return new_sites


def make_sim_factory(
    exp: dict,
    inference_data: InferenceData,
    randomize: bool = False,
):
    """Return a callable that builds fresh simulation components.

    Returns ``(make_sim, all_site_specs, all_replica_counts)`` where
    ``make_sim()`` produces ``(dict[str, DatacenterBackend], grid, tap_ctrl)``.
    """
    sys = exp["sys"]
    dc_sites: dict[str, DCSite] = exp["dc_sites"]
    pv_systems_base = exp.get("pv_systems", [])
    tvl_base = exp.get("time_varying_loads", [])

    # For single-DC, remap site key to "_default" to match grid's internal key
    is_single_dc = len(dc_sites) == 1
    if is_single_dc:
        orig_sid = next(iter(dc_sites))
        dc_sites = {"_default": dc_sites[orig_sid]}

    # Collect specs across all sites
    all_site_specs: dict[str, tuple[InferenceModelSpec, ...]] = {}
    all_replica_counts: dict[str, dict[str, int]] = {}
    site_inference: dict[str, InferenceData] = {}
    for sid, site in dc_sites.items():
        specs = tuple(md.spec for md in site.models)
        all_site_specs[sid] = specs
        all_replica_counts[sid] = {md.spec.model_label: md.num_replicas for md in site.models}
        site_inference[sid] = inference_data.filter_models(specs)

    _episode_counter = [0]

    def make_sim():
        ep = _episode_counter[0]
        _episode_counter[0] += 1

        if randomize:
            rng = np.random.default_rng(seed=ep * 1000 + 7)
            sites = _randomize_ramps(dc_sites, rng)
            pv_scale = rng.uniform(0.5, 2.0)
            load_scale = rng.uniform(0.5, 2.0)
            pv_systems = [
                PVSystemSpec(bus=s.bus, bus_kv=s.bus_kv, peak_kw=s.peak_kw * pv_scale) for s in pv_systems_base
            ]
            tvl = [TimeVaryingLoadSpec(bus=s.bus, bus_kv=s.bus_kv, peak_kw=s.peak_kw * load_scale) for s in tvl_base]
        else:
            sites = dc_sites
            pv_systems = pv_systems_base
            tvl = tvl_base

        # Build all datacenters
        datacenters: dict[str, OfflineDatacenter] = {}
        dc_loads: dict[str, DCLoadSpec] = {}
        for sid, site in sites.items():
            dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=site.base_kw_per_phase)
            rc = all_replica_counts[sid]
            wl_kwargs: dict = {"inference_data": site_inference[sid], "replica_counts": rc}
            if site.inference_ramps is not None:
                wl_kwargs["inference_ramps"] = site.inference_ramps
            workload = OfflineWorkload(**wl_kwargs)
            datacenters[sid] = OfflineDatacenter(
                dc_config,
                workload,
                dt_s=DT_DC,
                seed=site.seed,
                power_augmentation=POWER_AUG,
                total_gpu_capacity=site.total_gpu_capacity,
            )
            dc_loads[sid] = DCLoadSpec(bus=site.bus, bus_kv=site.bus_kv, connection_type=site.connection_type)

        # Build grid
        site_ids = list(sites.keys())
        dc_config_pf = DatacenterConfig(base_kw_per_phase=0).power_factor
        if len(site_ids) == 1:
            # Single-DC: use dc_bus= path (matches analyze script and Coordinator)
            # Remap datacenters dict to "_default" to match grid's internal key
            sid = site_ids[0]
            site = sites[sid]
            grid = ScenarioOpenDSSGrid(
                pv_systems=pv_systems,
                time_varying_loads=tvl,
                source_pu=sys["source_pu"],
                dss_case_dir=sys["dss_case_dir"],
                dss_master_file=sys["dss_master_file"],
                dc_bus=site.bus,
                dc_bus_kv=site.bus_kv,
                power_factor=dc_config_pf,
                dt_s=DT_GRID,
                connection_type=site.connection_type,
                initial_tap_position=sys["initial_taps"],
            )
            # datacenters dict already uses "_default" from the remap above
        else:
            exclude = tuple(sys.get("exclude_buses", ()))
            grid = ScenarioOpenDSSGrid(
                pv_systems=pv_systems,
                time_varying_loads=tvl,
                source_pu=sys["source_pu"],
                dss_case_dir=sys["dss_case_dir"],
                dss_master_file=sys["dss_master_file"],
                dc_loads=dc_loads,
                power_factor=dc_config_pf,
                dt_s=DT_GRID,
                initial_tap_position=sys["initial_taps"],
                exclude_buses=exclude,
            )

        tap_ctrl = TapScheduleController(schedule=TapSchedule(()), dt_s=DT_CTRL)
        return datacenters, grid, tap_ctrl

    return make_sim, all_site_specs, all_replica_counts


# ── Main ────────────────────────────────────────────────────────────────────


@dataclass
class Args:
    system: str = "ieee13"
    """System name (ieee13, ieee34, ieee123)."""
    total_timesteps: int = 200_000
    """Total environment timesteps for training (per site)."""
    learning_rate: float = 1e-3
    """PPO learning rate."""
    n_steps: int = 3600
    """Rollout length (steps per episode)."""
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
    ent_coef: float = 0.05
    """Entropy coefficient."""
    hidden_dim: int = 128
    """Hidden layer size for policy network."""
    w_voltage: float = 1000.0
    """Reward weight for voltage violations."""
    w_throughput: float = 0.01
    """Reward weight for throughput."""
    w_latency: float = 10.0
    """Reward weight for latency violations."""
    w_switch: float = 0.1
    """Reward weight for switching cost."""
    no_full_voltage: bool = False
    """Use summary-only observations (no full voltage vector)."""
    shared: bool = False
    """Train one shared PPO for all sites (instead of separate per-site)."""
    randomize: bool = False
    """Randomize scenario each episode (ramp timing/scale, PV/load)."""
    output_dir: str = ""
    """Output directory (default: outputs/<system>/ppo)."""
    log_level: str = "INFO"
    """Logging verbosity."""
    seed: int = 42
    """Random seed."""


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

    exp = EXPERIMENTS[args.system]()
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve() if args.output_dir else script_dir / "outputs" / args.system / "ppo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all model specs across sites
    dc_sites: dict[str, DCSite] = exp["dc_sites"]
    all_specs: list[InferenceModelSpec] = []
    for site in dc_sites.values():
        all_specs.extend(md.spec for md in site.models)
    all_specs_tuple = tuple({s.model_label: s for s in all_specs}.values())

    # Load data
    logger.info("Loading data for %s...", args.system)
    data_sources, _, data_dir = load_data_sources()
    needed_labels = {s.model_label for s in all_specs_tuple}
    filtered_sources = {k: v for k, v in data_sources.items() if k in needed_labels}
    inference_data = InferenceData.ensure(
        data_dir,
        all_specs_tuple,
        filtered_sources,
        plot=False,
        dt_s=float(DT_DC),
    )

    from openg2g.controller.ofo import LogisticModelStore

    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv",
        all_specs_tuple,
        filtered_sources,
        plot=False,
    )

    # Build simulation factory
    make_sim, all_site_specs, all_replica_counts = make_sim_factory(exp, inference_data, randomize=args.randomize)

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

    n_bus_phases = 0 if args.no_full_voltage else n_bus_phases_full
    logger.info("Grid has %d bus-phase pairs, using %d in obs", n_bus_phases_full, n_bus_phases)

    # Zone info for IEEE 123 (zone-local voltage filtering)
    zones: dict[str, list[str]] | None = exp.get("sys", {}).get("zones")

    reward_config = RewardConfig(
        w_voltage=args.w_voltage,
        w_throughput=args.w_throughput,
        w_latency=args.w_latency,
        w_switch=args.w_switch,
        v_min=V_MIN,
        v_max=V_MAX,
    )

    site_ids = list(all_site_specs.keys())

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    def _train_and_save(env, label: str, save_name: str) -> None:
        """Train one PPO model and save it."""
        obs_dim = env.observation_space.shape[0]
        n_act = env.action_space.shape[0] if hasattr(env.action_space, "shape") else len(env.action_space.nvec)
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training '%s': obs_dim=%d, n_actions=%d", label, obs_dim, n_act)
        logger.info("  shared=%s, randomize=%s", args.shared, args.randomize)
        logger.info("=" * 60)

        checkpoint_cb = CheckpointCallback(
            save_freq=max(args.n_steps * 10, 1),
            save_path=str(output_dir / "checkpoints" / label),
            name_prefix="ppo",
        )
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            verbose=1,
            seed=args.seed,
            policy_kwargs=dict(net_arch=[args.hidden_dim, args.hidden_dim]),
        )
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_cb)
        model_path = output_dir / save_name
        model.save(str(model_path))
        logger.info("Saved '%s' model to %s.zip", label, model_path)
        env.close()

    if args.shared and len(site_ids) > 1:
        # ── Shared multi-site PPO ──
        logger.info("Training SHARED PPO for %d sites: %s", len(site_ids), site_ids)

        site_model_mapping = {sid: [s.model_label for s in all_site_specs[sid]] for sid in site_ids}
        obs_config = ObservationConfig.from_multi_site(
            all_site_specs, all_replica_counts, n_bus_phases=n_bus_phases, v_min=V_MIN, v_max=V_MAX
        )
        env = SharedBatchSizeEnv(
            make_sim_fn=make_sim,
            obs_config=obs_config,
            site_model_mapping=site_model_mapping,
            reward_config=reward_config,
            logistic_models=logistic_models,
            dt_ctrl=DT_CTRL,
            total_duration_s=TOTAL_DURATION_S,
        )
        _train_and_save(env, "shared", "ppo_model_shared")

    else:
        # ── Per-site PPO ──
        logger.info("Training %d separate PPO(s): %s", len(site_ids), site_ids)

        for sid in site_ids:
            specs = all_site_specs[sid]
            replica_counts = all_replica_counts[sid]

            # Zone-local voltage filtering for systems with zone definitions
            zone_buses = None
            n_bp = n_bus_phases
            if zones is not None and sid in zones and not args.no_full_voltage:
                zone_buses = tuple(zones[sid])
                zone_mask = compute_zone_mask(v_index, zone_buses)
                n_bp = int(np.sum(zone_mask))
                logger.info("Site '%s': using zone-local obs with %d/%d bus-phases", sid, n_bp, n_bus_phases_full)

            obs_config = ObservationConfig.from_model_specs(
                specs, replica_counts, n_bus_phases=n_bp, zone_buses=zone_buses, v_min=V_MIN, v_max=V_MAX
            )
            env = BatchSizeEnv(
                make_sim_fn=make_sim,
                obs_config=obs_config,
                agent_site_id=sid,
                reward_config=reward_config,
                logistic_models=logistic_models,
                dt_ctrl=DT_CTRL,
                total_duration_s=TOTAL_DURATION_S,
            )
            _train_and_save(env, sid, f"ppo_model_{sid}")

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
