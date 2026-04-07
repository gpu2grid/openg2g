"""Train a PPO controller for batch-size voltage regulation.

Usage:
    python train_ppo.py --system ieee13 --total-timesteps 200000
    python train_ppo.py --system ieee13 --total-timesteps 7200  # smoke test (2 episodes)
    python train_ppo.py --system ieee13 --no-full-voltage       # summary-only obs
    python train_ppo.py --system ieee13 --randomize             # randomize scenarios
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
    tap,
)

from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.datacenter.config import (
    DatacenterConfig,
    InferenceRamp,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.grid.config import TapPosition, TapSchedule
from openg2g.rl.env import BatchSizeEnv, ObservationConfig, RewardConfig

logger = logging.getLogger(__name__)

TOTAL_DURATION_S = 3600

# ── Model definitions ───────────────────────────────────────────────────────

# Base replica counts for ieee13 (5 models)
_IEEE13_MODELS = {
    "Llama-3.1-8B": 720,
    "Llama-3.1-70B": 180,
    "Llama-3.1-405B": 90,
    "Qwen3-30B-A3B": 480,
    "Qwen3-235B-A22B": 210,
}


# ── IEEE 13-bus experiment (single DC) ───────────────────────────────────────


def _ieee13_experiment() -> dict:
    """IEEE 13-bus: single DC at bus 671 with 5 LLM models."""
    sys = SYSTEMS["ieee13"]()
    models = tuple(deploy(label, n) for label, n in _IEEE13_MODELS.items())
    ramps = (
        InferenceRamp(target=144, model="Llama-3.1-8B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=36, model="Llama-3.1-70B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=18, model="Llama-3.1-405B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=96, model="Qwen3-30B-A3B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=42, model="Qwen3-235B-A22B").at(t_start=2500, t_end=3000)
    )
    dc_site = DCSite(
        bus="671",
        bus_kv=sys["bus_kv"],
        base_kw_per_phase=500.0,
        total_gpu_capacity=7200,
        models=models,
        seed=0,
        inference_ramps=ramps,
    )
    tap_schedule = TapSchedule(
        (
            (1500, TapPosition(regulators={"creg1a": tap(16), "creg1b": tap(6), "creg1c": tap(17)})),
            (3300, TapPosition(regulators={"creg1a": tap(10), "creg1b": tap(6), "creg1c": tap(10)})),
        )
    )
    pv_systems = [PVSystemSpec(bus="675", bus_kv=4.16, peak_kw=10.0)]
    time_varying_loads = [TimeVaryingLoadSpec(bus="680", bus_kv=4.16, peak_kw=10.0)]
    return dict(
        sys=sys,
        dc_site=dc_site,
        tap_schedule=tap_schedule,
        pv_systems=pv_systems,
        time_varying_loads=time_varying_loads,
    )


EXPERIMENTS = {"ieee13": _ieee13_experiment}


# ── Simulation factory ──────────────────────────────────────────────────────


def make_sim_factory(
    exp: dict,
    inference_data: InferenceData,
    randomize: bool = False,
):
    """Return a callable that builds fresh simulation components.

    When *randomize* is ``True``, each call varies:
    - ramp target fraction (0.1 – 0.4 of base replicas)
    - ramp timing (start 1500–3000 s, duration 300–800 s)
    - PV peak scale (0.5 – 2.0×)
    - load peak scale (0.5 – 2.0×)
    - datacenter seed (different per episode)
    """
    sys = exp["sys"]
    dc_site: DCSite = exp["dc_site"]
    pv_systems_base = exp.get("pv_systems", [])
    tvl_base = exp.get("time_varying_loads", [])

    site_specs = tuple(md.spec for md in dc_site.models)
    replica_counts = {md.spec.model_label: md.num_replicas for md in dc_site.models}
    site_inference = inference_data.filter_models(site_specs)

    dc_config = DatacenterConfig(
        gpus_per_server=8,
        base_kw_per_phase=dc_site.base_kw_per_phase,
    )

    _episode_counter = [0]

    def make_sim():
        ep = _episode_counter[0]
        _episode_counter[0] += 1

        if randomize:
            rng = np.random.default_rng(seed=ep * 1000 + 7)
            # Randomize ramp targets and timing
            ramp_frac = rng.uniform(0.1, 0.4)
            ramp_start = rng.uniform(1500, 3000)
            ramp_dur = rng.uniform(300, 800)
            ramp_end = ramp_start + ramp_dur

            ramps = None
            for label, base_n in _IEEE13_MODELS.items():
                target = max(1, int(ramp_frac * base_n))
                r = InferenceRamp(target=target, model=label).at(t_start=ramp_start, t_end=ramp_end)
                ramps = r if ramps is None else (ramps | r)

            # Randomize PV and load scales
            pv_scale = rng.uniform(0.5, 2.0)
            load_scale = rng.uniform(0.5, 2.0)
            pv_systems = [
                PVSystemSpec(bus=s.bus, bus_kv=s.bus_kv, peak_kw=s.peak_kw * pv_scale) for s in pv_systems_base
            ]
            tvl = [TimeVaryingLoadSpec(bus=s.bus, bus_kv=s.bus_kv, peak_kw=s.peak_kw * load_scale) for s in tvl_base]
            dc_seed = ep
        else:
            ramps = dc_site.inference_ramps
            pv_systems = pv_systems_base
            tvl = tvl_base
            dc_seed = dc_site.seed

        workload_kwargs: dict = {
            "inference_data": site_inference,
            "replica_counts": replica_counts,
        }
        if ramps is not None:
            workload_kwargs["inference_ramps"] = ramps
        workload = OfflineWorkload(**workload_kwargs)

        dc = OfflineDatacenter(
            dc_config,
            workload,
            dt_s=DT_DC,
            seed=dc_seed,
            power_augmentation=POWER_AUG,
            total_gpu_capacity=dc_site.total_gpu_capacity,
        )

        grid = ScenarioOpenDSSGrid(
            pv_systems=pv_systems,
            time_varying_loads=tvl,
            source_pu=sys["source_pu"],
            dss_case_dir=sys["dss_case_dir"],
            dss_master_file=sys["dss_master_file"],
            dc_bus=dc_site.bus,
            dc_bus_kv=dc_site.bus_kv,
            power_factor=dc_config.power_factor,
            dt_s=DT_GRID,
            connection_type=dc_site.connection_type,
            initial_tap_position=sys["initial_taps"],
        )

        tap_ctrl = TapScheduleController(schedule=TapSchedule(()), dt_s=DT_CTRL)

        return dc, grid, tap_ctrl

    return make_sim, site_specs, replica_counts


# ── Main ────────────────────────────────────────────────────────────────────


@dataclass
class Args:
    system: str = "ieee13"
    """System name (ieee13)."""
    total_timesteps: int = 200_000
    """Total environment timesteps for training."""
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

    # Load data
    logger.info("Loading data for %s...", args.system)
    dc_site: DCSite = exp["dc_site"]
    site_specs = tuple(md.spec for md in dc_site.models)

    data_sources, _, data_dir = load_data_sources()
    # Filter to only models in our experiment
    needed_labels = {s.model_label for s in site_specs}
    filtered_sources = {k: v for k, v in data_sources.items() if k in needed_labels}
    inference_data = InferenceData.ensure(
        data_dir,
        site_specs,
        filtered_sources,
        plot=False,
        dt_s=float(DT_DC),
    )

    # Load logistic models for reward throughput computation
    from openg2g.controller.ofo import LogisticModelStore

    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv",
        site_specs,
        filtered_sources,
        plot=False,
    )

    # Build simulation factory
    make_sim, _, _ = make_sim_factory(exp, inference_data, randomize=args.randomize)

    # Probe grid for n_bus_phases (start a temporary grid, read v_index size)
    replica_counts = {md.spec.model_label: md.num_replicas for md in dc_site.models}
    dc_probe, grid_probe, _ = make_sim()
    dc_probe.do_reset()
    grid_probe.do_reset()
    dc_probe.start()
    grid_probe.start()
    n_bus_phases = len(grid_probe.v_index) if not args.no_full_voltage else 0
    logger.info("Grid has %d bus-phase pairs, using %d in obs", len(grid_probe.v_index), n_bus_phases)
    grid_probe.stop()
    dc_probe.stop()

    # Build observation and reward configs
    obs_config = ObservationConfig.from_model_specs(
        site_specs, replica_counts, n_bus_phases=n_bus_phases, v_min=V_MIN, v_max=V_MAX
    )
    reward_config = RewardConfig(
        w_voltage=args.w_voltage,
        w_throughput=args.w_throughput,
        w_latency=args.w_latency,
        w_switch=args.w_switch,
        v_min=V_MIN,
        v_max=V_MAX,
    )

    # Create environment
    env = BatchSizeEnv(
        make_sim_fn=make_sim,
        obs_config=obs_config,
        reward_config=reward_config,
        logistic_models=logistic_models,
        dt_ctrl=DT_CTRL,
        total_duration_s=TOTAL_DURATION_S,
    )

    # Train
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    logger.info("Starting PPO training: %d timesteps", args.total_timesteps)
    logger.info(
        "  obs_dim=%d, n_models=%d, full_voltage=%s, randomize=%s",
        obs_config.obs_dim,
        obs_config.n_models,
        not args.no_full_voltage,
        args.randomize,
    )
    logger.info("  action_dims=%s", [len(obs_config.feasible_batch_sizes[m]) for m in obs_config.model_labels])

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.n_steps * 10, 1),
        save_path=str(output_dir / "checkpoints"),
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

    # Save final model
    model_path = output_dir / "ppo_model"
    model.save(str(model_path))
    logger.info("Saved trained model to %s.zip", model_path)

    env.close()


if __name__ == "__main__":
    main()
