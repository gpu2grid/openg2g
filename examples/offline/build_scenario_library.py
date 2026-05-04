"""Build a pre-screened scenario library for PPO training.

Generates randomized ieee13 scenarios using the same randomization logic as
``train_ppo.py:make_sim_factory``, runs both baseline (no controller) and OFO
on each, and accepts a scenario only if:

  1) The baseline episode has non-trivial voltage violation (so the policy
     has something to learn), and
  2) OFO recovers at least ``--min-recovery-frac`` of the baseline integral
     violation (so the violation is within the GPU-flexibility envelope).

For each accepted scenario the library stores the seed (so the env can
reproduce it) plus the OFO per-second voltage-penalty trace, which the env
will subtract from PPO's voltage cost during training to give a fair,
scenario-difficulty-normalized reward.

Usage:
    python examples/offline/build_scenario_library.py --n-candidates 10
    python examples/offline/build_scenario_library.py --n-candidates 50 \\
        --pv-base-kw 200 --tvl-base-kw 200 --min-recovery-frac 0.8

Outputs:
    examples/offline/outputs/ieee13/scenario_library/<tag>/
        library.pkl              -- list of accepted ScenarioRecord
        candidates.csv           -- per-candidate stats (accepted + rejected)
        scenario_envelopes.png   -- voltage envelope per scenario, baseline vs OFO
        scenario_summary.png     -- bar chart of integral violation, baseline vs OFO

The .pkl is the artifact training will load via train_ppo.py's
``--scenario-library`` flag (added in the next step).
"""

from __future__ import annotations

import csv
import logging
import math
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openg2g.controller.ofo import LogisticModelStore, OFOConfig
from openg2g.controller.rule_based import RuleBasedBatchSizeController, RuleBasedConfig
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.config import DatacenterConfig, InferenceModelSpec, ReplicaSchedule, TrainingRun
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.config import TapSchedule
from openg2g.metrics.voltage import VoltageStats, compute_allbus_voltage_stats

from systems import (
    DT_CTRL,
    DT_DC,
    DT_GRID,
    EXPERIMENTS,
    POWER_AUG,
    SPECS_CACHE_DIR,
    TOTAL_DURATION_S,
    TRAINING_TRACE_PATH,
    V_MAX,
    V_MIN,
    DCSite,
    PVSystemSpec,
    ScenarioOpenDSSGrid,
    TimeVaryingLoadSpec,
    randomize_scenario,
)

logger = logging.getLogger("scenario_library")


def run_simulation(
    mode: str,
    *,
    sys: dict,
    dc_sites: dict[str, DCSite],
    ofo_config,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models,
    pv_systems: list[PVSystemSpec] | None = None,
    time_varying_loads: list[TimeVaryingLoadSpec] | None = None,
    tap_schedule: TapSchedule | None = None,
    rule_based_config: RuleBasedConfig | None = None,
    rule_zone_local: bool = False,
    ppo_model: str = "",
    obs_mode: str = "full-voltage",
    training_overlay: dict | None = None,
    save_dir: Path,
) -> tuple[VoltageStats, object]:
    """Run a simulation with the specified controller mode.

    Modes:
        'baseline': NoopController (no batch control)
        'rule_based': RuleBasedBatchSizeController
        'ofo': OFOBatchSizeController
        'ppo': PPOBatchSizeController / SharedPPOBatchSizeController

    rule_zone_local: when True AND sys defines `zones` AND there is more than
    one DC site, each rule-based controller observes only the buses in its
    own zone (looked up via the site_id key). Decentralizes credit assignment
    in multi-DC topologies (ieee123). No effect for single-site systems.

    Returns (VoltageStats, SimulationLog).
    """
    pv_systems = pv_systems or []
    time_varying_loads = time_varying_loads or []
    exclude_buses = tuple(sys["exclude_buses"])
    site_ids = list(dc_sites.keys())

    training_run = None
    if training_overlay is not None:
        training_run = TrainingRun(
            n_gpus=training_overlay["n_gpus"],
            trace=training_trace,
            target_peak_W_per_gpu=training_overlay["target_peak_W_per_gpu"],
        ).at(t_start=training_overlay["t_start"], t_end=training_overlay["t_end"])

    datacenters: dict[str, OfflineDatacenter] = {}
    controllers: list = []
    site_specs_map: dict[str, tuple[InferenceModelSpec, ...]] = {}
    primary_bus = ""

    for site_id, site in dc_sites.items():
        site_specs = tuple(md.spec for md, _ in site.models)
        site_specs_map[site_id] = site_specs
        site_inference = inference_data.filter_models(site_specs)

        replica_schedules: dict[str, ReplicaSchedule] = {md.spec.model_label: sched for md, sched in site.models}
        initial_batch_sizes = {md.spec.model_label: md.initial_batch_size for md, _ in site.models}

        dc_config = DatacenterConfig(
            gpus_per_server=8,
            base_kw_per_phase=site.base_kw_per_phase,
        )
        workload_kwargs: dict = {
            "inference_data": site_inference,
            "replica_schedules": replica_schedules,
            "initial_batch_sizes": initial_batch_sizes,
        }
        if training_run is not None:
            workload_kwargs["training"] = training_run
        workload = OfflineWorkload(**workload_kwargs)

        dc = OfflineDatacenter(
            dc_config,
            workload,
            name=site_id,
            dt_s=DT_DC,
            seed=site.seed,
            power_augmentation=POWER_AUG,
            total_gpu_capacity=site.total_gpu_capacity,
        )
        datacenters[site_id] = dc
        if not primary_bus:
            primary_bus = site.bus

    # Build grid then attach DCs (master's imperative pattern; the old
    # dc_loads / dc_bus kwargs are no longer accepted by OpenDSSGrid).
    grid = ScenarioOpenDSSGrid(
        pv_systems=pv_systems,
        time_varying_loads=time_varying_loads,
        source_pu=sys["source_pu"],
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        dt_s=DT_GRID,
        initial_tap_position=sys["initial_taps"],
        exclude_buses=exclude_buses,
    )
    dc_pf = DatacenterConfig(base_kw_per_phase=0).power_factor
    for site_id, dc in datacenters.items():
        site = dc_sites[site_id]
        grid.attach_dc(dc, bus=site.bus, connection_type=site.connection_type, power_factor=dc_pf)

    if mode == "baseline":
        sched = tap_schedule if tap_schedule is not None else TapSchedule(())
    else:
        sched = TapSchedule(())
    controllers.append(TapScheduleController(schedule=sched, dt_s=DT_CTRL))

    if mode == "ofo":
        from openg2g.controller.ofo import OFOBatchSizeController

        for site_id in site_ids:
            site_initial_bs = {md.spec.model_label: md.initial_batch_size for md, _ in dc_sites[site_id].models}
            ofo_ctrl = OFOBatchSizeController(
                site_specs_map[site_id],
                datacenter=datacenters[site_id],
                grid=grid,
                models=logistic_models,
                config=ofo_config,
                dt_s=DT_CTRL,
                initial_batch_sizes=site_initial_bs,
            )
            controllers.append(ofo_ctrl)

    elif mode == "rule_based" or mode.startswith("rule_based_"):
        rb_config = rule_based_config or RuleBasedConfig(v_min=V_MIN, v_max=V_MAX)
        zones = sys.get("zones") if rule_zone_local else None
        for site_id in site_ids:
            site_initial_bs = {md.spec.model_label: md.initial_batch_size for md, _ in dc_sites[site_id].models}
            zone_buses = None
            if zones is not None and len(site_ids) > 1 and site_id in zones:
                zone_buses = tuple(zones[site_id])
            rb_ctrl = RuleBasedBatchSizeController(
                site_specs_map[site_id],
                datacenter=datacenters[site_id],
                grid=grid,
                config=rb_config,
                dt_s=DT_CTRL,
                exclude_buses=exclude_buses,
                zone_buses=zone_buses,
                initial_batch_sizes=site_initial_bs,
            )
            controllers.append(rb_ctrl)

    elif mode == "ppo":
        from openg2g.controller.ppo import PPOBatchSizeController, SharedPPOBatchSizeController
        from openg2g.rl.env import ObservationConfig

        ppo_path = Path(ppo_model).resolve()

        def _find_vecnormalize(model_file: Path) -> Path | None:
            mf = Path(model_file)
            stem = mf.with_suffix("").name
            candidates = [
                mf.parent / f"{stem}_vecnormalize.pkl",
                mf.parent / "vecnormalize.pkl",
            ]
            if stem.startswith("ppo_") and stem.endswith("_steps"):
                candidates.append(mf.parent / f"ppo_vecnormalize_{stem[len('ppo_') :]}.pkl")
            for parent in [mf.parent, *mf.parent.parents][:4]:
                candidates.extend(sorted(parent.glob("ppo_model_*_vecnormalize.pkl")))
            for c in candidates:
                if c.is_file():
                    return c
            return None

        shared_model = None
        if ppo_path.is_dir():
            cand = ppo_path / "ppo_model_shared.zip"
            if cand.exists():
                shared_model = cand
        elif ppo_path.suffix == ".zip":
            looks_shared = (
                len(site_ids) > 1 or "shared" in ppo_path.parts or ppo_path.stem.startswith("ppo_model_shared")
            )
            if looks_shared and ppo_path.exists():
                shared_model = ppo_path
        if shared_model is not None and shared_model.exists():
            from stable_baselines3 import PPO as SB3PPO

            sb3 = SB3PPO.load(str(shared_model.with_suffix("")))
            saved_obs_dim = sb3.observation_space.shape[0]
            n_models_all = sum(len(dc_sites[sid].models) for sid in site_ids)
            zones = sys.get("zones")

            if obs_mode == "per-zone-summary":
                n_bus_phases = 0
                zone_summary = {zname: tuple(zbuses) for zname, zbuses in zones.items()} if zones else None
                bus_phase_groups = None
            elif obs_mode == "system-summary-only":
                n_bus_phases = 0
                zone_summary = None
                bus_phase_groups = None
            elif obs_mode == "per-bus-summary":
                from openg2g.rl.env import compute_bus_phase_groups

                grid.do_reset()
                grid.start()
                _v_index = grid.v_index
                grid.stop()
                bus_phase_groups = compute_bus_phase_groups(_v_index)
                n_bus_phases = 2 * len(bus_phase_groups)
                zone_summary = {zname: tuple(zbuses) for zname, zbuses in zones.items()} if zones else None
            else:  # "full-voltage"
                n_bus_phases = saved_obs_dim - 3 - 5 * n_models_all
                zone_summary = None
                bus_phase_groups = None

            site_model_mapping = {sid: [md.spec.model_label for md, _ in dc_sites[sid].models] for sid in site_ids}
            all_init_bs = {
                md.spec.model_label: md.initial_batch_size for sid in site_ids for md, _ in dc_sites[sid].models
            }
            obs_config = ObservationConfig.from_multi_site(
                site_specs_map,
                {sid: {md.spec.model_label: sched.initial for md, sched in dc_sites[sid].models} for sid in site_ids},
                n_bus_phases=n_bus_phases,
                initial_batch_sizes=all_init_bs,
                zone_summary=zone_summary,
                bus_phase_groups=bus_phase_groups,
                v_min=V_MIN,
                v_max=V_MAX,
            )
            vn_path = _find_vecnormalize(shared_model)
            if vn_path is not None:
                logger.info("PPO: loading VecNormalize stats from %s", vn_path)
            else:
                logger.warning(
                    "PPO: no VecNormalize stats found next to %s — policy will see UNNORMALIZED obs", shared_model
                )
            ppo_ctrl = SharedPPOBatchSizeController(
                datacenter=next(iter(datacenters.values())),
                grid=grid,
                model_path=str(shared_model),
                obs_config=obs_config,
                site_model_mapping=site_model_mapping,
                dt_s=DT_CTRL,
                vecnormalize_path=str(vn_path) if vn_path is not None else None,
            )
            controllers.append(ppo_ctrl)
        else:
            zones = sys.get("zones")

            for site_id in site_ids:
                if ppo_path.is_dir():
                    site_model = str(ppo_path / f"ppo_model_{site_id}.zip")
                elif ppo_path.suffix == ".zip" and len(site_ids) == 1:
                    site_model = str(ppo_path)
                else:
                    site_model = str(ppo_path.parent / f"ppo_model_{site_id}.zip")

                from stable_baselines3 import PPO as SB3PPO

                sb3 = SB3PPO.load(str(Path(site_model).with_suffix("")))
                saved_obs_dim = sb3.observation_space.shape[0]
                n_models_site = len(dc_sites[site_id].models)
                n_bus_phases = saved_obs_dim - 3 - 5 * n_models_site

                zone_buses = None
                if zones is not None and site_id in zones:
                    zone_buses = tuple(zones[site_id])

                specs = site_specs_map[site_id]
                replica_counts = {md.spec.model_label: sched.initial for md, sched in dc_sites[site_id].models}
                site_init_bs = {md.spec.model_label: md.initial_batch_size for md, _ in dc_sites[site_id].models}
                obs_config = ObservationConfig.from_model_specs(
                    specs,
                    replica_counts,
                    n_bus_phases=n_bus_phases,
                    initial_batch_sizes=site_init_bs,
                    zone_buses=zone_buses,
                    v_min=V_MIN,
                    v_max=V_MAX,
                )
                vn_path = _find_vecnormalize(Path(site_model))
                if vn_path is not None:
                    logger.info("PPO[%s]: loading VecNormalize stats from %s", site_id, vn_path)
                else:
                    logger.warning(
                        "PPO[%s]: no VecNormalize stats found next to %s — policy will see UNNORMALIZED obs",
                        site_id,
                        site_model,
                    )
                ppo_ctrl = PPOBatchSizeController(
                    specs,
                    datacenter=datacenters[site_id],
                    grid=grid,
                    model_path=site_model,
                    obs_config=obs_config,
                    dt_s=DT_CTRL,
                    vecnormalize_path=str(vn_path) if vn_path is not None else None,
                )
                controllers.append(ppo_ctrl)

    coord = Coordinator(
        datacenters=list(datacenters.values()),
        grid=grid,
        controllers=controllers,
        total_duration_s=TOTAL_DURATION_S,
    )

    from openg2g.controller.ppo import SharedPPOBatchSizeController

    for ctrl in controllers:
        if isinstance(ctrl, SharedPPOBatchSizeController):
            ctrl.attach_datacenters(datacenters)

    logger.info("Running %s...", mode)
    log = coord.run()

    vstats = compute_allbus_voltage_stats(
        log.grid_states,
        v_min=V_MIN,
        v_max=V_MAX,
        exclude_buses=exclude_buses,
    )
    logger.info(
        "  %s: viol=%.1fs  integral=%.4f  vmin=%.4f  vmax=%.4f",
        mode,
        vstats.violation_time_s,
        vstats.integral_violation_pu_s,
        vstats.worst_vmin,
        vstats.worst_vmax,
    )

    return vstats, log


@dataclass
class ScenarioRecord:
    """One pre-screened scenario, ready to be replayed at training time.

    ``__module__`` is forced to ``build_scenario_library`` below so pickles
    of this class work whether the script is run as ``__main__`` or imported.
    Without the override, running ``python build_scenario_library.py`` writes
    ``__main__.ScenarioRecord`` into the pickle, which then fails to unpickle
    in any other process (e.g. train_ppo.py). See openg2g/rl/env.py:ScenarioLibrary
    stub registration.
    """

    seed: int
    pv_scale: float
    load_scale: float
    training_overlay: dict | None
    baseline_integral: float
    ofo_integral: float
    baseline_violation_time_s: float
    ofo_violation_time_s: float
    recovery_frac: float
    # Per-second OFO voltage penalty (sum of squared violation, length =
    # total_duration_s). This is what gets subtracted from PPO's per-step
    # voltage_pen during training.
    ofo_voltage_pen_per_step: np.ndarray = field(repr=False)
    baseline_voltage_pen_per_step: np.ndarray = field(repr=False)
    # Episode window for truncated training: skip quiet prefix, stop after last violation.
    # Computed from baseline_voltage_pen_per_step at build time with configurable buffers.
    # env.py reads these via getattr(..., default) so old pickles without these fields degrade
    # gracefully to full-episode training.
    t_control_start: int = 0
    t_control_end: int = 3600
    bl_undervoltage_time_s: float = 0.0
    bl_overvoltage_time_s: float = 0.0
    # Per-scenario override for inference-ramp synthesis at replay time.
    # When None, train_ppo uses the library-wide default from the library config.
    randomize_ramps: bool | None = None
    # Per-scenario ramp bounds. None falls back to the library-wide defaults in train_ppo.
    ramp_frac_min: float | None = None
    ramp_frac_max: float | None = None
    ramp_start_min: float | None = None
    ramp_start_max: float | None = None
    ramp_dur_min: float | None = None
    ramp_dur_max: float | None = None
    # Resolved per-episode state — captured at build time so consumers don't
    # need to replay randomize_scenario. Old pickles default to None and must
    # be upgraded with examples/offline/migrate_scenario_library.py before use.
    resolved_dc_sites: dict | None = None
    resolved_pv_systems: tuple | None = None
    resolved_tvl: tuple | None = None


# Force stable pickle identity regardless of how build_scenario_library.py is
# invoked (as __main__ script vs imported module). See docstring above.
ScenarioRecord.__module__ = "build_scenario_library"
# When run as `python build_scenario_library.py`, the script's globals live in
# sys.modules['__main__'] and there is no entry for 'build_scenario_library'.
# Pickle uses ScenarioRecord.__module__ to find the class at unpickle time, so
# we alias sys.modules so both names resolve to the same module object.
sys.modules.setdefault("build_scenario_library", sys.modules[__name__])


def _per_step_voltage_pen(grid_states, *, v_min: float, v_max: float, exclude_buses: tuple[str, ...]) -> np.ndarray:
    """Compute the per-step voltage penalty (sum of squared violation magnitude).

    This matches ``compute_reward`` in ``openg2g/rl/env.py``: at each step the
    penalty is ``sum_max(v_min - v, 0)^2 + sum_max(v - v_max, 0)^2`` over all
    bus-phases (excluding the substation buses), with NaNs treated as
    "not in violation".
    """
    drop = {b.lower() for b in exclude_buses}
    out = np.zeros(len(grid_states), dtype=np.float64)
    for i, gs in enumerate(grid_states):
        s = 0.0
        for bus in gs.voltages.buses():
            if bus.lower() in drop:
                continue
            pv = gs.voltages[bus]
            for v in (pv.a, pv.b, pv.c):
                if math.isnan(v):
                    continue
                if v < v_min:
                    s += (v_min - v) ** 2
                elif v > v_max:
                    s += (v - v_max) ** 2
        out[i] = s
    return out


def _under_over_voltage_time(
    grid_states, *, v_min: float, v_max: float, exclude_buses: tuple[str, ...]
) -> tuple[float, float]:
    """Return (undervoltage_time_s, overvoltage_time_s).

    A step counts as 'undervoltage' if ANY non-excluded bus-phase is below
    v_min, and as 'overvoltage' if ANY is above v_max. A step can count for
    both (they are not mutually exclusive).
    """
    drop = {b.lower() for b in exclude_buses}
    under_steps = 0
    over_steps = 0
    for gs in grid_states:
        has_under = False
        has_over = False
        for bus in gs.voltages.buses():
            if bus.lower() in drop:
                continue
            pv = gs.voltages[bus]
            for v in (pv.a, pv.b, pv.c):
                if math.isnan(v):
                    continue
                if v < v_min:
                    has_under = True
                if v > v_max:
                    has_over = True
            if has_under and has_over:
                break
        if has_under:
            under_steps += 1
        if has_over:
            over_steps += 1
    return float(under_steps), float(over_steps)


def _voltage_envelope(grid_states, *, exclude_buses: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return (vmin_t, vmax_t) per step for plotting."""
    drop = {b.lower() for b in exclude_buses}
    vmin = np.full(len(grid_states), np.inf)
    vmax = np.full(len(grid_states), -np.inf)
    for i, gs in enumerate(grid_states):
        for bus in gs.voltages.buses():
            if bus.lower() in drop:
                continue
            pv = gs.voltages[bus]
            for v in (pv.a, pv.b, pv.c):
                if math.isnan(v):
                    continue
                if v < vmin[i]:
                    vmin[i] = v
                if v > vmax[i]:
                    vmax[i] = v
    return vmin, vmax


def _voltage_envelope_by_zone(
    grid_states,
    *,
    zones: dict[str, list[str]],
    exclude_buses: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {zone_name: (vmin_t, vmax_t)} per step, one array pair per zone."""
    drop = {b.lower() for b in exclude_buses}
    zone_sets = {z: {b.lower() for b in buses} for z, buses in zones.items()}
    n = len(grid_states)
    vmin = {z: np.full(n, np.inf) for z in zones}
    vmax = {z: np.full(n, -np.inf) for z in zones}
    for i, gs in enumerate(grid_states):
        for bus in gs.voltages.buses():
            bl = bus.lower()
            if bl in drop:
                continue
            pv = gs.voltages[bus]
            for z, bset in zone_sets.items():
                if bl not in bset:
                    continue
                for v in (pv.a, pv.b, pv.c):
                    if math.isnan(v):
                        continue
                    if v < vmin[z][i]:
                        vmin[z][i] = v
                    if v > vmax[z][i]:
                        vmax[z][i] = v
    return {z: (vmin[z], vmax[z]) for z in zones}


def _zone_phase_integral(
    grid_states,
    *,
    zones: dict[str, list[str]],
    exclude_buses: tuple[str, ...],
    v_min: float,
    v_max: float,
) -> dict[str, dict[str, float]]:
    """Return {zone: {phase: integral_pu_s}} for violation breakdown.

    Integral uses the same linear formula as compute_allbus_voltage_stats:
    sum over t of (max(v_min-v,0) + max(v-v_max,0)) * dt.
    """
    drop = {b.lower() for b in exclude_buses}
    zone_sets = {z: {b.lower() for b in buses} for z, buses in zones.items()}
    acc: dict[str, dict[str, float]] = {z: {"a": 0.0, "b": 0.0, "c": 0.0} for z in zones}
    times = [gs.time_s for gs in grid_states]
    dt = (times[1] - times[0]) if len(times) > 1 else 1.0
    for gs in grid_states:
        for bus in gs.voltages.buses():
            bl = bus.lower()
            if bl in drop:
                continue
            pv = gs.voltages[bus]
            for z, bset in zone_sets.items():
                if bl not in bset:
                    continue
                for ph in ("a", "b", "c"):
                    v = getattr(pv, ph, float("nan"))
                    if math.isnan(v):
                        continue
                    viol = max(v_min - v, 0.0) + max(v - v_max, 0.0)
                    acc[z][ph] += viol * dt
    return acc


def _build_run_kwargs(
    *,
    base_exp: dict,
    scenario: dict,
    ofo_config: OFOConfig,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    save_dir: Path,
) -> dict:
    """Pack arguments for simulation.run_simulation."""
    return dict(
        sys=base_exp["sys"],
        dc_sites=scenario["dc_sites"],
        ofo_config=ofo_config,
        inference_data=inference_data,
        training_trace=training_trace,
        logistic_models=logistic_models,
        pv_systems=scenario["pv_systems"],
        time_varying_loads=scenario["tvl"],
        tap_schedule=None,  # baseline AND ofo run in 'no-tap' mode (matches run_baseline.py --mode no-tap)
        rule_based_config=None,
        ppo_model="",
        training_overlay=scenario["params"]["training_overlay"],
        save_dir=save_dir,
    )


def _plot_batch_sizes(
    records: list[ScenarioRecord],
    batch_data: dict[int, dict],
    save_path: Path,
    *,
    max_rows: int = 40,
) -> None:
    """Plot batch size over time per accepted scenario, baseline vs OFO, one row per scenario.

    When the library has more than ``max_rows`` scenarios, only the first
    ``max_rows`` are shown. A single tall figure of hundreds of rows quickly
    exceeds matplotlib's 65535-pixel dimension limit, so we cap here.
    """
    n = len(records)
    if n == 0:
        return
    if n > max_rows:
        logger.info("_plot_batch_sizes: capping at first %d of %d records", max_rows, n)
        records = records[:max_rows]
        n = max_rows

    # Collect all (site_id, label) columns from the first scenario. For
    # single-DC feeders (ieee13) there's one site; multi-DC feeders
    # (ieee34) get one column per (site, model) pair.
    first_seed = records[0].seed
    ofo_by_site = batch_data[first_seed]["ofo"]
    cols_meta: list[tuple[str, str]] = []
    for site_id, sdata in ofo_by_site.items():
        for label in sdata["batch_by_model"]:
            cols_meta.append((site_id, label))
    n_cols = len(cols_meta)

    fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 2.5 * n), sharex=True, squeeze=False)

    for row, rec in enumerate(records):
        bd = batch_data[rec.seed]
        for col, (site_id, label) in enumerate(cols_meta):
            ax = axes[row][col]
            bl_site = bd["baseline"][site_id]
            ofo_site = bd["ofo"][site_id]
            ax.plot(
                bl_site["time_s"],
                bl_site["batch_by_model"][label],
                color="#888",
                linewidth=0.7,
                alpha=0.7,
                label="baseline",
            )
            ax.plot(
                ofo_site["time_s"],
                ofo_site["batch_by_model"][label],
                color="#2196F3",
                linewidth=0.7,
                alpha=0.9,
                label="OFO",
            )
            if row == 0:
                short = label.split("/")[-1] if "/" in label else label
                title = f"{site_id}:{short}" if len(ofo_by_site) > 1 else short
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"seed={rec.seed}\nBatch", fontsize=8)
            ax.grid(True, alpha=0.2)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")

    for col in range(n_cols):
        axes[-1][col].set_xlabel("Time (s)")
    fig.suptitle("Accepted scenarios — batch size (baseline vs OFO)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _extract_batch_data(log) -> dict:
    """Extract time_s and per-model batch sizes from a simulation log.

    Returns {site_id: {"time_s": [...], "batch_by_model": {label: [bs]}}}.
    Uses ``log.dc_states_by_site`` (per-site lists) so multi-DC feeders
    like ieee34 don't get interleaved timestamps or alternating zeros.
    """
    per_site: dict[str, dict] = {}
    for site_id, states in log.dc_states_by_site.items():
        time_s = [s.time_s for s in states]
        labels: list[str] = []
        if states:
            for m in states[0].batch_size_by_model:
                if m not in labels:
                    labels.append(m)
        batch_by_model = {m: [s.batch_size_by_model.get(m, 0) for s in states] for m in labels}
        per_site[site_id] = {"time_s": time_s, "batch_by_model": batch_by_model}
    return per_site


def _is_always_minimum_batch(ofo_log, min_batch_size: int = 8, warmup_s: float = 10.0) -> bool:
    """Return True if every model stays at min_batch_size for the entire episode after warmup_s.

    Skips the first warmup_s seconds to allow for initial transient drops before OFO
    has had a chance to act (batch may be at minimum at t=0 before the first control step).
    """
    for states in ofo_log.dc_states_by_site.values():
        for s in states:
            if s.time_s <= warmup_s:
                continue
            for bs in s.batch_size_by_model.values():
                if bs > min_batch_size:
                    return False
    return True


def _plot_envelopes(
    records: list[ScenarioRecord],
    envelopes: dict,
    save_path: Path,
    *,
    total_duration_s: int,
    zones: dict[str, list[str]] | None = None,
    max_rows: int = 40,
) -> None:
    """Plot voltage envelope per accepted scenario, baseline vs OFO.

    When ``zones`` is provided (multi-zone feeders like ieee123), each scenario
    gets one subplot per zone showing the per-zone vmin/vmax band. Otherwise a
    single subplot with the global envelope is used.

    Caps at ``max_rows * 2`` records (global mode) or ``max_rows`` records
    (per-zone mode) to stay under matplotlib's 65535-pixel dimension limit.
    """
    n = len(records)
    if n == 0:
        return

    t = np.arange(total_duration_s)

    if zones:
        zone_names = list(zones.keys())
        n_zones = len(zone_names)
        cap = max_rows
        if n > cap:
            logger.info("_plot_envelopes: capping at first %d of %d records", cap, n)
            records = records[:cap]
            n = cap
        zone_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        fig, axes = plt.subplots(n, n_zones, figsize=(5 * n_zones, 3 * n), sharex=True, squeeze=False)
        for row, rec in enumerate(records):
            env = envelopes[rec.seed]
            for col, z in enumerate(zone_names):
                ax = axes[row][col]
                bl_z = env["baseline_zones"].get(z)
                of_z = env["ofo_zones"].get(z)
                color = zone_colors[col % len(zone_colors)]
                if bl_z is not None:
                    ax.fill_between(t, bl_z[0], bl_z[1], alpha=0.25, color="#888", label="baseline")
                if of_z is not None:
                    ax.fill_between(t, of_z[0], of_z[1], alpha=0.4, color=color, label="OFO")
                ax.axhline(V_MIN, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
                ax.axhline(V_MAX, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
                ax.grid(True, alpha=0.2)
                if row == 0:
                    ax.set_title(z, fontsize=10, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(
                        f"seed={rec.seed}\npv×{rec.pv_scale:.2f} ld×{rec.load_scale:.2f}\n"
                        f"bl={rec.baseline_integral:.1f} ofo={rec.ofo_integral:.1f} "
                        f"rec={rec.recovery_frac:.0%}",
                        fontsize=7,
                    )
                else:
                    ax.set_ylabel("V (pu)", fontsize=8)
                if row == 0 and col == 0:
                    ax.legend(loc="lower right", fontsize=7)
        for col in range(n_zones):
            axes[-1][col].set_xlabel("Time (s)", fontsize=8)
        fig.suptitle(
            "Accepted scenarios — per-zone voltage envelope (baseline vs OFO)",
            fontsize=13,
            fontweight="bold",
        )
    else:
        cap = max_rows * 2
        if n > cap:
            logger.info("_plot_envelopes: capping at first %d of %d records", cap, n)
            records = records[:cap]
            n = cap
        cols = 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3 * rows), sharex=True)
        axes = np.atleast_2d(axes)

        for idx, rec in enumerate(records):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            bmin, bmax = envelopes[rec.seed]["baseline"]
            omin, omax = envelopes[rec.seed]["ofo"]
            ax.fill_between(t, bmin, bmax, alpha=0.25, color="#888", label="baseline")
            ax.fill_between(t, omin, omax, alpha=0.4, color="#2196F3", label="OFO")
            ax.axhline(V_MIN, color="red", linestyle="--", linewidth=1, alpha=0.6)
            ax.axhline(V_MAX, color="red", linestyle="--", linewidth=1, alpha=0.6)
            ax.set_title(
                f"seed={rec.seed} pv×{rec.pv_scale:.2f} load×{rec.load_scale:.2f}\n"
                f"int: bl={rec.baseline_integral:.2f} ofo={rec.ofo_integral:.2f} "
                f"recov={rec.recovery_frac:.0%}",
                fontsize=9,
            )
            ax.set_ylabel("V (pu)", fontsize=9)
            ax.grid(True, alpha=0.2)
            if idx == 0:
                ax.legend(loc="lower right", fontsize=8)

        for k in range(n, rows * cols):
            r, c = divmod(k, cols)
            axes[r][c].axis("off")

        for c in range(cols):
            axes[-1][c].set_xlabel("Time (s)")
        fig.suptitle("Accepted scenarios — voltage envelope (baseline vs OFO)", fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(all_stats: list[dict], save_path: Path) -> None:
    """Bar chart of baseline vs OFO integral for every candidate (accepted + rejected)."""
    n = len(all_stats)
    if n == 0:
        return
    seeds = [s["seed"] for s in all_stats]
    bl = [s["baseline_integral"] for s in all_stats]
    of = [s["ofo_integral"] for s in all_stats]
    accepted = [s["accepted"] for s in all_stats]

    x = np.arange(n)
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * n), 5))
    ax.bar(x - w / 2, bl, w, color="#888", label="baseline integral")
    ax.bar(x + w / 2, of, w, color="#2196F3", label="OFO integral")
    for i, ok in enumerate(accepted):
        marker = "✓" if ok else "✗"
        color = "green" if ok else "red"
        ax.annotate(marker, xy=(i, max(bl[i], of[i])), ha="center", va="bottom", color=color, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], rotation=45)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Integral voltage violation (pu·s)")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title("Candidate scenarios — baseline vs OFO integral violation")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(
    *,
    system: str,
    n_candidates: int,
    seeds: tuple[int, ...] = (),
    min_recovery_frac: float,
    max_recovery_frac: float = 1.0,
    min_baseline_integral: float,
    min_baseline_integral_over: float,
    max_baseline_integral: float,
    pv_base_kw: float | None,
    tvl_base_kw: float | None,
    sensitivity_update_interval: int,
    ofo_w_throughput: float,
    tag: str,
    seed_start: int = 0,
    use_training_overlay: bool = True,
    randomize_ramps: bool = True,
    randomization_profile: bool = True,
    pv_scale_min: float = 0.5,
    pv_scale_max: float = 2.0,
    load_scale_min: float = 0.5,
    load_scale_max: float = 2.0,
    pv_t_shift_max_s: float = 0.0,
    tvl_t_shift_max_s: float = 0.0,
    pv_warp_min: float = 1.0,
    pv_warp_max: float = 1.0,
    tvl_warp_min: float = 1.0,
    tvl_warp_max: float = 1.0,
    overlay_prob: float = 1.0,
    overlay_intensity_min: float = 1.0,
    overlay_intensity_max: float = 1.0,
    overlay_gpu_frac_min: float = 0.85,
    overlay_gpu_frac_max: float = 1.0,
    n_ramps_per_site_choices: tuple[int, ...] = (1,),
    ramp_up_prob: float = 0.0,
    ramp_down_frac_min: float = 0.5,
    ramp_down_frac_max: float = 0.85,
    ramp_up_frac_min: float = 1.05,
    ramp_up_frac_max: float = 1.5,
    ramp_start_min: float = 500.0,
    ramp_start_max: float = 3000.0,
    ramp_dur_min: float = 300.0,
    ramp_dur_max: float = 800.0,
    randomize_pv_profile: bool = False,
    pv_shape_choices: tuple[str, ...] = (
        "flat",
        "rising_falling",
        "morning_ramp",
        "afternoon_decline",
        "midday_dip",
    ),
    pv_baseline_min: float = 0.75,
    pv_baseline_max: float = 0.95,
    pv_cloud_count_max: int = 3,
    pv_cloud_depth_min: float = 0.30,
    pv_cloud_depth_max: float = 0.70,
    pv_cloud_width_min: float = 60.0,
    pv_cloud_width_max: float = 300.0,
    randomize_tvl_profile: bool = False,
    tvl_shape_choices: tuple[str, ...] = ("flat", "increasing", "decreasing", "peaked", "valley"),
    max_always_min: int = 5,
    t_control_start_buffer: int = 200,
    t_control_end_buffer: int = 300,
    log_level: str,
    append_to: Path | None = None,
) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("openg2g.coordinator").setLevel(logging.WARNING)
    logging.getLogger("openg2g.datacenter").setLevel(logging.WARNING)
    logging.getLogger("openg2g.grid").setLevel(logging.WARNING)
    logging.getLogger("openg2g.controller.ofo").setLevel(logging.WARNING)
    logging.getLogger("controller_comparison").setLevel(logging.WARNING)

    if system not in EXPERIMENTS:
        raise ValueError(f"Unknown system '{system}'. Available: {list(EXPERIMENTS)}")
    out_dir = Path(__file__).resolve().parent / "outputs" / system / "scenario_library" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing scenario library to %s", out_dir)
    logger.info(
        "system=%s  use_training_overlay=%s  randomize_ramps=%s",
        system,
        use_training_overlay,
        randomize_ramps,
    )

    # ── Load data once (per-spec content-addressed cache under SPECS_CACHE_DIR) ──
    logger.info("Loading data...")
    base_exp = EXPERIMENTS[system]()
    sys_cfg = base_exp["sys"]
    exclude_buses = tuple(sys_cfg["exclude_buses"])
    zones: dict[str, list[str]] | None = sys_cfg.get("zones") or None

    # Override OFO sensitivity_update_interval and w_throughput so the
    # screening OFO matches the eval OFO exactly. 300 = re-estimate every
    # 5 simulated minutes, keeping the H-matrix fresh through the violation
    # window. w_throughput override ensures the screening OFO is laser-focused
    # on voltage (matching the eval pipeline value of 0.0001).
    base_ofo = base_exp["ofo_config"]
    ofo_config = base_ofo.model_copy(
        update={
            "sensitivity_update_interval": sensitivity_update_interval,
            "w_throughput": ofo_w_throughput,
        }
    )
    logger.info(
        "OFO config: sensitivity_update_interval=%d, w_throughput=%g (was %g)",
        sensitivity_update_interval,
        ofo_w_throughput,
        base_ofo.w_throughput,
    )

    # Override the base PV/TVL with caller-provided values, then let
    # randomize_scenario apply the [0.5, 2.0] scale factor on top.
    # When pv_base_kw/tvl_base_kw is None, keep the per-system defaults
    # from the experiment definition (this is the default for ieee34
    # since each bus has a distinct peak kW that we want to preserve).
    pv_systems_base = [
        PVSystemSpec(bus=p.bus, bus_kv=p.bus_kv, peak_kw=(pv_base_kw if pv_base_kw is not None else p.peak_kw))
        for p in base_exp["pv_systems"]
    ]
    tvl_base = [
        TimeVaryingLoadSpec(bus=t.bus, bus_kv=t.bus_kv, peak_kw=(tvl_base_kw if tvl_base_kw is not None else t.peak_kw))
        for t in base_exp["time_varying_loads"]
    ]
    logger.info(
        "PV base total=%.0f kW across %d systems, TVL base total=%.0f kW across %d loads (× scale ∈ [0.5, 2.0] per scenario)",  # noqa: E501
        sum(p.peak_kw for p in pv_systems_base),
        len(pv_systems_base),
        sum(t.peak_kw for t in tvl_base),
        len(tvl_base),
    )

    # All model specs across the single DC site
    all_specs = []
    for site in base_exp["dc_sites"].values():
        for md, _ in site.models:
            all_specs.append(md.spec)
    all_specs_tuple = tuple(all_specs)

    inference_data = InferenceData.ensure(
        SPECS_CACHE_DIR,
        all_specs_tuple,
        plot=False,
        dt_s=float(DT_DC),
    )
    training_trace = TrainingTrace.ensure(TRAINING_TRACE_PATH)
    logistic_models = LogisticModelStore.ensure(
        SPECS_CACHE_DIR,
        all_specs_tuple,
        plot=False,
    )

    # Training overlay base — same defaults as train_ppo._ieee13_experiment.
    # Only used when use_training_overlay is True (ieee13 default). For ieee34
    # the in-distribution scenarios deliberately skip the training overlay so
    # PV/TVL scales are the sole source of randomness (overlay is reserved for
    # OOD scenarios).
    if use_training_overlay:
        training_base = {
            "trace": training_trace,
            "n_gpus": 2400,
            "target_peak_W_per_gpu": 400.0,
            "t_start": 1000.0,
            "t_end": 2000.0,
        }
    else:
        training_base = None

    accepted: list[ScenarioRecord] = []
    envelopes: dict[int, dict] = {}
    batch_data: dict[int, dict] = {}
    all_stats: list[dict] = []
    n_always_min_accepted: int = 0
    MAX_ALWAYS_MIN = max_always_min

    seed_list = list(seeds) if seeds else [(seed_start + i) * 1000 + 7 for i in range(n_candidates)]
    n_total = len(seed_list)
    for cand_idx, effective_seed in enumerate(seed_list):
        logger.info("=" * 60)
        logger.info("Candidate %d/%d (seed=%d)", cand_idx + 1, n_total, effective_seed)

        scenario = randomize_scenario(
            seed=effective_seed,
            dc_sites_base=base_exp["dc_sites"],
            pv_systems_base=pv_systems_base,
            tvl_base=tvl_base,
            training_base=training_base,
            randomize_ramps=randomize_ramps,
            randomization_profile=randomization_profile,
            pv_scale_min=pv_scale_min,
            pv_scale_max=pv_scale_max,
            load_scale_min=load_scale_min,
            load_scale_max=load_scale_max,
            pv_t_shift_max_s=pv_t_shift_max_s,
            tvl_t_shift_max_s=tvl_t_shift_max_s,
            pv_warp_min=pv_warp_min,
            pv_warp_max=pv_warp_max,
            tvl_warp_min=tvl_warp_min,
            tvl_warp_max=tvl_warp_max,
            overlay_prob=overlay_prob,
            overlay_intensity_min=overlay_intensity_min,
            overlay_intensity_max=overlay_intensity_max,
            overlay_gpu_frac_min=overlay_gpu_frac_min,
            overlay_gpu_frac_max=overlay_gpu_frac_max,
            n_ramps_per_site_choices=tuple(n_ramps_per_site_choices),
            ramp_up_prob=ramp_up_prob,
            ramp_down_frac_min=ramp_down_frac_min,
            ramp_down_frac_max=ramp_down_frac_max,
            ramp_up_frac_min=ramp_up_frac_min,
            ramp_up_frac_max=ramp_up_frac_max,
            ramp_start_min=ramp_start_min,
            ramp_start_max=ramp_start_max,
            ramp_dur_min=ramp_dur_min,
            ramp_dur_max=ramp_dur_max,
            randomize_pv_profile=randomize_pv_profile,
            pv_shape_choices=tuple(pv_shape_choices),
            pv_baseline_min=pv_baseline_min,
            pv_baseline_max=pv_baseline_max,
            pv_cloud_count_max=pv_cloud_count_max,
            pv_cloud_depth_min=pv_cloud_depth_min,
            pv_cloud_depth_max=pv_cloud_depth_max,
            pv_cloud_width_min=pv_cloud_width_min,
            pv_cloud_width_max=pv_cloud_width_max,
            randomize_tvl_profile=randomize_tvl_profile,
            tvl_shape_choices=tuple(tvl_shape_choices),
        )

        # Build run kwargs once
        run_kwargs = _build_run_kwargs(
            base_exp=base_exp,
            scenario=scenario,
            ofo_config=ofo_config,
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            save_dir=out_dir,
        )

        # ── baseline ──
        bl_stats, bl_log = run_simulation("baseline", **run_kwargs)
        baseline_pen = _per_step_voltage_pen(bl_log.grid_states, v_min=V_MIN, v_max=V_MAX, exclude_buses=exclude_buses)
        baseline_int = float(bl_stats.integral_violation_pu_s)
        baseline_viol_t = float(bl_stats.violation_time_s)
        bl_under_t, bl_over_t = _under_over_voltage_time(
            bl_log.grid_states, v_min=V_MIN, v_max=V_MAX, exclude_buses=exclude_buses
        )

        # ── OFO ──
        ofo_stats, ofo_log = run_simulation("ofo", **run_kwargs)
        ofo_pen = _per_step_voltage_pen(ofo_log.grid_states, v_min=V_MIN, v_max=V_MAX, exclude_buses=exclude_buses)
        ofo_int = float(ofo_stats.integral_violation_pu_s)
        ofo_viol_t = float(ofo_stats.violation_time_s)
        ofo_under_t, ofo_over_t = _under_over_voltage_time(
            ofo_log.grid_states, v_min=V_MIN, v_max=V_MAX, exclude_buses=exclude_buses
        )

        recovery = (baseline_int - ofo_int) / baseline_int if baseline_int > 0 else 0.0
        is_pure_overvoltage = bl_over_t > 0 and bl_under_t == 0
        is_mixed = bl_over_t > 0 and bl_under_t > 0
        effective_min_integral = (
            min_baseline_integral_over if (is_pure_overvoltage or is_mixed) else min_baseline_integral
        )
        passes = (
            baseline_int >= effective_min_integral
            and baseline_int <= max_baseline_integral
            and recovery >= min_recovery_frac
            and recovery <= max_recovery_frac
        )

        logger.info(
            "  baseline: int=%.3f viol=%.0fs (under=%.0fs over=%.0fs) | "
            "ofo: int=%.3f viol=%.0fs (under=%.0fs over=%.0fs) | recovery=%.0f%% | %s",
            baseline_int,
            baseline_viol_t,
            bl_under_t,
            bl_over_t,
            ofo_int,
            ofo_viol_t,
            ofo_under_t,
            ofo_over_t,
            100 * recovery,
            "ACCEPT" if passes else "reject",
        )

        all_stats.append(
            dict(
                seed=effective_seed,
                pv_scale=scenario["params"]["pv_scale"],
                load_scale=scenario["params"]["load_scale"],
                training_overlay=scenario["params"]["training_overlay"],
                baseline_integral=baseline_int,
                ofo_integral=ofo_int,
                baseline_violation_time_s=baseline_viol_t,
                ofo_violation_time_s=ofo_viol_t,
                bl_undervoltage_time_s=bl_under_t,
                bl_overvoltage_time_s=bl_over_t,
                ofo_undervoltage_time_s=ofo_under_t,
                ofo_overvoltage_time_s=ofo_over_t,
                recovery_frac=recovery,
                accepted=passes,
            )
        )

        # Log per-zone, per-phase baseline integral breakdown for every candidate
        # (especially useful for diagnosing rejected scenarios in multi-zone feeders).
        if zones:
            bl_zone_phase = _zone_phase_integral(
                bl_log.grid_states,
                zones=zones,
                exclude_buses=exclude_buses,
                v_min=V_MIN,
                v_max=V_MAX,
            )
            zone_summary = "  baseline zone/phase integral: " + " | ".join(
                f"{z}: A={bl_zone_phase[z]['a']:.1f} B={bl_zone_phase[z]['b']:.1f} C={bl_zone_phase[z]['c']:.1f}"
                for z in zones
            )
            logger.info(zone_summary)

        if passes:
            always_min = _is_always_minimum_batch(ofo_log)
            if always_min:
                if n_always_min_accepted >= MAX_ALWAYS_MIN:
                    logger.info(
                        "  seed=%d: OFO always at min batch — cap reached (%d/%d), skipping",
                        effective_seed,
                        n_always_min_accepted,
                        MAX_ALWAYS_MIN,
                    )
                    passes = False
                else:
                    n_always_min_accepted += 1
                    logger.info(
                        "  seed=%d: OFO always at min batch — accepting (%d/%d)",
                        effective_seed,
                        n_always_min_accepted,
                        MAX_ALWAYS_MIN,
                    )
        if passes:
            nonzero_steps = np.nonzero(baseline_pen > 0)[0]
            if len(nonzero_steps) > 0:
                t_first = int(nonzero_steps[0])
                t_last = int(nonzero_steps[-1])
            else:
                t_first = 0
                t_last = len(baseline_pen)
            t_ctrl_start = max(0, t_first - t_control_start_buffer)
            t_ctrl_end = min(len(baseline_pen), t_last + t_control_end_buffer)
            logger.info(
                "  control window: violation [%d, %d] → [%d, %d] (%d steps, saves %d%%)",
                t_first,
                t_last,
                t_ctrl_start,
                t_ctrl_end,
                t_ctrl_end - t_ctrl_start,
                100 * (len(baseline_pen) - (t_ctrl_end - t_ctrl_start)) // len(baseline_pen),
            )
            rec = ScenarioRecord(
                seed=effective_seed,
                pv_scale=scenario["params"]["pv_scale"],
                load_scale=scenario["params"]["load_scale"],
                training_overlay=scenario["params"]["training_overlay"],
                baseline_integral=baseline_int,
                ofo_integral=ofo_int,
                baseline_violation_time_s=baseline_viol_t,
                ofo_violation_time_s=ofo_viol_t,
                recovery_frac=recovery,
                bl_undervoltage_time_s=bl_under_t,
                bl_overvoltage_time_s=bl_over_t,
                ofo_voltage_pen_per_step=ofo_pen,
                baseline_voltage_pen_per_step=baseline_pen,
                t_control_start=t_ctrl_start,
                t_control_end=t_ctrl_end,
                resolved_dc_sites=scenario["dc_sites"],
                resolved_pv_systems=tuple(scenario["pv_systems"]),
                resolved_tvl=tuple(scenario["tvl"]),
            )
            accepted.append(rec)
            env_entry: dict = {
                "baseline": _voltage_envelope(bl_log.grid_states, exclude_buses=exclude_buses),
                "ofo": _voltage_envelope(ofo_log.grid_states, exclude_buses=exclude_buses),
            }
            if zones:
                env_entry["baseline_zones"] = _voltage_envelope_by_zone(
                    bl_log.grid_states,
                    zones=zones,
                    exclude_buses=exclude_buses,
                )
                env_entry["ofo_zones"] = _voltage_envelope_by_zone(
                    ofo_log.grid_states,
                    zones=zones,
                    exclude_buses=exclude_buses,
                )
            envelopes[effective_seed] = env_entry
            batch_data[effective_seed] = {
                "baseline": _extract_batch_data(bl_log),
                "ofo": _extract_batch_data(ofo_log),
            }

    # ── Save artifacts ──
    library_path = out_dir / "library.pkl"
    randomize_kwargs = {
        "randomize_ramps": randomize_ramps,
        "randomization_profile": randomization_profile,
        "pv_scale_min": pv_scale_min,
        "pv_scale_max": pv_scale_max,
        "load_scale_min": load_scale_min,
        "load_scale_max": load_scale_max,
        "pv_t_shift_max_s": pv_t_shift_max_s,
        "tvl_t_shift_max_s": tvl_t_shift_max_s,
        "pv_warp_min": pv_warp_min,
        "pv_warp_max": pv_warp_max,
        "tvl_warp_min": tvl_warp_min,
        "tvl_warp_max": tvl_warp_max,
        "overlay_prob": overlay_prob,
        "overlay_intensity_min": overlay_intensity_min,
        "overlay_intensity_max": overlay_intensity_max,
        "overlay_gpu_frac_min": overlay_gpu_frac_min,
        "overlay_gpu_frac_max": overlay_gpu_frac_max,
        "n_ramps_per_site_choices": tuple(n_ramps_per_site_choices),
        "ramp_up_prob": ramp_up_prob,
        "ramp_down_frac_min": ramp_down_frac_min,
        "ramp_down_frac_max": ramp_down_frac_max,
        "ramp_up_frac_min": ramp_up_frac_min,
        "ramp_up_frac_max": ramp_up_frac_max,
        "ramp_start_min": ramp_start_min,
        "ramp_start_max": ramp_start_max,
        "ramp_dur_min": ramp_dur_min,
        "ramp_dur_max": ramp_dur_max,
        "randomize_pv_profile": randomize_pv_profile,
        "pv_shape_choices": tuple(pv_shape_choices),
        "pv_baseline_min": pv_baseline_min,
        "pv_baseline_max": pv_baseline_max,
        "pv_cloud_count_max": pv_cloud_count_max,
        "pv_cloud_depth_min": pv_cloud_depth_min,
        "pv_cloud_depth_max": pv_cloud_depth_max,
        "pv_cloud_width_min": pv_cloud_width_min,
        "pv_cloud_width_max": pv_cloud_width_max,
        "randomize_tvl_profile": randomize_tvl_profile,
        "tvl_shape_choices": tuple(tvl_shape_choices),
    }

    with open(library_path, "wb") as f:
        pickle.dump(
            {
                "scenarios": accepted,
                "config": {
                    "system": system,
                    "n_candidates": n_total,
                    "min_recovery_frac": min_recovery_frac,
                    "max_recovery_frac": max_recovery_frac,
                    "min_baseline_integral": min_baseline_integral,
                    "pv_base_kw": pv_base_kw,
                    "tvl_base_kw": tvl_base_kw,
                    "use_training_overlay": use_training_overlay,
                    "randomize_ramps": randomize_ramps,
                    "randomize_kwargs": randomize_kwargs,
                    "pv_systems_base": [
                        {"bus": p.bus, "bus_kv": p.bus_kv, "peak_kw": p.peak_kw} for p in pv_systems_base
                    ],
                    "tvl_base": [{"bus": t.bus, "bus_kv": t.bus_kv, "peak_kw": t.peak_kw} for t in tvl_base],
                    "v_min": V_MIN,
                    "v_max": V_MAX,
                },
            },
            f,
        )
    logger.info("Wrote %d accepted scenarios to %s", len(accepted), library_path)

    # ── Optional: merge into an existing library ──
    if append_to is not None:
        if not append_to.exists():
            raise FileNotFoundError(f"--append-to target not found: {append_to}")
        with open(append_to, "rb") as f:
            existing = pickle.load(f)
        existing_seeds = {s.seed for s in existing["scenarios"]}
        new_scenarios = [s for s in accepted if s.seed not in existing_seeds]
        if new_scenarios:
            existing["scenarios"] = existing["scenarios"] + new_scenarios
            with open(append_to, "wb") as f:
                pickle.dump(existing, f)
            logger.info(
                "Appended %d new scenario(s) to %s (total now: %d)",
                len(new_scenarios),
                append_to,
                len(existing["scenarios"]),
            )
        else:
            logger.info("No new scenarios to append — all seeds already present in %s", append_to)

    csv_path = out_dir / "candidates.csv"
    with open(csv_path, "w", newline="") as f:
        cols = [
            "seed",
            "pv_scale",
            "load_scale",
            "baseline_integral",
            "ofo_integral",
            "baseline_violation_time_s",
            "ofo_violation_time_s",
            "bl_undervoltage_time_s",
            "bl_overvoltage_time_s",
            "ofo_undervoltage_time_s",
            "ofo_overvoltage_time_s",
            "recovery_frac",
            "accepted",
        ]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for s in all_stats:
            writer.writerow({k: s[k] for k in cols})
    logger.info("Wrote candidate stats to %s", csv_path)

    # Plots
    if accepted:
        _plot_envelopes(
            accepted,
            envelopes,
            out_dir / "scenario_envelopes.png",
            total_duration_s=len(accepted[0].ofo_voltage_pen_per_step),
            zones=zones,
        )
        logger.info("Wrote envelope plot to %s", out_dir / "scenario_envelopes.png")
        _plot_batch_sizes(accepted, batch_data, out_dir / "scenario_batch_sizes.png")
        logger.info("Wrote batch-size plot to %s", out_dir / "scenario_batch_sizes.png")
    _plot_summary(all_stats, out_dir / "scenario_summary.png")
    logger.info("Wrote summary plot to %s", out_dir / "scenario_summary.png")

    # Headline
    n_acc = len(accepted)
    accept_rate = n_acc / max(1, n_total)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Library build complete: %d/%d accepted (%.0f%%)", n_acc, n_total, 100 * accept_rate)
    if n_acc > 0:
        recs = np.array([r.recovery_frac for r in accepted])
        bls = np.array([r.baseline_integral for r in accepted])
        logger.info("  recovery (mean ± std): %.2f ± %.2f", recs.mean(), recs.std())
        logger.info("  baseline integral (mean ± std): %.2f ± %.2f", bls.mean(), bls.std())


# ── Optional supplement-phase helpers (used by the CLI when --n-supplement-candidates > 0) ──


def _is_pure_overvoltage(rec) -> bool:
    """Classify a ScenarioRecord as pure-overvoltage (over > 0, under == 0).

    Uses the saved per-record fields when present (newer builds); falls back to
    the load_scale heuristic for old builds where the fields weren't written.
    """
    if "bl_overvoltage_time_s" in rec.__dict__:
        return rec.bl_overvoltage_time_s > 0 and rec.bl_undervoltage_time_s == 0
    return getattr(rec, "load_scale", 99.0) <= 1.0  # rough proxy: low load implies overvoltage


def _merge_supplement_into_base(
    base_pkl: Path,
    supp_pkl: Path,
    swap_undervoltage_for_overvoltage: int = 0,
) -> None:
    """Merge pure-overvoltage scenarios from a supplement library into a base library.

    Train mode (swap_undervoltage_for_overvoltage = 0): append ALL pure-overvoltage
    scenarios from the supplement to the base. Library size grows.

    Test mode (swap_undervoltage_for_overvoltage > 0): take up to N pure-overvoltage
    from the supplement, remove the same number of undervoltage-only scenarios
    (highest load_scale first) from the base. Library size unchanged.
    """
    if not base_pkl.exists():
        raise FileNotFoundError(f"--n-supplement-candidates set but base library missing: {base_pkl}")
    if not supp_pkl.exists():
        raise FileNotFoundError(f"supplement library missing: {supp_pkl}")

    with open(base_pkl, "rb") as f:
        base_lib = pickle.load(f)
    with open(supp_pkl, "rb") as f:
        supp_lib = pickle.load(f)

    pure_over = [r for r in supp_lib["scenarios"] if _is_pure_overvoltage(r)]
    base_seeds = {r.seed for r in base_lib["scenarios"]}
    pure_over_new = [r for r in pure_over if r.seed not in base_seeds]
    logger.info(
        "Supplement merge: %d/%d supplement scenarios are pure-overvoltage (%d new vs base)",
        len(pure_over),
        len(supp_lib["scenarios"]),
        len(pure_over_new),
    )

    if swap_undervoltage_for_overvoltage <= 0:
        new_scenarios = list(base_lib["scenarios"]) + pure_over_new
    else:
        n_take = min(swap_undervoltage_for_overvoltage, len(pure_over_new))
        recs = list(base_lib["scenarios"])
        under_only_idx = [
            i
            for i, r in enumerate(recs)
            if getattr(r, "bl_undervoltage_time_s", None) is not None
            and r.bl_undervoltage_time_s > 0
            and getattr(r, "bl_overvoltage_time_s", 0) == 0
        ]
        if not under_only_idx:
            under_only_idx = [i for i, r in enumerate(recs) if getattr(r, "load_scale", 0) > 2.0]
        under_only_idx.sort(key=lambda i: getattr(recs[i], "load_scale", 0), reverse=True)
        remove_idx = set(under_only_idx[:n_take])
        if remove_idx:
            ls_min = min(getattr(recs[i], "load_scale", 0) for i in remove_idx)
            ls_max = max(getattr(recs[i], "load_scale", 0) for i in remove_idx)
            logger.info(
                "  test-set swap: removing %d undervoltage-only scenarios (load_scale %.2f-%.2f)",
                len(remove_idx),
                ls_min,
                ls_max,
            )
        new_scenarios = [r for i, r in enumerate(recs) if i not in remove_idx] + pure_over_new[:n_take]

    base_lib["scenarios"] = new_scenarios
    with open(base_pkl, "wb") as f:
        pickle.dump(base_lib, f)
    logger.info("Wrote merged library to %s (%d scenarios total)", base_pkl, len(new_scenarios))


if __name__ == "__main__":
    from dataclasses import dataclass
    from typing import Annotated

    import tyro

    @dataclass
    class Args:
        system: str = "ieee13"
        """Which feeder experiment to use. Valid: ieee13, ieee34, ieee123."""
        n_candidates: int = 20
        """Number of randomized candidate scenarios to evaluate."""
        min_recovery_frac: float = 0.7
        """Reject scenarios where OFO recovers less than this fraction of the baseline integral violation."""
        max_recovery_frac: float = 1.0
        """Reject scenarios where OFO recovers more than this fraction (use with min to select a recovery band, e.g. 0.4-0.6)."""  # noqa: E501
        min_baseline_integral: float = 0.2
        """Reject scenarios where the baseline undervoltage integral is below this threshold (no learning signal)."""
        min_baseline_integral_over: float = 0.01
        """Reject pure-overvoltage scenarios where the baseline integral is below this threshold. Lower than min_baseline_integral since OFO recovers overvoltage less aggressively."""  # noqa: E501
        max_baseline_integral: float = 1e9
        """Reject scenarios where the baseline integral violation exceeds this threshold (saturated, dominate gradients). Default = no upper bound."""  # noqa: E501
        pv_base_kw: float | None = None
        """Base PV peak power per system (kW) before per-scenario scaling. If unset, use the per-bus defaults from the experiment definition (recommended for ieee34 where each PV has a distinct peak_kw)."""  # noqa: E501
        tvl_base_kw: float | None = None
        """Base time-varying load peak power per load (kW) before per-scenario scaling. If unset, use the per-bus defaults from the experiment definition."""  # noqa: E501
        sensitivity_update_interval: int = 300
        """OFO H-matrix re-estimation interval in control steps. 300 = every 5 simulated minutes."""
        ofo_w_throughput: float = 0.0001
        """OFO throughput weight in the primal objective. 0 = pure voltage focus (the screening only cares how much violation OFO can recover)."""  # noqa: E501
        tag: str = "v3"
        """Subdirectory under outputs/<system>/scenario_library/ to write artifacts to."""
        seed_start: int = 0
        """Starting candidate index. Effective seed = (seed_start + cand_idx) * 1000 + 7. Use to generate non-overlapping scenario sets (e.g., seed-start=400 for eval set when training used 0-349)."""  # noqa: E501
        seeds: tuple[int, ...] = ()
        """Explicit list of effective seeds to run (overrides seed_start + n_candidates). Use to re-run only known-accepted seeds from a prior build log."""  # noqa: E501
        use_training_overlay: bool = True
        """Whether to add the training overlay (a few-hundred-kW training GPU spike) to in-distribution scenarios. Default True (ieee13 behavior). Set --no-use-training-overlay for ieee34 in-dist libraries — the overlay is reserved for OOD there."""  # noqa: E501
        randomize_ramps: bool = True
        """Whether randomize_scenario synthesizes per-episode inference ramps. Default True (ieee13 behavior). Set --no-randomize-ramps for ieee34 in-dist libraries — ramps are reserved for OOD there."""  # noqa: E501
        # ── Randomization profile ──
        randomization_profile: Annotated[bool, tyro.conf.FlagConversionOff] = True
        """True (broad, default) = bidirectional multi-ramps, wider PV/TVL scales, shape randomness, stochastic overlay. False (narrow) = legacy behavior."""  # noqa: E501
        pv_scale_min: float = 0.5
        pv_scale_max: float = 2.0
        load_scale_min: float = 0.5
        load_scale_max: float = 2.0
        pv_t_shift_max_s: float = 0.0
        """Max abs PV profile peak-time shift in seconds (broad profile only). 0 disables shape randomness on PV."""
        tvl_t_shift_max_s: float = 0.0
        """Max abs TVL profile peak-time shift in seconds (broad profile only)."""
        pv_warp_min: float = 1.0
        pv_warp_max: float = 1.0
        tvl_warp_min: float = 1.0
        tvl_warp_max: float = 1.0
        overlay_prob: float = 1.0
        """Probability the training overlay is applied to a candidate (broad profile only)."""
        overlay_intensity_min: float = 1.0
        overlay_intensity_max: float = 1.0
        overlay_gpu_frac_min: float = 0.85
        overlay_gpu_frac_max: float = 1.0
        n_ramps_per_site_choices: tuple[int, ...] = (1,)
        """Possible numbers of ramps per DC site (broad profile). e.g. (1, 2) gives 50/50 single/double ramps."""
        ramp_up_prob: float = 0.0
        """Probability a sampled ramp goes UP (broad profile). GPU capacity is checked per-site so up-ramps are clamped automatically."""  # noqa: E501
        ramp_down_frac_min: float = 0.5
        ramp_down_frac_max: float = 0.85
        ramp_up_frac_min: float = 1.05
        ramp_up_frac_max: float = 1.5
        ramp_start_min: float = 500.0
        """Earliest time (s) a ramp can begin (broad mode). Lower this to allow ramps near episode start."""
        ramp_start_max: float = 3000.0
        """Latest time (s) a ramp can begin."""
        ramp_dur_min: float = 300.0
        """Minimum ramp duration (s). Lower for fast/sharp ramps."""
        ramp_dur_max: float = 800.0
        """Maximum ramp duration (s). Higher for slow/gentle ramps."""
        randomize_pv_profile: bool = False
        """Replace the analytical pv_profile_kw with a multi-shape random PV profile (broad mode only)."""
        pv_shape_choices: tuple[str, ...] = (
            "flat",
            "rising_falling",
            "morning_ramp",
            "afternoon_decline",
            "midday_dip",
        )
        """PV shapes to sample from (broad mode + --randomize-pv-profile). Default covers all 5 transient/flat envelopes."""  # noqa: E501
        pv_baseline_min: float = 0.75
        pv_baseline_max: float = 0.95
        pv_cloud_count_max: int = 3
        pv_cloud_depth_min: float = 0.30
        pv_cloud_depth_max: float = 0.70
        pv_cloud_width_min: float = 60.0
        pv_cloud_width_max: float = 300.0
        randomize_tvl_profile: bool = False
        """Replace the analytical load_profile_kw with one of {flat, increasing, decreasing, peaked, valley} per scenario (broad mode only)."""  # noqa: E501
        tvl_shape_choices: tuple[str, ...] = ("flat", "increasing", "decreasing", "peaked", "valley")
        max_always_min: int = 5
        """Max number of 'OFO always at min batch' scenarios to accept. Increase to 10 for training libraries to avoid under-representing easy scenarios."""  # noqa: E501
        t_control_start_buffer: int = 200
        """Steps of buffer before the first baseline voltage violation (for episode truncation)."""
        t_control_end_buffer: int = 300
        """Steps of buffer after the last baseline voltage violation (for episode truncation)."""
        log_level: str = "INFO"
        append_to: Path | None = None
        """If set, merge newly accepted scenarios into this existing library.pkl (deduplicates by seed). The new run is still saved under --tag for diagnostic plots."""  # noqa: E501
        # ── Optional supplement phase (paper-balanced library workflow) ──
        n_supplement_candidates: int = 0
        """If > 0, after the main build run a second pass with overvoltage-favouring params (high PV, low load) and merge pure-overvoltage scenarios into the base library. Replaces the standalone build_balanced_library_ieee13.py orchestrator."""  # noqa: E501
        supplement_seed_start: int = -1
        """Starting candidate index for the supplement pass. Default -1 means seed_start + 4000 (prevents seed overlap)."""  # noqa: E501
        supplement_pv_scale_min: float = 2.0
        supplement_pv_scale_max: float = 5.0
        supplement_load_scale_min: float = 0.5
        supplement_load_scale_max: float = 1.0
        supplement_min_baseline_integral_over: float = 1.0
        """Tighter overvoltage-integral floor for the supplement pass (we want real overvoltage scenarios, not borderline ones)."""  # noqa: E501
        supplement_tag: str = ""
        """Subdirectory for the supplement build artifacts. Default empty means f'{tag}_over_supp'."""
        swap_undervoltage_for_overvoltage: int = 0
        """If > 0, after the supplement merge swap N undervoltage-only scenarios out of the base for N pure-overvoltage scenarios from the supplement (test-set mode that keeps library size constant). Default 0 = train mode (append all pure-over)."""  # noqa: E501

    args = tyro.cli(Args)
    main(
        system=args.system,
        n_candidates=args.n_candidates,
        min_recovery_frac=args.min_recovery_frac,
        max_recovery_frac=args.max_recovery_frac,
        min_baseline_integral=args.min_baseline_integral,
        min_baseline_integral_over=args.min_baseline_integral_over,
        max_baseline_integral=args.max_baseline_integral,
        pv_base_kw=args.pv_base_kw,
        tvl_base_kw=args.tvl_base_kw,
        sensitivity_update_interval=args.sensitivity_update_interval,
        ofo_w_throughput=args.ofo_w_throughput,
        tag=args.tag,
        seed_start=args.seed_start,
        seeds=args.seeds,
        use_training_overlay=args.use_training_overlay,
        randomize_ramps=args.randomize_ramps,
        randomization_profile=args.randomization_profile,
        pv_scale_min=args.pv_scale_min,
        pv_scale_max=args.pv_scale_max,
        load_scale_min=args.load_scale_min,
        load_scale_max=args.load_scale_max,
        pv_t_shift_max_s=args.pv_t_shift_max_s,
        tvl_t_shift_max_s=args.tvl_t_shift_max_s,
        pv_warp_min=args.pv_warp_min,
        pv_warp_max=args.pv_warp_max,
        tvl_warp_min=args.tvl_warp_min,
        tvl_warp_max=args.tvl_warp_max,
        overlay_prob=args.overlay_prob,
        overlay_intensity_min=args.overlay_intensity_min,
        overlay_intensity_max=args.overlay_intensity_max,
        overlay_gpu_frac_min=args.overlay_gpu_frac_min,
        overlay_gpu_frac_max=args.overlay_gpu_frac_max,
        n_ramps_per_site_choices=args.n_ramps_per_site_choices,
        ramp_up_prob=args.ramp_up_prob,
        ramp_down_frac_min=args.ramp_down_frac_min,
        ramp_down_frac_max=args.ramp_down_frac_max,
        ramp_up_frac_min=args.ramp_up_frac_min,
        ramp_up_frac_max=args.ramp_up_frac_max,
        ramp_start_min=args.ramp_start_min,
        ramp_start_max=args.ramp_start_max,
        ramp_dur_min=args.ramp_dur_min,
        ramp_dur_max=args.ramp_dur_max,
        randomize_pv_profile=args.randomize_pv_profile,
        pv_shape_choices=args.pv_shape_choices,
        pv_baseline_min=args.pv_baseline_min,
        pv_baseline_max=args.pv_baseline_max,
        pv_cloud_count_max=args.pv_cloud_count_max,
        pv_cloud_depth_min=args.pv_cloud_depth_min,
        pv_cloud_depth_max=args.pv_cloud_depth_max,
        pv_cloud_width_min=args.pv_cloud_width_min,
        pv_cloud_width_max=args.pv_cloud_width_max,
        randomize_tvl_profile=args.randomize_tvl_profile,
        tvl_shape_choices=args.tvl_shape_choices,
        max_always_min=args.max_always_min,
        t_control_start_buffer=args.t_control_start_buffer,
        t_control_end_buffer=args.t_control_end_buffer,
        log_level=args.log_level,
        append_to=args.append_to,
    )

    # ── Optional supplement phase: second pass with overvoltage-favouring params + merge ──
    if args.n_supplement_candidates > 0:
        supp_tag = args.supplement_tag or f"{args.tag}_over_supp"
        supp_seed_start = args.supplement_seed_start if args.supplement_seed_start >= 0 else args.seed_start + 4000

        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "Supplement phase: %d candidates, seed_start=%d, tag=%s",
            args.n_supplement_candidates,
            supp_seed_start,
            supp_tag,
        )
        logger.info(
            "  PV scale [%.2f-%.2f], load scale [%.2f-%.2f] (overvoltage-favouring)",
            args.supplement_pv_scale_min,
            args.supplement_pv_scale_max,
            args.supplement_load_scale_min,
            args.supplement_load_scale_max,
        )
        logger.info("=" * 60)

        main(
            system=args.system,
            n_candidates=args.n_supplement_candidates,
            min_recovery_frac=args.min_recovery_frac,
            max_recovery_frac=args.max_recovery_frac,
            min_baseline_integral=args.min_baseline_integral,
            min_baseline_integral_over=args.supplement_min_baseline_integral_over,
            max_baseline_integral=args.max_baseline_integral,
            pv_base_kw=args.pv_base_kw,
            tvl_base_kw=args.tvl_base_kw,
            sensitivity_update_interval=args.sensitivity_update_interval,
            ofo_w_throughput=args.ofo_w_throughput,
            tag=supp_tag,
            seed_start=supp_seed_start,
            seeds=(),
            use_training_overlay=args.use_training_overlay,
            randomize_ramps=args.randomize_ramps,
            randomization_profile=args.randomization_profile,
            pv_scale_min=args.supplement_pv_scale_min,
            pv_scale_max=args.supplement_pv_scale_max,
            load_scale_min=args.supplement_load_scale_min,
            load_scale_max=args.supplement_load_scale_max,
            pv_t_shift_max_s=args.pv_t_shift_max_s,
            tvl_t_shift_max_s=args.tvl_t_shift_max_s,
            pv_warp_min=args.pv_warp_min,
            pv_warp_max=args.pv_warp_max,
            tvl_warp_min=args.tvl_warp_min,
            tvl_warp_max=args.tvl_warp_max,
            overlay_prob=args.overlay_prob,
            overlay_intensity_min=args.overlay_intensity_min,
            overlay_intensity_max=args.overlay_intensity_max,
            overlay_gpu_frac_min=args.overlay_gpu_frac_min,
            overlay_gpu_frac_max=args.overlay_gpu_frac_max,
            n_ramps_per_site_choices=args.n_ramps_per_site_choices,
            ramp_up_prob=args.ramp_up_prob,
            ramp_down_frac_min=args.ramp_down_frac_min,
            ramp_down_frac_max=args.ramp_down_frac_max,
            ramp_up_frac_min=args.ramp_up_frac_min,
            ramp_up_frac_max=args.ramp_up_frac_max,
            ramp_start_min=args.ramp_start_min,
            ramp_start_max=args.ramp_start_max,
            ramp_dur_min=args.ramp_dur_min,
            ramp_dur_max=args.ramp_dur_max,
            randomize_pv_profile=args.randomize_pv_profile,
            pv_shape_choices=args.pv_shape_choices,
            pv_baseline_min=args.pv_baseline_min,
            pv_baseline_max=args.pv_baseline_max,
            pv_cloud_count_max=args.pv_cloud_count_max,
            pv_cloud_depth_min=args.pv_cloud_depth_min,
            pv_cloud_depth_max=args.pv_cloud_depth_max,
            pv_cloud_width_min=args.pv_cloud_width_min,
            pv_cloud_width_max=args.pv_cloud_width_max,
            randomize_tvl_profile=args.randomize_tvl_profile,
            tvl_shape_choices=args.tvl_shape_choices,
            max_always_min=args.max_always_min,
            t_control_start_buffer=args.t_control_start_buffer,
            t_control_end_buffer=args.t_control_end_buffer,
            log_level=args.log_level,
            append_to=None,  # we run a separate merge below (with the pure-over filter)
        )

        scenario_root = Path(__file__).resolve().parent / "outputs" / args.system / "scenario_library"
        _merge_supplement_into_base(
            base_pkl=scenario_root / args.tag / "library.pkl",
            supp_pkl=scenario_root / supp_tag / "library.pkl",
            swap_undervoltage_for_overvoltage=args.swap_undervoltage_for_overvoltage,
        )
