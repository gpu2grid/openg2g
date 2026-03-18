"""Unified baseline simulation for any IEEE test system.

Runs without batch-size control (fixed batch sizes). Optionally applies
a tap schedule from the config.

Modes:
    no-tap       Baseline without tap schedule (1 case, default).
    tap-change   Baseline with tap schedule from config (1 case).
    both         Baseline with and without tap schedule (2 cases).

Default outputs per case folder (same as run_ofo.py):
    allbus_voltages_phase_{A,B,C}.png   Per-phase voltage trajectories
    tap_positions.png                   Regulator tap positions over time
    power_latency_{site}.png            Per-site 3-phase power + ITL
    pv_load_profiles.png                PV and time-varying load curves
    result_{case}.csv                   Voltage violation summary

Usage:
    # IEEE 13 — baseline, no tap schedule
    python run_baseline.py --config config_ieee13.json --system ieee13

    # IEEE 34 — baseline, no tap schedule
    python run_baseline.py --config config_ieee34.json --system ieee34

    # IEEE 123 — baseline, no tap schedule
    python run_baseline.py --config config_ieee123.json --system ieee123

    # Baseline with tap schedule
    python run_baseline.py --config config_ieee34.json --system ieee34 --mode tap-change

    # Baseline with and without tap schedule
    python run_baseline.py --config config_ieee34.json --system ieee34 --mode both
"""

from __future__ import annotations

import logging
from pathlib import Path

from run_ofo import _build_tap_schedule, run_mode
from sweep_dc_locations import (
    SweepConfig,
    _parse_fraction,
)

from openg2g.controller.ofo import LogisticModelStore
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.config import TapSchedule
from openg2g.metrics.voltage import VoltageStats

logger = logging.getLogger("run_baseline")


def main(*, config_path: Path, system: str, mode: str = "no-tap") -> None:
    config_path = config_path.resolve()
    config = SweepConfig.model_validate_json(config_path.read_bytes())
    config_dir = config_path.parent
    config.ieee_case_dir = (config_dir / config.ieee_case_dir).resolve()

    sim = config.simulation
    all_models = tuple(config.models)
    data_dir = config.data_dir or Path("data/offline") / config.data_hash
    data_dir = (config_dir / data_dir).resolve() if not data_dir.is_absolute() else data_dir

    dt_dc = _parse_fraction(sim.dt_dc)
    dt_grid = _parse_fraction(sim.dt_grid)
    dt_ctrl = _parse_fraction(sim.dt_ctrl)

    # Load data
    logger.info("Loading data for %s...", system)
    data_sources = {s.model_label: s for s in config.data_sources}
    inference_data = InferenceData.ensure(
        data_dir,
        all_models,
        data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir,
        plot=False,
        dt_s=float(dt_dc),
    )
    training_trace = TrainingTrace.ensure(
        data_dir / "training_trace.csv",
        config.training_trace_params,
    )
    # LogisticModelStore is needed by run_mode signature but not used for baseline
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv",
        all_models,
        data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir,
        plot=False,
    )

    # Build tap schedule
    has_tap_schedule = config.tap_schedule is not None and len(config.tap_schedule) > 0
    tap_sched = _build_tap_schedule(config.tap_schedule, config.initial_taps) if has_tap_schedule else None

    # Build cases based on mode
    cases: list[tuple[str, TapSchedule | None]] = []
    if mode in ("no-tap", "both"):
        cases.append(("baseline_no-tap", None))
    if mode in ("tap-change", "both") and has_tap_schedule:
        cases.append(("baseline_tap-change", tap_sched))
    if not cases:
        if mode == "tap-change" and not has_tap_schedule:
            logger.warning("No tap_schedule in config; running no-tap baseline.")
            cases.append(("baseline_no-tap", None))
        else:
            logger.warning("No cases to run (mode=%s, has_tap_schedule=%s).", mode, has_tap_schedule)
            return

    save_dir = Path(__file__).resolve().parent / "outputs" / system
    save_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, VoltageStats] = {}
    for folder, sched in cases:
        logger.info("Running %s baseline (%s)...", system, folder)
        stats, _log = run_mode(
            "baseline",
            config=config,
            all_models=all_models,
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            dt_dc=dt_dc,
            dt_grid=dt_grid,
            dt_ctrl=dt_ctrl,
            save_dir=save_dir,
            tap_schedule=sched,
            folder_name=folder,
        )
        results[folder] = stats

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("%s BASELINE SUMMARY", system.upper())
    logger.info("=" * 70)
    logger.info("%-25s %10s %10s %10s %14s", "Mode", "Viol(s)", "Vmin", "Vmax", "Integral")
    logger.info("-" * 70)
    for folder, s in results.items():
        logger.info(
            "%-25s %10.1f %10.4f %10.4f %14.4f",
            folder,
            s.violation_time_s,
            s.worst_vmin,
            s.worst_vmax,
            s.integral_violation_pu_s,
        )
    logger.info("-" * 70)
    logger.info("Outputs: %s", save_dir)


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        config: str
        """Path to the config JSON file."""
        system: str = "ieee13"
        """System name (ieee13, ieee34, ieee123) for output directory."""
        mode: str = "no-tap"
        """Run mode: 'no-tap' (default), 'tap-change', or 'both' (w/ and w/o tap)."""
        log_level: str = "INFO"
        """Logging verbosity (DEBUG, INFO, WARNING)."""

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    main(config_path=Path(args.config), system=args.system, mode=args.mode)
