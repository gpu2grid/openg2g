"""Unified baseline simulation for any IEEE test system.

Runs without batch-size control (fixed batch sizes). Optionally applies
a tap schedule from the config.

Usage:
    # IEEE 13 (single DC, no tap changes)
    python run_baseline.py --config config_ieee13.json --system ieee13

    # IEEE 34 (two DCs)
    python run_baseline.py --config config_ieee34.json --system ieee34

    # IEEE 123 (four DCs)
    python run_baseline.py --config config_ieee123.json --system ieee123

    # With tap schedule changes
    python run_baseline.py --config config_ieee13.json --system ieee13 --mode tap-change
"""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

import numpy as np

from run_ofo import _build_tap_schedule, run_mode

from sweep_dc_locations import (
    SweepConfig,
    _parse_fraction,
)
from openg2g.datacenter.config import InferenceModelSpec
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.controller.ofo import LogisticModelStore
from openg2g.grid.config import TapSchedule
from openg2g.metrics.voltage import VoltageStats

logger = logging.getLogger("run_baseline")

<<<<<<< HEAD
# fmt: off
TAP_STEP = 0.00625
INITIAL_TAPS = TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP)
TAP_CHANGE_SCHEDULE = (
    TapPosition(a=1.0 + 16 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 17 * TAP_STEP).at(t=25 * 60)
    | TapPosition(a=1.0 + 10 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 10 * TAP_STEP).at(t=55 * 60)
)
# fmt: on
=======
>>>>>>> f03cf6c (Add multi-datacenter architecture: Coordinator accepts multiple DCs. Add functions to sweep ofo parameters, sweep DC locations, find DC hosting capacity, and optimize PV locations and capacities. Add IEEE 13, 34, 123 test feeders and example scripts. Include simulation outputs for IEEE 13, 34, 123 under multiple scenarios.)

def main(*, config_path: Path, system: str, mode: str = "no-tap") -> None:
    config_path = config_path.resolve()
    config = SweepConfig.model_validate_json(config_path.read_bytes())
    config_dir = config_path.parent
    config.ieee_case_dir = (config_dir / config.ieee_case_dir).resolve()

    sim = config.simulation
    all_models = tuple(config.models)
    data_dir = config.data_dir or Path("data/offline") / config.data_hash
    data_dir = (config_dir / data_dir).resolve() if not data_dir.is_absolute() else data_dir

<<<<<<< HEAD
    save_dir = (Path("outputs") / f"baseline_{mode}").resolve()
=======
    dt_dc = _parse_fraction(sim.dt_dc)
    dt_grid = _parse_fraction(sim.dt_grid)
    dt_ctrl = _parse_fraction(sim.dt_ctrl)

    # Load data
    logger.info("Loading data for %s...", system)
    data_sources = {s.model_label: s for s in config.data_sources}
    inference_data = InferenceData.ensure(
        data_dir, all_models, data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir, plot=False, dt_s=float(dt_dc),
    )
    training_trace = TrainingTrace.ensure(
        data_dir / "training_trace.csv", config.training_trace_params,
    )
    # LogisticModelStore is needed by run_mode signature but not used for baseline
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv", all_models, data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir, plot=False,
    )

    # Build tap schedule
    has_tap_schedule = (
        config.tap_schedule is not None and len(config.tap_schedule) > 0
    )
    tap_sched = (
        _build_tap_schedule(config.tap_schedule, config.initial_taps)
        if has_tap_schedule else None
    )

    # Determine folder name
    if mode == "tap-change" and has_tap_schedule:
        folder = "baseline_tap-change"
        sched = tap_sched
    else:
        folder = "baseline_no-tap"
        sched = None

    if mode == "tap-change" and not has_tap_schedule:
        logger.warning("No tap_schedule in config; running no-tap baseline.")

    save_dir = Path(__file__).resolve().parent / "outputs" / system
>>>>>>> f03cf6c (Add multi-datacenter architecture: Coordinator accepts multiple DCs. Add functions to sweep ofo parameters, sweep DC locations, find DC hosting capacity, and optimize PV locations and capacities. Add IEEE 13, 34, 123 test feeders and example scripts. Include simulation outputs for IEEE 13, 34, 123 under multiple scenarios.)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running %s baseline (%s)...", system, folder)

    stats, log = run_mode(
        "baseline",
        config=config,
        all_models=all_models,
        inference_data=inference_data,
        training_trace=training_trace,
        logistic_models=logistic_models,
        dt_dc=dt_dc, dt_grid=dt_grid, dt_ctrl=dt_ctrl,
        save_dir=save_dir,
        tap_schedule=sched,
        folder_name=folder,
    )

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("%s BASELINE (%s)", system.upper(), folder)
    logger.info("=" * 70)
    logger.info("  Violation time:  %.1f s", stats.violation_time_s)
    logger.info("  Worst Vmin:      %.4f pu", stats.worst_vmin)
    logger.info("  Worst Vmax:      %.4f pu", stats.worst_vmax)
    logger.info("  Integral:        %.4f pu*s", stats.integral_violation_pu_s)
    logger.info("  Outputs: %s", save_dir / folder)


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
        """Run mode: 'no-tap' (default) or 'tap-change'."""
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
