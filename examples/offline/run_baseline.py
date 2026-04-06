"""Baseline simulation (no batch-size control) for any IEEE test system.

Reuses the same experiment definitions and ``run_mode`` from ``run_ofo``,
but only runs baseline cases (fixed batch sizes).

Modes:
    no-tap       Baseline without tap schedule (1 case, default).
    tap-change   Baseline with tap schedule (1 case).
    both         Baseline with and without tap schedule (2 cases).

Usage:
    python run_baseline.py --system ieee13
    python run_baseline.py --system ieee34 --mode tap-change
    python run_baseline.py --system ieee34 --mode both
"""

from __future__ import annotations

import logging
from pathlib import Path

from run_ofo import _EXPERIMENTS, run_mode
from systems import DT_DC, SYSTEMS, all_model_specs, load_data_sources

from openg2g.controller.ofo import LogisticModelStore
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.config import TapSchedule
from openg2g.metrics.voltage import VoltageStats

logger = logging.getLogger("run_baseline")


def main(*, system: str, mode: str = "no-tap") -> None:
    sys = SYSTEMS[system]()

    data_sources, training_trace_params, data_dir = load_data_sources()
    all_models = all_model_specs()

    logger.info("Loading data for %s...", system)
    inference_data = InferenceData.ensure(data_dir, all_models, data_sources, plot=False, dt_s=float(DT_DC))
    training_trace = TrainingTrace.ensure(data_dir / "training_trace.csv", training_trace_params)
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv",
        all_models,
        data_sources,
        plot=False,
    )

    experiment = _EXPERIMENTS[system](sys, inference_data, training_trace, logistic_models)
    has_tap_schedule = bool(experiment.get("tap_schedule"))

    cases: list[tuple[str, TapSchedule | None]] = []
    if mode in ("no-tap", "both"):
        cases.append(("baseline_no-tap", None))
    if mode in ("tap-change", "both") and has_tap_schedule:
        cases.append(("baseline_tap-change", experiment["tap_schedule"]))
    if not cases:
        if mode == "tap-change" and not has_tap_schedule:
            logger.warning("No tap_schedule defined; running no-tap baseline.")
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
            sys=sys,
            dc_sites=experiment["dc_sites"],
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            training=experiment.get("training"),
            ofo_config=experiment.get("ofo_config"),
            tap_schedule=sched,
            pv_systems=experiment.get("pv_systems"),
            time_varying_loads=experiment.get("time_varying_loads"),
            zones=experiment.get("zones"),
            save_dir=save_dir,
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
        system: str = "ieee13"
        """System name (ieee13, ieee34, ieee123)."""
        mode: str = "no-tap"
        """Run mode: 'no-tap' (default), 'tap-change', or 'both'."""
        log_level: str = "INFO"
        """Logging verbosity (DEBUG, INFO, WARNING)."""

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    main(system=args.system, mode=args.mode)
