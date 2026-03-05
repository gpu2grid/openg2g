"""Baseline simulation using the openg2g library.

Components: OfflineDatacenter + OpenDSSGrid + TapScheduleController + Coordinator.

Two modes correspond to two baselines in the G2G paper:
  no-tap       "No control, no tap": tap positions are fixed throughout.
  tap-change   "Tap change only": regulator taps change at t=1500s and t=3300s.

Usage:
    python examples/offline/run_baseline.py --config examples/offline/config.json
    python examples/offline/run_baseline.py --config examples/offline/config.json --mode tap-change
"""

from __future__ import annotations

import hashlib
import json
import logging
from fractions import Fraction
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.config import (
    DatacenterConfig,
    InferenceModelSpec,
    InferenceRamp,
    PowerAugmentationConfig,
    TrainingRun,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData, MLEnergySource
from openg2g.datacenter.workloads.training import TrainingTrace, TrainingTraceParams
from openg2g.grid.config import TapPosition, TapSchedule
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import compute_allbus_voltage_stats

from plotting import (
    extract_per_model_timeseries,
    plot_allbus_voltages_per_phase,
    plot_power_3ph,
    plot_power_and_itl_2panel,
)

logger = logging.getLogger("run_baseline")

# fmt: off
TAP_STEP = 0.00625
INITIAL_TAPS = TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP)
TAP_CHANGE_SCHEDULE = (
    TapPosition(a=1.0 + 16 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 17 * TAP_STEP).at(t=25 * 60)
    | TapPosition(a=1.0 + 10 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 10 * TAP_STEP).at(t=55 * 60)
)
# fmt: on

V_MIN = 0.95
V_MAX = 1.05
DC_BUS = "671"
DT_DC = Fraction(1, 10)
DT_CTRL = Fraction(1)
T_TOTAL_S = 3600


class OfflineConfig(BaseModel):
    models: list[InferenceModelSpec]
    data_sources: list[MLEnergySource]
    training_trace_params: TrainingTraceParams = TrainingTraceParams()
    data_dir: Path | None = None
    ieee_case_dir: Path
    mlenergy_data_dir: Path | None = None

    @property
    def data_hash(self) -> str:
        blob = json.dumps(
            (
                sorted([s.model_dump(mode="json") for s in self.data_sources], key=lambda s: s["model_label"]),
                self.training_trace_params.model_dump(mode="json"),
            ),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def main(*, config_path: Path, mode: str = "no-tap") -> None:
    config = OfflineConfig.model_validate_json(config_path.read_bytes())

    models = tuple(config.models)
    data_sources = {s.model_label: s for s in config.data_sources}
    data_dir = config.data_dir or Path("data/offline") / config.data_hash

    save_dir = (Path("outputs") / f"baseline_{mode}").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(save_dir / "console_output.txt", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    inference_data = InferenceData.ensure(
        data_dir,
        models,
        data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir,
        plot=False,
        dt_s=float(DT_DC),
    )
    training_trace = TrainingTrace.ensure(data_dir / "training_trace.csv", config.training_trace_params)

    dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=500.0)
    workload = OfflineWorkload(
        inference_data=inference_data,
        training=TrainingRun(n_gpus=300 * 8, trace=training_trace, target_peak_W_per_gpu=400.0).at(
            t_start=1000.0, t_end=2000.0
        ),
        inference_ramps=InferenceRamp(target=0.2).at(t_start=2500.0, t_end=3000.0),
    )

    logger.info("Initializing OfflineDatacenter...")
    dc = OfflineDatacenter(
        dc_config,
        workload,
        dt_s=DT_DC,
        seed=0,
        power_augmentation=PowerAugmentationConfig(
            amplitude_scale_range=(0.98, 1.02),
            noise_fraction=0.005,
        ),
    )

    logger.info("Initializing OpenDSSGrid...")
    grid = OpenDSSGrid(
        dss_case_dir=config.ieee_case_dir,
        dss_master_file="IEEE13Nodeckt.dss",
        dc_bus=DC_BUS,
        dc_bus_kv=4.16,
        power_factor=dc_config.power_factor,
        dt_s=Fraction(1, 10),
        connection_type="wye",
        initial_tap_position=INITIAL_TAPS,
    )

    tap_ctrl_schedule = TAP_CHANGE_SCHEDULE if mode == "tap-change" else TapSchedule(())
    ctrl = TapScheduleController(schedule=tap_ctrl_schedule, dt_s=DT_CTRL)

    logger.info("Running simulation (mode=%s)...", mode)
    coord = Coordinator(
        datacenter=dc,
        grid=grid,
        controllers=[ctrl],
        total_duration_s=T_TOTAL_S,
        dc_bus=DC_BUS,
    )
    log = coord.run()

    stats = compute_allbus_voltage_stats(log.grid_states, v_min=V_MIN, v_max=V_MAX)
    logger.info("=== Voltage Statistics (all-bus) ===")
    logger.info("  voltage_violation_time = %.1f s", stats.violation_time_s)
    logger.info("  worst_vmin             = %.6f", stats.worst_vmin)
    logger.info("  worst_vmax             = %.6f", stats.worst_vmax)
    logger.info("  integral_violation     = %.4f pu·s", stats.integral_violation_pu_s)

    time_s = np.array(log.time_s)
    dc_time_s = np.array([s.time_s for s in log.dc_states])
    kW_A = np.array([s.power_w.a / 1e3 for s in log.dc_states])
    kW_B = np.array([s.power_w.b / 1e3 for s in log.dc_states])
    kW_C = np.array([s.power_w.c / 1e3 for s in log.dc_states])

    per_model = extract_per_model_timeseries(log.dc_states)

    plot_power_and_itl_2panel(
        dc_time_s,
        kW_A,
        kW_B,
        kW_C,
        avg_itl_by_model=per_model.itl_s,
        itl_time_s=per_model.time_s,
        save_path=save_dir / "power_latency_subfigs.png",
    )

    plot_allbus_voltages_per_phase(
        log.grid_states,
        time_s,
        save_dir=save_dir,
        v_min=V_MIN,
        v_max=V_MAX,
        title_template="Voltage trajectories without GPU flexibility (Phase {label})",
    )

    plot_power_3ph(
        dc_time_s,
        kW_A,
        kW_B,
        kW_C,
        save_path=save_dir / "dc_power_3ph.png",
        title="DC Power by Phase",
    )

    logger.info("Outputs saved to: %s", save_dir)


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        config: str
        """Path to the offline config JSON file."""
        mode: str = "no-tap"
        """Baseline variant: 'no-tap' (fixed taps) or 'tap-change' (scheduled tap changes)."""
        log_level: str = "INFO"
        """Logging verbosity (DEBUG, INFO, WARNING)."""

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    main(config_path=Path(args.config), mode=args.mode)
