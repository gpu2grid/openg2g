"""Online hardware-in-the-loop simulation using real GPUs.

Connects to live vLLM servers and zeusd instances for real-time
simulation. Power readings from a small number of real GPUs are augmented
to datacenter scale using the shared InferencePowerAugmenter pipeline.

Modes:
    baseline-no-tap       No batch control, fixed taps.
    baseline-tap-change   No batch control, scheduled tap changes.
    ofo                   OFO closed-loop batch-size optimization.

Edit the deployment definitions in config.json to match your cluster.

Usage:
    python examples/online/run_ofo.py --config examples/online/config.json --mode baseline-no-tap
    python examples/online/run_ofo.py --config examples/online/config.json --mode ofo
"""

from __future__ import annotations

import hashlib
import json
import logging
from fractions import Fraction
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from openg2g.controller.ofo import (
    LogisticModelStore,
    OFOBatchSizeController,
    OFOConfig,
)
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.config import DatacenterConfig, PowerAugmentationConfig
from openg2g.datacenter.online import (
    LiveServerConfig,
    OnlineDatacenter,
    VLLMDeployment,
)
from openg2g.datacenter.workloads.inference import MLEnergySource, RequestsConfig, RequestStore
from openg2g.grid.config import TapPosition, TapSchedule
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.performance import compute_performance_stats
from openg2g.metrics.voltage import compute_allbus_voltage_stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger("run_ofo")

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
GPUS_PER_SERVER = 8
DT_DC = Fraction(1, 10)
DT_CTRL = Fraction(1)
T_TOTAL_S = 3600


class OnlineConfig(BaseModel):
    deployments: list[VLLMDeployment]
    requests: RequestsConfig = RequestsConfig()
    requests_dir: Path | None = None
    ieee_case_dir: Path
    data_dir: Path | None = None
    data_sources: list[MLEnergySource] = []
    mlenergy_data_dir: Path | None = None

    @property
    def requests_hash(self) -> str:
        blob = json.dumps(
            (self.requests.model_dump(mode="json"), sorted(d.spec.model_label for d in self.deployments)),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @property
    def data_hash(self) -> str:
        blob = json.dumps(
            (sorted([s.model_dump(mode="json") for s in self.data_sources], key=lambda s: s["model_label"]),),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def main(*, config_path: Path, mode: str = "ofo") -> None:
    config = OnlineConfig.model_validate_json(config_path.read_bytes())

    models = tuple(d.spec for d in config.deployments)
    requests_dir = config.requests_dir or Path("data/online") / config.requests_hash

    save_dir = (Path("outputs") / f"online_{mode}").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(save_dir / "console_output.txt", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    RequestStore.ensure(requests_dir, [d.spec for d in config.deployments], config.requests)

    # Datacenter
    dc_config = DatacenterConfig(gpus_per_server=GPUS_PER_SERVER, base_kw_per_phase=500.0)
    dc = OnlineDatacenter(
        dc_config,
        config.deployments,
        name="dc",
        dt_s=DT_DC,
        seed=0,
        power_augmentation=PowerAugmentationConfig(
            amplitude_scale_range=(0.9, 1.1),
            noise_fraction=0.02,
        ),
        live_server=LiveServerConfig(
            requests_dir=requests_dir,
            max_output_tokens=config.requests.max_completion_tokens,
            itl_window_s=1.0,
        ),
    )

    # Grid
    grid = OpenDSSGrid(
        dss_case_dir=config.ieee_case_dir,
        dss_master_file="IEEE13Bus.dss",
        dt_s=Fraction(1, 10),
        initial_tap_position=INITIAL_TAPS,
        exclude_buses=["sourcebus", "650", "rg60"],
    )
    grid.attach_dc(dc, bus=DC_BUS, connection_type="wye", power_factor=dc_config.power_factor)

    # Controllers
    controllers: list = []
    if mode == "baseline-tap-change":
        controllers.append(TapScheduleController(schedule=TAP_CHANGE_SCHEDULE, dt_s=DT_CTRL))
    elif mode == "baseline-no-tap":
        controllers.append(TapScheduleController(schedule=TapSchedule(()), dt_s=DT_CTRL))
    elif mode == "ofo":
        controllers.append(TapScheduleController(schedule=TapSchedule(()), dt_s=DT_CTRL))

        data_sources = {s.model_label: s for s in config.data_sources} if config.data_sources else None
        data_dir = config.data_dir or _PROJECT_ROOT / "data" / "offline" / config.data_hash
        logistic_models = LogisticModelStore.ensure(
            data_dir / "logistic_fits.csv",
            models,
            data_sources,
            mlenergy_data_dir=config.mlenergy_data_dir,
            plot=False,
        )
        controllers.append(
            OFOBatchSizeController(
                models,
                datacenter=dc,
                models=logistic_models,
                config=OFOConfig(
                    primal_step_size=0.1,
                    w_throughput=1e-3,
                    w_switch=1.0,
                    voltage_gradient_scale=1e6,
                    v_min=V_MIN,
                    v_max=V_MAX,
                    voltage_dual_step_size=1.0,
                    latency_dual_step_size=1.0,
                    sensitivity_update_interval=3600,
                    sensitivity_perturbation_kw=100.0,
                ),
                dt_s=DT_CTRL,
                grid=grid,
            )
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}. Choose: baseline-no-tap, baseline-tap-change, ofo.")

    # Run
    logger.info("Running online simulation (mode=%s) for %d seconds...", mode, T_TOTAL_S)
    coord = Coordinator(
        datacenters=[dc],
        grid=grid,
        controllers=controllers,
        total_duration_s=T_TOTAL_S,
        live=True,
    )
    log = coord.run()

    # Results
    stats = compute_allbus_voltage_stats(
        log.grid_states, v_min=V_MIN, v_max=V_MAX, exclude_buses=("sourcebus", "650", "rg60")
    )
    itl_deadlines = {d.spec.model_label: d.spec.itl_deadline_s for d in config.deployments}
    pstats = compute_performance_stats(log.dc_states, itl_deadline_s_by_model=itl_deadlines)
    logger.info("=== Voltage Statistics (all-bus) ===")
    logger.info("  voltage_violation_time = %.1f s", stats.violation_time_s)
    logger.info("  worst_vmin             = %.6f", stats.worst_vmin)
    logger.info("  worst_vmax             = %.6f", stats.worst_vmax)
    logger.info("  integral_violation     = %.5f pu-s", stats.integral_violation_pu_s)
    logger.info("=== Performance Statistics ===")
    logger.info("  mean_throughput        = %.1f k tok/s", pstats.mean_throughput_tps / 1e3)
    logger.info("  integrated_throughput  = %.0f tokens", pstats.integrated_throughput_tokens)
    logger.info("  itl_over_deadline      = %.2f%%", pstats.itl_deadline_fraction * 100.0)

    # Per-case CSV (matches offline convention)
    import csv as _csv

    csv_path = save_dir / f"result_{mode.replace('-', '_')}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(
            [
                "case",
                "violation_time_s",
                "integral_violation_pu_s",
                "worst_vmin",
                "worst_vmax",
                "mean_throughput_tps",
                "integrated_throughput_tokens",
                "itl_deadline_fraction",
            ]
        )
        writer.writerow(
            [
                mode,
                stats.violation_time_s,
                stats.integral_violation_pu_s,
                stats.worst_vmin,
                stats.worst_vmax,
                pstats.mean_throughput_tps,
                pstats.integrated_throughput_tokens,
                pstats.itl_deadline_fraction,
            ]
        )
    logger.info("Per-case CSV: %s", csv_path)

    if mode == "ofo" and log.dc_states:
        logger.info("=== Batch Schedule Summary ===")
        model_labels = sorted(log.dc_states[0].batch_size_by_model.keys())
        for label in model_labels:
            batches = np.array([s.batch_size_by_model.get(label, 0) for s in log.dc_states])
            if batches.size:
                avg = float(np.mean(batches))
                changes = int(np.sum(np.diff(batches) != 0))
                logger.info("  %s: avg_batch=%.1f, changes=%d", label, avg, changes)

    logger.info("Outputs saved to: %s", save_dir)


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        """Command-line arguments.

        Attributes:
            config: Path to the online config JSON file.
            mode: Mode: baseline-no-tap, baseline-tap-change, or ofo.
            log_level: Logging verbosity (DEBUG, INFO, WARNING).
        """

        config: str
        mode: str = "ofo"
        log_level: str = "INFO"

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    main(config_path=Path(args.config), mode=args.mode)
