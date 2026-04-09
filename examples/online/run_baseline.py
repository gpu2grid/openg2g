"""Online baseline (no OFO control) simulation using real GPUs.

Connects to live vLLM servers and zeusd instances for hardware-in-the-loop
baseline measurement.  Power readings from a small number of real GPUs are
augmented to datacenter scale using the shared InferencePowerAugmenter pipeline.

Two modes correspond to two baselines:
  no-tap       Fixed tap positions throughout.
  tap-change   Regulator taps change at scheduled times.

Edit the deployment definitions in config.json to match your cluster.

Usage:
    python examples/online/run_baseline.py --config examples/online/config.json
    python examples/online/run_baseline.py --config examples/online/config.json --mode tap-change
"""

from __future__ import annotations

import hashlib
import json
import logging
from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel

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
from openg2g.metrics.voltage import compute_allbus_voltage_stats

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


def main(*, config_path: Path, mode: str = "no-tap") -> None:
    config = OnlineConfig.model_validate_json(config_path.read_bytes())

    requests_dir = config.requests_dir or Path("data/online") / config.requests_hash

    save_dir = (Path("outputs") / f"online_baseline_{mode}").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(save_dir / "console_output.txt", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    RequestStore.ensure(requests_dir, [d.spec for d in config.deployments], config.requests)

    tap_ctrl_schedule = TAP_CHANGE_SCHEDULE if mode == "tap-change" else TapSchedule(())

    logger.info("Initializing OnlineDatacenter...")
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

    logger.info("Initializing OpenDSSGrid...")
    grid = OpenDSSGrid(
        dss_case_dir=config.ieee_case_dir,
        dss_master_file="IEEE13Bus.dss",
        dt_s=Fraction(1, 10),
        initial_tap_position=INITIAL_TAPS,
    )
    grid.attach_dc(dc, bus=DC_BUS, connection_type="wye", power_factor=dc_config.power_factor)

    tap_ctrl = TapScheduleController(schedule=tap_ctrl_schedule, dt_s=DT_CTRL)

    logger.info("Running online baseline simulation (mode=%s) for %d seconds...", mode, T_TOTAL_S)
    coord = Coordinator(
        datacenters=[dc],
        grid=grid,
        controllers=[tap_ctrl],
        total_duration_s=T_TOTAL_S,
        live=True,
    )
    log = coord.run()

    stats = compute_allbus_voltage_stats(log.grid_states, v_min=V_MIN, v_max=V_MAX)
    logger.info("=== Voltage Statistics (all-bus) ===")
    logger.info("  voltage_violation_time = %.1f s", stats.violation_time_s)
    logger.info("  worst_vmin             = %.6f", stats.worst_vmin)
    logger.info("  worst_vmax             = %.6f", stats.worst_vmax)
    logger.info("  integral_violation     = %.4f pu·s", stats.integral_violation_pu_s)

    logger.info("Outputs saved to: %s", save_dir)


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        config: str
        """Path to the online config JSON file."""
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
