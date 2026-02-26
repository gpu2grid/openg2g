"""Online OFO closed-loop simulation using real GPUs.

Connects to live vLLM servers and zeusd instances for hardware-in-the-loop
OFO control.  Power readings from a small number of real GPUs are augmented
to datacenter scale using the shared PowerAugmenter pipeline.

Edit the deployment definitions below to match your cluster before running.

Usage:
    python -m examples.online.run_ofo --data-dir data/generated \
        --ieee-case-dir examples/ieee13 --requests-dir data/online/requests
"""

from __future__ import annotations

import argparse
import json
import logging
from fractions import Fraction
from pathlib import Path

import numpy as np

from openg2g.controller.ofo import (
    LogisticModelStore,
    OFOBatchSizeController,
    OFOConfig,
)
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.config import DatacenterConfig, PowerAugmentationConfig
from openg2g.datacenter.online import (
    GPUEndpointMapping,
    LiveServerConfig,
    OnlineDatacenter,
    VLLMDeployment,
)
from openg2g.grid.base import Phase
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import compute_allbus_voltage_stats
from openg2g.models.spec import LLMInferenceModelSpec, LLMInferenceWorkload
from openg2g.types import TapPosition, TapSchedule

logger = logging.getLogger("run_ofo")

TAP_STEP = 0.00625
INITIAL_TAPS = TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP)

MAX_BATCH_SIZE = 512

DEPLOYMENTS = [
    VLLMDeployment(
        spec=LLMInferenceModelSpec(
            model_label="Llama-3.1-8B",
            num_replicas=720,
            gpus_per_replica=1,
            feasible_batch_sizes=tuple(range(1, MAX_BATCH_SIZE + 1)),
            initial_batch_size=128,
            itl_deadline_s=0.08,
        ),
        vllm_base_url="http://node1:8000",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        gpu_endpoints=(
            GPUEndpointMapping(host="node1", port=4938, gpu_indices=(0, 1, 2, 3), phase=Phase.A),
            GPUEndpointMapping(host="node1", port=4938, gpu_indices=(4, 5, 6, 7), phase=Phase.B),
        ),
    ),
    VLLMDeployment(
        spec=LLMInferenceModelSpec(
            model_label="Llama-3.1-70B",
            num_replicas=180,
            gpus_per_replica=4,
            feasible_batch_sizes=tuple(range(1, MAX_BATCH_SIZE + 1)),
            initial_batch_size=128,
            itl_deadline_s=0.10,
        ),
        vllm_base_url="http://node2:8000",
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        gpu_endpoints=(
            GPUEndpointMapping(host="node2", port=4938, gpu_indices=(0, 1, 2, 3), phase=Phase.A),
            GPUEndpointMapping(host="node2", port=4938, gpu_indices=(4, 5, 6, 7), phase=Phase.C),
        ),
    ),
]

WORKLOAD = LLMInferenceWorkload(models=tuple(d.spec for d in DEPLOYMENTS))

V_MIN = 0.95
V_MAX = 1.05
DC_BUS = "671"


def _load_requests(requests_dir: Path, model_labels: list[str]) -> dict[str, list[dict]]:
    """Load pre-built request dicts from per-model JSONL files."""
    result: dict[str, list[dict]] = {}
    for label in model_labels:
        path = requests_dir / f"{label}.jsonl"
        if not path.exists():
            logger.warning("No request file found for %s at %s", label, path)
            continue
        requests = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    requests.append(json.loads(line))
        result[label] = requests
        logger.info("Loaded %d requests for %s", len(requests), label)
    return result


def main(*, data_dir: Path, ieee_case_dir: Path, requests_dir: Path, duration: int = 3600) -> None:
    save_dir = Path("outputs") / "online_ofo"
    save_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(save_dir / "console_output.txt", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    dt_dc = Fraction(1, 10)
    dt_ctrl = Fraction(1)
    t_total_s = duration

    model_labels = [d.model_label for d in DEPLOYMENTS]

    logger.info("Loading pre-built requests...")
    requests_by_model = _load_requests(requests_dir, model_labels)

    logger.info("Loading logistic fits...")
    logistic_models = LogisticModelStore.load(data_dir / "logistic_fits.csv")

    logger.info("Initializing OnlineDatacenter...")
    dc = OnlineDatacenter(
        DatacenterConfig(gpus_per_server=8, base_kw_per_phase=500.0),
        DEPLOYMENTS,
        requests_by_model,
        dt_s=dt_dc,
        seed=0,
        power_augmentation=PowerAugmentationConfig(
            amplitude_scale_range=(0.9, 1.1),
            noise_fraction=0.02,
        ),
        live_server=LiveServerConfig(
            max_output_tokens=512,
            itl_window_s=1.0,
        ),
    )

    logger.info("Initializing OpenDSSGrid...")
    grid = OpenDSSGrid(
        dss_case_dir=str(ieee_case_dir),
        dss_master_file="IEEE13Nodeckt.dss",
        dc_bus=DC_BUS,
        dc_bus_kv=4.16,
        power_factor=0.95,
        dt_s=Fraction(1, 10),
        connection_type="wye",
        initial_tap_position=INITIAL_TAPS,
    )

    tap_ctrl = TapScheduleController(schedule=TapSchedule(()), dt_s=dt_ctrl)

    ofo_ctrl = OFOBatchSizeController(
        WORKLOAD,
        models=logistic_models,
        config=OFOConfig(
            descent_step_size=0.1,
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
        dt_s=dt_ctrl,
    )

    logger.info("Running online simulation for %d seconds...", t_total_s)
    coord = Coordinator(
        datacenter=dc,
        grid=grid,
        controllers=[tap_ctrl, ofo_ctrl],
        total_duration_s=t_total_s,
        dc_bus=DC_BUS,
        live=True,
    )
    log = coord.run()

    stats = compute_allbus_voltage_stats(log.grid_states, v_min=V_MIN, v_max=V_MAX)
    logger.info("=== Voltage Statistics (all-bus) ===")
    logger.info("  voltage_violation_time = %.1f s", stats.violation_time_s)
    logger.info("  worst_vmin             = %.6f", stats.worst_vmin)
    logger.info("  worst_vmax             = %.6f", stats.worst_vmax)
    logger.info("  integral_violation     = %.5f pu·s", stats.integral_violation_pu_s)

    logger.info("=== Batch Schedule Summary ===")
    if log.dc_states:
        model_labels = sorted(log.dc_states[0].batch_size_by_model.keys())
        for label in model_labels:
            batches = np.array([s.batch_size_by_model.get(label, 0) for s in log.dc_states])
            if batches.size:
                avg = float(np.mean(batches))
                changes = int(np.sum(np.diff(batches) != 0))
                logger.info("  %s: avg_batch=%.1f, changes=%d", label, avg, changes)

    logger.info("Outputs saved to: %s", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online OFO closed-loop simulation (hardware-in-the-loop)")
    parser.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="Simulation duration in seconds (default: 3600).",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Toolkit-generated data directory (contains logistic_fits.csv).",
    )
    parser.add_argument(
        "--ieee-case-dir",
        required=True,
        help="OpenDSS case directory (e.g. examples/ieee13).",
    )
    parser.add_argument(
        "--requests-dir",
        required=True,
        help="Directory with per-model JSONL request files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )

    main(
        data_dir=Path(args.data_dir),
        duration=args.duration,
        ieee_case_dir=Path(args.ieee_case_dir),
        requests_dir=Path(args.requests_dir),
    )
