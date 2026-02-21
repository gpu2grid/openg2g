"""Online OFO closed-loop simulation using real GPUs.

Connects to live vLLM servers and zeusd instances for hardware-in-the-loop
OFO control.  Power readings from a small number of real GPUs are augmented
to datacenter scale using temporal staggering.

Usage:
    python examples/online/run_ofo.py --config examples/online/online_config.example.yaml --duration 3600
"""

from __future__ import annotations

import argparse
import json
import logging
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from mlenergy_data.modeling import LogisticModel
from zeus.monitor.power_streaming import PowerStreamingClient, ZeusdTcpConfig

from openg2g.controller.ofo import (
    OFOBatchController,
    PrimalConfig,
    VoltageDualConfig,
)
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.online import (
    GPUEndpointMapping,
    LoadGenerationConfig,
    OnlineDatacenter,
    OnlineModelDeployment,
    PowerAugmentationConfig,
)
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import compute_allbus_voltage_stats
from openg2g.models.spec import LLMInferenceModelSpec, LLMInferenceWorkload
from openg2g.types import Phase, TapPosition

logger = logging.getLogger("run_ofo")

BATCH_SET = (8, 16, 32, 64, 128, 256, 512)


def _load_logistic_fits_merged(
    csv_path: Path,
) -> tuple[dict[str, LogisticModel], dict[str, LogisticModel], dict[str, LogisticModel]]:
    """Load power, latency, throughput fits from a merged CSV."""
    df = pd.read_csv(csv_path)
    power: dict[str, LogisticModel] = {}
    latency: dict[str, LogisticModel] = {}
    throughput: dict[str, LogisticModel] = {}
    targets = {"power": power, "latency": latency, "throughput": throughput}
    for row in df.to_dict(orient="records"):
        metric = str(row["metric"]).strip().lower()
        if metric in targets:
            targets[metric][str(row["model_label"])] = LogisticModel.from_dict(row)
    return power, latency, throughput


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


def _parse_phase(s: str) -> Phase:
    """Parse a phase string like 'a', 'b', 'c' into a Phase enum."""
    return Phase(s.lower())


def _build_deployments_from_config(
    config: dict,
) -> tuple[list[OnlineModelDeployment], LLMInferenceWorkload]:
    """Build OnlineModelDeployment list and workload from config."""
    deployments: list[OnlineModelDeployment] = []
    specs: list[LLMInferenceModelSpec] = []

    for m in config["models"]:
        spec = LLMInferenceModelSpec(
            model_label=m["model_label"],
            num_replicas=m["num_replicas"],
            gpus_per_replica=m["gpus_per_replica"],
            feasible_batch_sizes=tuple(m.get("feasible_batch_sizes", BATCH_SET)),
            initial_batch_size=m.get("initial_batch_size", 128),
            itl_deadline_s=m.get("itl_deadline_s"),
        )
        specs.append(spec)

        endpoints = tuple(
            GPUEndpointMapping(
                host=ep["host"],
                port=ep.get("port", 4938),
                gpu_indices=tuple(ep.get("gpu_indices", [0])),
                phase=_parse_phase(ep.get("phase", "a")),
            )
            for ep in m.get("gpu_endpoints", [])
        )

        deployments.append(
            OnlineModelDeployment(
                spec=spec,
                vllm_base_url=m["vllm_base_url"],
                model_name=m["model_name"],
                gpu_endpoints=endpoints,
            )
        )

    workload = LLMInferenceWorkload(models=tuple(specs))
    return deployments, workload


def main(args: argparse.Namespace) -> None:
    project_dir = Path(__file__).resolve().parent.parent.parent
    case_dir = Path(__file__).resolve().parent.parent / "ieee13"
    save_dir = project_dir / "outputs" / "online_ofo"
    save_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(save_dir / "console_output.txt", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    v_min = config.get("v_min", 0.95)
    v_max = config.get("v_max", 1.05)
    dc_bus = config.get("dc_bus", "671")
    dt_dc = Fraction(1, 10)
    dt_ctrl = Fraction(1)
    t_total_s = args.duration

    TAP_STEP = 0.00625
    tap_ratios = config.get("initial_taps", {"a": 14, "b": 6, "c": 15})
    tap_schedule = TapPosition(
        a=1.0 + tap_ratios["a"] * TAP_STEP,
        b=1.0 + tap_ratios["b"] * TAP_STEP,
        c=1.0 + tap_ratios["c"] * TAP_STEP,
    ).at(t=0)

    logger.info("Building deployments from config...")
    deployments, workload = _build_deployments_from_config(config)
    model_labels = [d.model_label for d in deployments]

    logger.info("Loading pre-built requests...")
    requests_dir = Path(config.get("requests_dir", "data/online/requests"))
    if not requests_dir.is_absolute():
        requests_dir = project_dir / requests_dir
    requests_by_model = _load_requests(requests_dir, model_labels)

    logger.info("Loading logistic fits...")
    data_dir = Path(args.data_dir) if args.data_dir else project_dir / "data" / "generated"
    power_fits, latency_fits, throughput_fits = _load_logistic_fits_merged(data_dir / "logistic_fits.csv")

    logger.info("Setting up PowerStreamingClient...")
    servers_by_key: dict[str, ZeusdTcpConfig] = {}
    gpu_indices_by_key: dict[str, list[int]] = {}
    for d in deployments:
        for ep in d.gpu_endpoints:
            key = ep.endpoint_key
            if key not in gpu_indices_by_key:
                gpu_indices_by_key[key] = []
            for idx in ep.gpu_indices:
                if idx not in gpu_indices_by_key[key]:
                    gpu_indices_by_key[key].append(idx)
            servers_by_key[key] = ZeusdTcpConfig(
                host=ep.host,
                port=ep.port,
                gpu_indices=gpu_indices_by_key[key],
            )

    power_client = PowerStreamingClient(servers=list(servers_by_key.values()))

    aug_cfg = config.get("augmentation", {})
    augmentation = PowerAugmentationConfig(
        base_kw_per_phase=aug_cfg.get("base_kw_per_phase", 500.0),
        noise_frac=aug_cfg.get("noise_frac", 0.02),
        stagger_buffer_s=aug_cfg.get("stagger_buffer_s", 10.0),
        num_virtual_groups=aug_cfg.get("num_virtual_groups", 64),
        seed=aug_cfg.get("seed", 0),
    )

    load_gen_cfg = config.get("load_gen", {})
    load_gen = LoadGenerationConfig(
        max_output_tokens=load_gen_cfg.get("max_output_tokens", 512),
        concurrency_multiplier=load_gen_cfg.get("concurrency_multiplier", 3.0),
        itl_window_s=load_gen_cfg.get("itl_window_s", 1.0),
    )

    logger.info("Initializing OnlineDatacenter...")
    dc = OnlineDatacenter(
        deployments=deployments,
        power_client=power_client,
        augmentation=augmentation,
        load_gen=load_gen,
        requests_by_model=requests_by_model,
        dt_s=dt_dc,
    )

    logger.info("Initializing OpenDSSGrid...")
    grid = OpenDSSGrid(
        dss_case_dir=str(case_dir),
        dss_master_file="IEEE13Nodeckt.dss",
        dc_bus=dc_bus,
        dc_bus_kv=4.16,
        power_factor=0.95,
        dt_s=Fraction(1, 10),
        connection_type="wye",
        controls_off=True,
        tap_schedule=tap_schedule,
        freeze_regcontrols=True,
    )

    tap_ctrl = TapScheduleController(schedule=[], dt_s=dt_ctrl)

    ofo_ctrl = OFOBatchController.from_workload(
        workload=workload,
        power_fits=power_fits,
        latency_fits=latency_fits,
        throughput_fits=throughput_fits,
        primal_config=PrimalConfig(
            descent_step_size=0.1,
            w_latency=1.0,
            w_throughput=1e-3,
            w_switch=1.0,
            voltage_gradient_scale=1e6,
        ),
        voltage_dual_config=VoltageDualConfig(
            v_min=v_min,
            v_max=v_max,
            ascent_step_size=1.0,
        ),
        latency_dual_step_size=1.0,
        dt_s=dt_ctrl,
        sensitivity_update_interval=3600,
        sensitivity_perturbation_kw=100.0,
    )

    logger.info("Running online simulation for %d seconds...", t_total_s)
    coord = Coordinator(
        datacenter=dc,
        grid=grid,
        controllers=[tap_ctrl, ofo_ctrl],
        total_duration_s=t_total_s,
        dc_bus=dc_bus,
        live=True,
    )
    log = coord.run()

    stats = compute_allbus_voltage_stats(log.grid_states, v_min=v_min, v_max=v_max)
    logger.info("=== Voltage Statistics (all-bus) ===")
    logger.info("  voltage_violation_time = %.1f s", stats.violation_time_s)
    logger.info("  worst_vmin             = %.6f", stats.worst_vmin)
    logger.info("  worst_vmax             = %.6f", stats.worst_vmax)
    logger.info("  integral_violation     = %.5f pu·s", stats.integral_violation_pu_s)

    logger.info("=== Batch Schedule Summary ===")
    for label, batches in log.batch_log_by_model.items():
        if batches:
            avg = np.mean(batches)
            changes = sum(1 for i in range(1, len(batches)) if batches[i] != batches[i - 1])
            logger.info("  %s: avg_batch=%.1f, changes=%d", label, avg, changes)

    logger.info("Outputs saved to: %s", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online OFO closed-loop simulation (hardware-in-the-loop)")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file with deployment details.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="Simulation duration in seconds (default: 3600).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Toolkit-generated data directory (contains logistic_fits.csv). Defaults to data/generated/.",
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

    main(args)
