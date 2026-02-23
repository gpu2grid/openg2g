"""Online baseline (no OFO control) simulation using real GPUs.

Connects to live vLLM servers and zeusd instances for hardware-in-the-loop
baseline measurement.  Power readings from a small number of real GPUs are
augmented to datacenter scale using temporal staggering.

Two modes correspond to two baselines:
  no-tap       Fixed tap positions throughout.
  tap-change   Regulator taps change at scheduled times.

An optional batch size schedule can drive controlled batch size changes
via `BatchSizeScheduleController`.

Usage:
    python examples/online/run_baseline.py --config examples/online/online_config.example.yaml --mode no-tap
"""

from __future__ import annotations

import argparse
import json
import logging
from fractions import Fraction
from pathlib import Path

import yaml
from zeus.monitor.power_streaming import PowerStreamingClient
from zeus.utils.zeusd import ZeusdConfig

from openg2g.controller.batch_size_schedule import BatchSizeChange, BatchSizeSchedule, BatchSizeScheduleController
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.online import (
    GPUEndpointMapping,
    LoadGenerationConfig,
    OnlineDatacenter,
    OnlineModelDeployment,
    PowerAugmentationConfig,
)
from openg2g.grid.base import Phase
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import compute_allbus_voltage_stats
from openg2g.models.spec import LLMInferenceModelSpec
from openg2g.types import TapPosition, TapSchedule

logger = logging.getLogger("run_baseline")

MAX_BATCH_SIZE = 512


def _batch_set_from_config(m: dict) -> tuple[int, ...]:
    """Derive feasible batch sizes from a model config entry."""
    max_bs = m.get("max_batch_size", MAX_BATCH_SIZE)
    return tuple(range(1, int(max_bs) + 1))


TAP_STEP = 0.00625


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
) -> list[OnlineModelDeployment]:
    """Build OnlineModelDeployment list from config."""
    deployments: list[OnlineModelDeployment] = []

    for m in config["models"]:
        spec = LLMInferenceModelSpec(
            model_label=m["model_label"],
            num_replicas=m["num_replicas"],
            gpus_per_replica=m["gpus_per_replica"],
            feasible_batch_sizes=_batch_set_from_config(m),
            initial_batch_size=m.get("initial_batch_size", 128),
            itl_deadline_s=m["itl_deadline_s"],
        )

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

    return deployments


def _build_initial_taps(config: dict) -> TapPosition:
    """Build initial tap position from config."""
    tap_ratios = config.get("initial_taps", {"a": 14, "b": 6, "c": 15})
    return TapPosition(
        a=1.0 + tap_ratios["a"] * TAP_STEP,
        b=1.0 + tap_ratios["b"] * TAP_STEP,
        c=1.0 + tap_ratios["c"] * TAP_STEP,
    )


def _build_tap_ctrl_schedule(config: dict, mode: str) -> TapSchedule:
    """Build tap controller schedule from config and mode."""
    if mode == "no-tap":
        return TapSchedule(())

    tap_changes = config.get("tap_changes", [])
    schedule = TapSchedule(())
    for entry in tap_changes:
        schedule = schedule | TapPosition(
            a=1.0 + entry["a"] * TAP_STEP,
            b=1.0 + entry["b"] * TAP_STEP,
            c=1.0 + entry["c"] * TAP_STEP,
        ).at(t=entry["t"])
    return schedule


def _build_batch_schedule(config: dict) -> dict[str, BatchSizeSchedule]:
    """Build per-model batch size schedules from config.

    Config format under `batch_schedule`:
        {
            "model_label": [
                {"t": 40, "batch_size": 48},
                {"t": 60, "batch_size": 32, "ramp_up_rate": 4}
            ]
        }
    """
    raw = config.get("batch_schedule", {})
    if not raw:
        return {}

    schedules: dict[str, BatchSizeSchedule] = {}
    for label, entries in raw.items():
        schedule: BatchSizeSchedule | None = None
        for entry in sorted(entries, key=lambda e: e["t"]):
            change = BatchSizeChange(
                batch_size=entry["batch_size"],
                ramp_up_rate=entry.get("ramp_up_rate", 0.0),
            ).at(t=entry["t"])
            schedule = change if schedule is None else (schedule | change)
        if schedule is not None:
            schedules[label] = schedule
    return schedules


def main(args: argparse.Namespace) -> None:
    mode = args.mode
    project_dir = Path(__file__).resolve().parent.parent.parent
    case_dir = Path(__file__).resolve().parent.parent / "ieee13"
    save_dir = project_dir / "outputs" / f"online_baseline_{mode}"
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

    initial_taps = _build_initial_taps(config)
    tap_ctrl_schedule = _build_tap_ctrl_schedule(config, mode)

    logger.info("Building deployments from config...")
    deployments = _build_deployments_from_config(config)
    model_labels = [d.model_label for d in deployments]

    logger.info("Loading pre-built requests...")
    requests_dir = Path(config.get("requests_dir", "data/online/requests"))
    if not requests_dir.is_absolute():
        requests_dir = project_dir / requests_dir
    requests_by_model = _load_requests(requests_dir, model_labels)

    logger.info("Setting up PowerStreamingClient...")
    servers_by_key: dict[str, ZeusdConfig] = {}
    gpu_indices_by_key: dict[str, list[int]] = {}
    for d in deployments:
        for ep in d.gpu_endpoints:
            key = ep.endpoint_key
            if key not in gpu_indices_by_key:
                gpu_indices_by_key[key] = []
            for idx in ep.gpu_indices:
                if idx not in gpu_indices_by_key[key]:
                    gpu_indices_by_key[key].append(idx)
            servers_by_key[key] = ZeusdConfig.tcp(
                ep.host,
                ep.port,
                gpu_indices=gpu_indices_by_key[key],
                cpu_indices=[],
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
        initial_tap_position=initial_taps,
    )

    controllers = []

    tap_ctrl = TapScheduleController(schedule=tap_ctrl_schedule, dt_s=dt_ctrl)
    controllers.append(tap_ctrl)

    batch_schedules = _build_batch_schedule(config)
    if batch_schedules:
        batch_ctrl = BatchSizeScheduleController(schedules=batch_schedules, dt_s=dt_ctrl)
        controllers.append(batch_ctrl)
        logger.info("BatchSizeScheduleController active for %d models", len(batch_schedules))

    logger.info("Running online baseline simulation (mode=%s) for %d seconds...", mode, t_total_s)
    coord = Coordinator(
        datacenter=dc,
        grid=grid,
        controllers=controllers,
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
    logger.info("  integral_violation     = %.4f pu·s", stats.integral_violation_pu_s)

    logger.info("Outputs saved to: %s", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online baseline simulation (no OFO control, hardware-in-the-loop)")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file with deployment details.",
    )
    parser.add_argument(
        "--mode",
        choices=["no-tap", "tap-change"],
        default="no-tap",
        help="Baseline variant: 'no-tap' (fixed taps) or 'tap-change' (scheduled tap changes).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="Simulation duration in seconds (default: 3600).",
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
