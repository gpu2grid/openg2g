"""Weight-precision sweep on the IEEE 13 scenario.

Sweeps bf16 vs FP8 on three matched-hardware pairs of the Qwen 3 235B
A22B family:

- Instruct, H100 8 GPU, `lm-arena-chat`.
- Thinking, H100 8 GPU, `gpqa`.
- Thinking, B200 4 GPU, `gpqa`.

Within each pair, replica count is match-peak sized so the bf16 variant's
peak inference power at its largest feasible batch matches the shared
3.12 MW anchor; the FP8 sibling runs at the same replica count, so the
DC footprint is identical within the pair.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import common as aic
import tyro

from openg2g.datacenter.config import ModelDeployment, ReplicaSchedule

logger = logging.getLogger("model_insights.precision")


# (pair_label, bf16_spec_label, fp8_spec_label, hardware_description, deadline_ms)
PAIRS: list[tuple[str, str, str, str, int]] = [
    (
        "qwen-235b-a22b-instruct-h100-8gpu",
        "Qwen3-235B-A22B-Instruct-8GPU",
        "Qwen3-235B-A22B-Instruct-FP8-8GPU",
        "H100 8 GPU (lm-arena-chat)",
        100,
    ),
    (
        "qwen-235b-a22b-thinking-h100-8gpu",
        "Qwen3-235B-A22B-Thinking-8GPU",
        "Qwen3-235B-A22B-Thinking-FP8-8GPU",
        "H100 8 GPU (gpqa)",
        100,
    ),
    (
        "qwen-235b-a22b-thinking-b200-4gpu",
        "Qwen3-235B-A22B-Thinking-4GPU-B200",
        "Qwen3-235B-A22B-Thinking-FP8-4GPU-B200",
        "B200 4 GPU (gpqa)",
        100,
    ),
]

ANCHOR_PEAK_KW: float = 3_120.0
DEFAULT_MODES: tuple[aic.Mode, ...] = ("baseline-no-tap", "ofo-no-tap")
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "precision_b200.csv"


@dataclass
class Row:
    experiment: str
    variant: str
    pair: str
    precision_label: str
    hardware: str
    mode: str
    seed: int
    num_replicas: int
    gpus_per_replica: int
    deadline_ms: int
    feasible_batch_min: int
    feasible_batch_max: int
    initial_batch: int
    violation_time_s: float
    integral_violation_pu_s: float
    worst_vmin: float
    worst_vmax: float
    mean_throughput_tps: float
    integrated_throughput_tokens: float
    itl_deadline_fraction: float
    achievable_power_range_mw: float


def run_variant(
    pair: str,
    spec_label: str,
    precision_label: str,
    hardware: str,
    num_replicas: int,
    *,
    seed: int,
    shared: aic.SharedData,
    modes: tuple[aic.Mode, ...],
    total_duration_s: int,
    deadline_ms: int,
) -> list[Row]:
    base = aic.SPECS[spec_label]
    spec = aic.restrict_spec_by_deadline(base, shared.logistic_models, deadline_ms / 1000.0)
    deployments = [
        (
            ModelDeployment(spec=spec, initial_batch_size=max(spec.feasible_batch_sizes)),
            ReplicaSchedule(initial=num_replicas),
        )
    ]

    apr_mw = aic.compute_achievable_power_range(deployments=deployments, logistic_models=shared.logistic_models)

    # Match-peak sizing can push replica × GPU count past the preset's
    # 7,200 GPU cap; provision the GPU budget for this deployment plus
    # headroom.
    gpu_budget = max(7200, num_replicas * spec.gpus_per_replica + 200)
    scenario = aic.build_scenario(
        deployments,
        shared=shared,
        overrides=aic.ScenarioOverrides(seed=seed, total_gpu_capacity=gpu_budget),
    )
    rows: list[Row] = []
    for mode in modes:
        result = aic.run_scenario(scenario, shared.logistic_models, mode=mode, total_duration_s=total_duration_s)
        rows.append(
            Row(
                experiment="precision",
                variant=spec_label,
                pair=pair,
                precision_label=precision_label,
                hardware=hardware,
                mode=mode,
                seed=seed,
                num_replicas=num_replicas,
                gpus_per_replica=spec.gpus_per_replica,
                deadline_ms=deadline_ms,
                feasible_batch_min=min(spec.feasible_batch_sizes),
                feasible_batch_max=max(spec.feasible_batch_sizes),
                initial_batch=deployments[0][0].initial_batch_size,
                violation_time_s=result.voltage.violation_time_s,
                integral_violation_pu_s=result.voltage.integral_violation_pu_s,
                worst_vmin=result.voltage.worst_vmin,
                worst_vmax=result.voltage.worst_vmax,
                mean_throughput_tps=result.performance.mean_throughput_tps,
                integrated_throughput_tokens=result.performance.integrated_throughput_tokens,
                itl_deadline_fraction=result.performance.itl_deadline_fraction,
                achievable_power_range_mw=apr_mw,
            )
        )
        logger.info(
            "  [%s (%s) n_rep=%d dl=%dms seed=%d mode=%s]  viol=%.2fs  integral=%.4f  "
            "vmin=%.4f  thpt=%.1fk tok/s  itl_miss=%.2f%%  APR=%.2f MW",
            spec_label,
            precision_label,
            num_replicas,
            deadline_ms,
            seed,
            mode,
            result.voltage.violation_time_s,
            result.voltage.integral_violation_pu_s,
            result.voltage.worst_vmin,
            result.performance.mean_throughput_tps / 1e3,
            result.performance.itl_deadline_fraction * 100.0,
            apr_mw,
        )
    return rows


def main(
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    total_duration_s: int = aic.TOTAL_DURATION_S,
    modes: tuple[
        Literal["baseline-no-tap", "baseline-tap-change", "ofo-no-tap", "ofo-tap-change"], ...
    ] = DEFAULT_MODES,
    out: Path = DEFAULT_OUT,
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    shared = aic.load_shared_data()

    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running %d (bf16, FP8) pairs × %d seeds × %d modes on IEEE 13",
        len(PAIRS),
        len(seeds),
        len(modes),
    )

    all_rows: list[Row] = []
    for pair, bf16_label, fp8_label, hw, deadline_ms in PAIRS:
        # Match-peak sizing uses the bf16 variant; FP8 runs at the same
        # replica count so DC footprint is identical within the pair.
        bf16_spec = aic.restrict_spec_by_deadline(aic.SPECS[bf16_label], shared.logistic_models, deadline_ms / 1000.0)
        n_replicas = aic.compute_matched_peak_replicas(bf16_spec, ANCHOR_PEAK_KW, shared.logistic_models)
        logger.info("=== pair=%s (%s)  n_replicas=%d (match-peak @ %.0f kW) ===", pair, hw, n_replicas, ANCHOR_PEAK_KW)
        for precision_label, spec_label in (("bf16", bf16_label), ("FP8", fp8_label)):
            for seed in seeds:
                all_rows.extend(
                    run_variant(
                        pair,
                        spec_label,
                        precision_label,
                        hw,
                        n_replicas,
                        seed=seed,
                        shared=shared,
                        modes=modes,
                        total_duration_s=total_duration_s,
                        deadline_ms=deadline_ms,
                    )
                )

    _write_csv(all_rows, out)
    logger.info("Wrote %d rows to %s", len(all_rows), out)


def _write_csv(rows: list[Row], path: Path) -> None:
    fields = list(Row.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: getattr(r, k) for k in fields})


if __name__ == "__main__":
    tyro.cli(main)
