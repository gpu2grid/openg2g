"""Parallelism sweep on the IEEE 13 scenario.

Runs parallelism-pair sweeps on both B200 and H100 (selectable via
`--gpu`) and writes one CSV per GPU (`parallelism_b200.csv`,
`parallelism_h100.csv`).

B200 pairs:
    GPT-OSS 120B on gpqa, 50 ms ITL: 1 GPU vs 2 GPU
    Qwen 3 235B A22B Thinking on gpqa, 100 ms ITL: 4 GPU vs 8 GPU

H100 pair:
    Qwen 3 30B A3B Instruct on lm-arena-chat, 100 ms ITL: 1 GPU vs 2 GPU

Each variant is match-peak sized to the shared 3.12 MW anchor at the
largest batch meeting its ITL target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import common as aic
import tyro

from openg2g.datacenter.config import ModelDeployment, ReplicaSchedule

logger = logging.getLogger("model_insights.parallelism")


# (pair_label, spec_label, parallelism_degree_label, deadline_ms)
PAIRS_B200: list[tuple[str, str, str, int]] = [
    ("gpt-oss-120b", "GPT-OSS-120B-B200-1GPU", "1 GPU", 50),
    ("gpt-oss-120b", "GPT-OSS-120B-B200-2GPU", "2 GPU", 50),
    ("qwen-235b-a22b-thinking", "Qwen3-235B-A22B-Thinking-4GPU-B200", "4 GPU", 100),
    ("qwen-235b-a22b-thinking", "Qwen3-235B-A22B-Thinking-8GPU-B200", "8 GPU", 100),
]

PAIRS_H100: list[tuple[str, str, str, int]] = [
    ("qwen-30b-a3b-instruct", "Qwen3-30B-A3B-Instruct-1GPU", "1 GPU", 100),
    ("qwen-30b-a3b-instruct", "Qwen3-30B-A3B-Instruct-2GPU", "2 GPU", 100),
]

PAIRS_BY_GPU: dict[str, list[tuple[str, str, str, int]]] = {
    "b200": PAIRS_B200,
    "h100": PAIRS_H100,
}


ANCHOR_PEAK_KW: float = 3_120.0

DEFAULT_MODES: tuple[aic.Mode, ...] = ("baseline-no-tap", "ofo-no-tap")
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class Row:
    experiment: str
    variant: str
    pair: str
    parallelism_label: str
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
    num_replicas: int,
    parallelism_label: str,
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

    scenario = aic.build_scenario(deployments, shared=shared, overrides=aic.ScenarioOverrides(seed=seed))

    rows: list[Row] = []
    for mode in modes:
        result = aic.run_scenario(scenario, shared.logistic_models, mode=mode, total_duration_s=total_duration_s)
        rows.append(
            Row(
                experiment="parallelism",
                variant=spec_label,
                pair=pair,
                parallelism_label=parallelism_label,
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
            "  [%s %s n_rep=%d dl=%dms seed=%d mode=%s]  viol=%.2fs  integral=%.4f  "
            "vmin=%.4f  thpt=%.1fk tok/s  itl_miss=%.2f%%  APR=%.2f MW",
            pair,
            parallelism_label,
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
    gpu: Literal["both", "b200", "h100"] = "both",
    only_pair: str | None = None,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    shared = aic.load_shared_data()
    outdir.mkdir(parents=True, exist_ok=True)

    targets = ["b200", "h100"] if gpu == "both" else [gpu]
    for gpu_tag in targets:
        pairs = [p for p in PAIRS_BY_GPU[gpu_tag] if (only_pair is None or p[0] == only_pair)]
        if not pairs:
            logger.warning("No pairs match filter for gpu=%s only_pair=%s; skipping", gpu_tag, only_pair)
            continue

        rows: list[Row] = []
        for pair, spec_label, parallelism_label, deadline_ms in pairs:
            base = aic.SPECS[spec_label]
            spec = aic.restrict_spec_by_deadline(base, shared.logistic_models, deadline_ms / 1000.0)
            num_replicas = aic.compute_matched_peak_replicas(spec, ANCHOR_PEAK_KW, shared.logistic_models)
            logger.info(
                "=== gpu=%s  pair=%s  variant=%s (%s)  n_replicas=%d (match-peak @ %.0f kW, dl=%d ms) ===",
                gpu_tag.upper(),
                pair,
                spec_label,
                parallelism_label,
                num_replicas,
                ANCHOR_PEAK_KW,
                deadline_ms,
            )
            for seed in seeds:
                rows.extend(
                    run_variant(
                        pair,
                        spec_label,
                        num_replicas,
                        parallelism_label,
                        seed=seed,
                        shared=shared,
                        modes=modes,
                        total_duration_s=total_duration_s,
                        deadline_ms=deadline_ms,
                    )
                )

        out_path = outdir / f"parallelism_{gpu_tag}.csv"
        aic.write_csv(rows, out_path)
        logger.info("Wrote %d rows to %s", len(rows), out_path)


if __name__ == "__main__":
    tyro.cli(main)
