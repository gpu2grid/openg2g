"""Model size and architecture sweep on the IEEE 13 scenario.

Runs a model ladder on both B200 and H100 (selectable via `--gpu`) and
writes one CSV per GPU (`model_size_b200.csv`, `model_size_h100.csv`).

Both `lm-arena-chat` and `gpqa` tasks are represented to cover chat and
reasoning workloads. Each variant uses a per-variant ITL deadline and a
pre-computed match-peak replica count so peak inference power at the
variant's largest feasible batch equals a shared 3.12 MW anchor (Qwen 3
8B at 4,800 replicas on H100).
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

logger = logging.getLogger("model_insights.model_size")


# Variant catalogs, indexed by GPU. Each entry is
# (spec_label, num_replicas, deadline_ms); replica counts are match-peak
# sized to a 3.12 MW anchor (Qwen 3 8B at 4,800 replicas on H100).
VARIANTS_B200: list[tuple[str, int, int]] = [
    ("Qwen3-8B-B200", 4326, 50),  # 1 GPU, B200 BF16, lm-arena-chat
    ("Qwen3-32B-1GPU-B200", 3361, 50),  # 1 GPU, B200 BF16, lm-arena-chat
    ("Qwen3-30B-A3B-Instruct-1GPU-B200", 3483, 50),  # 1 GPU MoE, B200 BF16, lm-arena-chat
    ("GPT-OSS-120B-B200-2GPU", 2092, 50),  # 2 GPU, B200 MXFP4, gpqa (reasoning)
    ("Qwen3-235B-A22B-Instruct-8GPU-B200", 632, 100),  # 8 GPU MoE, B200 BF16, lm-arena-chat
]

VARIANTS_H100: list[tuple[str, int, int]] = [
    ("Qwen3-8B", 4800, 50),  # 1 GPU, H100 BF16, lm-arena-chat (anchor)
    ("Qwen3-32B-1GPU-H100", 5206, 50),  # 1 GPU, H100 BF16, lm-arena-chat
    ("GPT-OSS-120B-H100-2GPU", 2489, 50),  # 2 GPU, H100 MXFP4, gpqa (reasoning)
    ("Qwen3-235B-A22B-Instruct-8GPU", 773, 100),  # 8 GPU MoE, H100 BF16, lm-arena-chat
    ("Llama-3.1-70B", 1264, 100),  # 4 GPU, H100 BF16, lm-arena-chat (H100-only)
    ("Llama-3.1-405B", 637, 120),  # 8 GPU, H100 FP8, lm-arena-chat (H100-only)
]

VARIANTS_BY_GPU: dict[str, list[tuple[str, int, int]]] = {
    "b200": VARIANTS_B200,
    "h100": VARIANTS_H100,
}

DEFAULT_MODES: tuple[aic.Mode, ...] = ("baseline-no-tap", "ofo-no-tap")
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class Row:
    experiment: str
    variant: str
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
    spec_label: str,
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

    scenario = aic.build_scenario(deployments, shared=shared, overrides=aic.ScenarioOverrides(seed=seed))
    rows: list[Row] = []
    for mode in modes:
        result = aic.run_scenario(scenario, shared.logistic_models, mode=mode, total_duration_s=total_duration_s)
        rows.append(
            Row(
                experiment="model_size",
                variant=spec_label,
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
            "  [%s dl=%dms seed=%d mode=%s]  viol=%.2fs  integral=%.4f  "
            "vmin=%.4f  thpt=%.1fk tok/s  itl_miss=%.2f%%  APR=%.2f MW",
            spec_label,
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
    override_deadline_ms: int | None = None,
    only: tuple[str, ...] | None = None,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    """Run Experiment A on one or both GPU ladders.

    Args:
        seeds: seeds to run per variant.
        total_duration_s: scenario duration; default is the master preset.
        modes: which controller modes to evaluate per seed.
        gpu: which GPU ladder(s) to run. Default runs both and writes one
            CSV per GPU.
        override_deadline_ms: when set, applies this deadline to every
            variant (for ablation studies). When None, each variant uses
            its VARIANTS-declared deadline.
        only: subset of spec labels to run (default: all variants in the
            selected ladders).
        outdir: directory for per-GPU CSV outputs (`model_size_<gpu>.csv`).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    shared = aic.load_shared_data()
    outdir.mkdir(parents=True, exist_ok=True)

    targets = ["b200", "h100"] if gpu == "both" else [gpu]
    for gpu_tag in targets:
        variants = [(label, n, d) for label, n, d in VARIANTS_BY_GPU[gpu_tag] if (only is None or label in only)]
        if not variants:
            logger.warning("No variants match filter for gpu=%s only=%s; skipping", gpu_tag, only)
            continue
        if override_deadline_ms is not None:
            variants = [(label, n, override_deadline_ms) for label, n, _ in variants]
            logger.info("override_deadline_ms=%d applied to every variant", override_deadline_ms)
        logger.info(
            "=== GPU=%s  %d variants × %d seeds × %d modes on IEEE 13 ===",
            gpu_tag.upper(),
            len(variants),
            len(seeds),
            len(modes),
        )

        rows: list[Row] = []
        for spec_label, n_replicas, deadline_ms in variants:
            logger.info("  variant: %s × %d replicas @ %d ms", spec_label, n_replicas, deadline_ms)
            for seed in seeds:
                rows.extend(
                    run_variant(
                        spec_label,
                        n_replicas,
                        seed=seed,
                        shared=shared,
                        modes=modes,
                        total_duration_s=total_duration_s,
                        deadline_ms=deadline_ms,
                    )
                )

        out_path = outdir / f"model_size_{gpu_tag}.csv"
        _write_csv(rows, out_path)
        logger.info("Wrote %d rows to %s", len(rows), out_path)


def _write_csv(rows: list[Row], path: Path) -> None:
    fields = list(Row.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: getattr(r, k) for k in fields})


if __name__ == "__main__":
    tyro.cli(main)
