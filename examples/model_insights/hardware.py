"""H100 → B200 hardware-upgrade sweep on the IEEE 13 scenario.

Paired (model, parallelism) variants measured on both H100 and B200 at
the same DC footprint. The simulator only tracks an active-GPU count;
per-replica power and ITL come from hardware-specific logistic fits in
`LogisticModelStore`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import common as aic
import tyro

from openg2g.datacenter.config import ModelDeployment, ReplicaSchedule

logger = logging.getLogger("model_insights.hardware")


# (pair_label, spec_label, hardware_label, role).
# `role="anchor"` runs at `ANCHOR_REPLICAS` active GPUs. `role="match-peak"`
# sizes replicas so peak inference power at the variant's max-feasible-
# under-SLO batch matches the anchor's peak.
PAIRS: list[tuple[str, str, str, str]] = [
    ("qwen-8b-1gpu", "Qwen3-8B", "H100", "anchor"),
    ("qwen-8b-1gpu", "Qwen3-8B-B200", "B200", "match-peak"),
    ("qwen-32b-1gpu", "Qwen3-32B-1GPU-H100", "H100", "anchor"),
    ("qwen-32b-1gpu", "Qwen3-32B-1GPU-B200", "B200", "match-peak"),
    ("qwen-30b-a3b-instruct-1gpu", "Qwen3-30B-A3B-Instruct-1GPU", "H100", "anchor"),
    ("qwen-30b-a3b-instruct-1gpu", "Qwen3-30B-A3B-Instruct-1GPU-B200", "B200", "match-peak"),
]


ANCHOR_REPLICAS: int = 4800
COMMON_DEADLINE_MS: int = 50
DEFAULT_MODES: tuple[aic.Mode, ...] = ("baseline-no-tap", "ofo-no-tap")
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "hardware_b200.csv"


@dataclass
class Row:
    experiment: str
    variant: str
    pair: str
    hardware: str
    role: str
    mode: str
    seed: int
    num_replicas: int
    gpus_per_replica: int
    feasible_batch_min: int
    feasible_batch_max: int
    initial_batch: int
    peak_inference_kw: float
    violation_time_s: float
    integral_violation_pu_s: float
    worst_vmin: float
    worst_vmax: float
    mean_throughput_tps: float
    integrated_throughput_tokens: float
    itl_deadline_fraction: float
    achievable_power_range_mw: float


def _peak_inference_kw(spec, n_replicas, logistic_models) -> float:
    fit = logistic_models.power(spec.model_label)
    return float(fit.eval(max(spec.feasible_batch_sizes))) * n_replicas / 1e3


def _resolve_replicas(spec, role: str, anchor_peak_kw: float | None, logistic_models) -> int:
    if role == "anchor":
        return ANCHOR_REPLICAS
    if role == "match-peak":
        if anchor_peak_kw is None:
            raise ValueError("match-peak role requires an anchor peak first")
        return aic.compute_matched_peak_replicas(spec, anchor_peak_kw, logistic_models)
    raise ValueError(f"unknown role: {role}")


def run_variant(
    pair: str,
    spec_label: str,
    hardware: str,
    role: str,
    *,
    anchor_peak_kw: float | None,
    seed: int,
    shared: aic.SharedData,
    modes: tuple[aic.Mode, ...],
    total_duration_s: int,
    deadline_ms: int,
) -> tuple[list[Row], float]:
    base = aic.SPECS[spec_label]
    spec = aic.restrict_spec_by_deadline(base, shared.logistic_models, deadline_ms / 1000.0)
    num_replicas = _resolve_replicas(spec, role, anchor_peak_kw, shared.logistic_models)
    deployments = [
        (
            ModelDeployment(spec=spec, initial_batch_size=max(spec.feasible_batch_sizes)),
            ReplicaSchedule(initial=num_replicas),
        )
    ]
    peak_kw = _peak_inference_kw(spec, num_replicas, shared.logistic_models)

    apr_mw = aic.compute_achievable_power_range(deployments=deployments, logistic_models=shared.logistic_models)

    scenario = aic.build_scenario(deployments, shared=shared, overrides=aic.ScenarioOverrides(seed=seed))

    rows: list[Row] = []
    for mode in modes:
        result = aic.run_scenario(scenario, shared.logistic_models, mode=mode, total_duration_s=total_duration_s)
        rows.append(
            Row(
                experiment="hardware",
                variant=spec_label,
                pair=pair,
                hardware=hardware,
                role=role,
                mode=mode,
                seed=seed,
                num_replicas=num_replicas,
                gpus_per_replica=spec.gpus_per_replica,
                feasible_batch_min=min(spec.feasible_batch_sizes),
                feasible_batch_max=max(spec.feasible_batch_sizes),
                initial_batch=deployments[0][0].initial_batch_size,
                peak_inference_kw=peak_kw,
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
            "  [%s %s seed=%d mode=%s]  n_rep=%d peak=%.0fkW  viol=%.2fs  "
            "integral=%.4f  vmin=%.4f  thpt=%.1fk tok/s  itl_miss=%.2f%%  APR=%.2f MW",
            pair,
            hardware,
            seed,
            mode,
            num_replicas,
            peak_kw,
            result.voltage.violation_time_s,
            result.voltage.integral_violation_pu_s,
            result.voltage.worst_vmin,
            result.performance.mean_throughput_tps / 1e3,
            result.performance.itl_deadline_fraction * 100.0,
            apr_mw,
        )
    return rows, peak_kw


def main(
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    total_duration_s: int = aic.TOTAL_DURATION_S,
    modes: tuple[
        Literal["baseline-no-tap", "baseline-tap-change", "ofo-no-tap", "ofo-tap-change"], ...
    ] = DEFAULT_MODES,
    deadline_ms: int = COMMON_DEADLINE_MS,
    only_pair: str | None = None,
    out: Path = DEFAULT_OUT,
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    shared = aic.load_shared_data()
    out.parent.mkdir(parents=True, exist_ok=True)

    pairs = [p for p in PAIRS if (only_pair is None or p[0] == only_pair)]
    if not pairs:
        raise ValueError(f"No pairs match filter: only_pair={only_pair}")

    all_rows: list[Row] = []
    anchor_peak_by_pair: dict[str, float] = {}
    for pair, spec_label, hardware, role in pairs:
        logger.info("=== pair=%s  variant=%s (%s, role=%s) ===", pair, spec_label, hardware, role)
        for seed in seeds:
            anchor_peak = anchor_peak_by_pair.get(pair) if role == "match-peak" else None
            rows, peak_kw = run_variant(
                pair,
                spec_label,
                hardware,
                role,
                anchor_peak_kw=anchor_peak,
                seed=seed,
                shared=shared,
                modes=modes,
                total_duration_s=total_duration_s,
                deadline_ms=deadline_ms,
            )
            if role == "anchor" and pair not in anchor_peak_by_pair:
                anchor_peak_by_pair[pair] = peak_kw
            all_rows.extend(rows)

    aic.write_csv(all_rows, out)
    logger.info("Wrote %d rows to %s", len(all_rows), out)


if __name__ == "__main__":
    tyro.cli(main)
