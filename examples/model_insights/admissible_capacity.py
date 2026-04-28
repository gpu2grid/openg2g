"""Experiment B — DC capacity under an ITL SLO on the IEEE 13 scenario.

Question: for a chosen ITL SLO, how large a datacenter can you put at bus 671
before the IEEE 13 feeder's voltage constraint blocks you? And how much extra
capacity does a coordination controller (OFO) unlock vs a controller-less
baseline?

For each SLO in {40, 60, 80, 120, 200} ms we binary-search the largest
replica count of `Qwen 3 Coder 30B A3B` on B200 (`sourcegraph-fim`) such that
the 3,600-s simulation's integral voltage violation stays at or below
`CAPACITY_THRESHOLD_PU_S`. Search is run once per mode (baseline-no-tap and
ofo-no-tap) on a fixed seed. The found `R_max` is then re-run across five
seeds to give a variance estimate.

The deliverable is a curve `R_max(SLO)` for each mode — the gap between the
two curves is the DC capacity that a coordination controller unlocks at the
chosen SLO. Written to a CSV compatible with `plots.py`.
"""

from __future__ import annotations

import csv
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import common as aic
import tyro

from openg2g.datacenter.config import ModelDeployment, ReplicaSchedule

logger = logging.getLogger("model_insights.admissible_capacity")


BASES_BY_GPU: dict[str, str] = {
    "b200": "Qwen3-Coder-30B-A3B-B200",
    "h100": "Llama-3.1-70B",
}
DEADLINES_MS: tuple[int, ...] = (40, 60, 80, 120, 200)
DEFAULT_MODES: tuple[aic.Mode, ...] = ("baseline-no-tap", "ofo-no-tap")
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "outputs"

# Grid-feasibility is a threshold on the existing integral voltage
# violation (pu·s). This captures both duration and severity and is what
# grid standards like IEEE 1547 actually specify. Integral is U-shaped
# in R (over-voltage at small DC, under-voltage at large DC), so we scan
# left and right from an interior feasible point rather than a single
# monotone binary search.
FEASIBILITY_THRESHOLD_PU_S = 1.0

# Binary search bounds over replica count.
R_LO_INITIAL = 1
R_TOLERANCE = 20  # stop when hi - lo ≤ this

# Per-replica peak at B_max may be ~1 kW; IEEE 13 feeder saturates around
# 5-6 MW. R_HI is derived per-spec below from the logistic power fit so the
# search stays within OpenDSS's convergent range. Beyond that, OpenDSS
# returns nonsense voltages (seen experimentally: vmin values > 5 pu).
R_HI_HEADROOM_KW = 6_000.0

# Sanity bounds on reported voltages. Outside this range the OpenDSS solver
# has failed to converge and the reported vmin/vmax are meaningless — treat
# that sample as infeasible.
V_SANITY_LO = 0.5
V_SANITY_HI = 1.2

# Single seed used during the binary search (fast path); verification uses
# the full VERIFY_SEEDS set at the resolved R_max.
SEARCH_SEED = 0
VERIFY_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass
class Row:
    experiment: str
    variant: str
    mode: str
    seed: int
    deadline_ms: int
    endpoint: str  # "r_min" or "r_max"
    num_replicas: int
    gpus_per_replica: int
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


def _sim(
    spec,
    n_replicas: int,
    shared: aic.SharedData,
    *,
    mode: aic.Mode,
    seed: int,
    total_duration_s: int,
):
    """Single-run simulation with `n_replicas`. Returns the `VariantResult`."""
    gpu_budget = max(7200, n_replicas * spec.gpus_per_replica + 200)
    deployments = [
        (
            ModelDeployment(spec=spec, initial_batch_size=max(spec.feasible_batch_sizes)),
            ReplicaSchedule(initial=n_replicas),
        )
    ]
    scenario = aic.build_scenario(
        deployments,
        shared=shared,
        overrides=aic.ScenarioOverrides(seed=seed, total_gpu_capacity=gpu_budget),
    )
    return aic.run_scenario(scenario, shared.logistic_models, mode=mode, total_duration_s=total_duration_s)


def _is_feasible(result) -> bool:
    """Grid-acceptable iff the 3,600-s integral voltage violation stays at
    or below `FEASIBILITY_THRESHOLD_PU_S`, and voltage readings pass a
    convergence sanity check (OpenDSS can return absurd voltages near
    extreme overload)."""
    vmin = result.voltage.worst_vmin
    vmax = result.voltage.worst_vmax
    if not (V_SANITY_LO <= vmin <= V_SANITY_HI) or not (V_SANITY_LO <= vmax <= V_SANITY_HI):
        return False
    return result.voltage.integral_violation_pu_s <= FEASIBILITY_THRESHOLD_PU_S


def _r_hi_for_spec(spec, shared) -> int:
    """Upper-bound R so the search stays within OpenDSS's convergent regime."""
    per_rep_peak_w = float(shared.logistic_models.power(spec.model_label).eval(max(spec.feasible_batch_sizes)))
    return max(R_LO_INITIAL + 1, int(R_HI_HEADROOM_KW * 1000.0 / per_rep_peak_w))


def _find_interior(
    spec,
    shared,
    *,
    mode,
    total_duration_s,
    r_hi,
) -> int | None:
    """Find any R with integral ≤ threshold by sampling a few R values
    across [R_LO_INITIAL, r_hi]. Returns None if no point is feasible
    (meaning the scenario admits no DC size at all)."""
    probes = [int(r_hi * frac) for frac in (0.08, 0.17, 0.30, 0.50, 0.75)]
    best_r = None
    best_int = float("inf")
    for r in probes:
        res = _sim(spec, r, shared, mode=mode, seed=SEARCH_SEED, total_duration_s=total_duration_s)
        integ = res.voltage.integral_violation_pu_s
        ok = _is_feasible(res)
        logger.info(
            "    [%s] probe R=%d  int=%.3f  vmin=%.4f  vmax=%.4f  %s",
            mode,
            r,
            integ,
            res.voltage.worst_vmin,
            res.voltage.worst_vmax,
            "OK" if ok else "FAIL",
        )
        if ok and integ < best_int:
            best_r, best_int = r, integ
    return best_r


def _bisect_edge(
    spec,
    shared,
    *,
    mode,
    total_duration_s,
    lo: int,
    hi: int,
    direction: str,
) -> int:
    """Binary-search one boundary of the feasible R interval.

    `direction="up"`  → lo is feasible (interior), hi is infeasible (R too large).
                        Returns largest feasible R (R_max).
    `direction="down"` → hi is feasible (interior), lo is infeasible (R too small).
                        Returns smallest feasible R (R_min).
    """
    while hi - lo > R_TOLERANCE:
        mid = (lo + hi) // 2
        res = _sim(spec, mid, shared, mode=mode, seed=SEARCH_SEED, total_duration_s=total_duration_s)
        ok = _is_feasible(res)
        logger.info(
            "    [%s] bisect %-4s R=%d  int=%.3f  %s",
            mode,
            direction,
            mid,
            res.voltage.integral_violation_pu_s,
            "OK" if ok else "FAIL",
        )
        if direction == "up":
            if ok:
                lo = mid
            else:
                hi = mid
        else:  # "down"
            if ok:
                hi = mid
            else:
                lo = mid
    return lo if direction == "up" else hi


def _find_feasible_interval(
    spec,
    shared,
    *,
    mode,
    total_duration_s,
) -> tuple[int | None, int | None]:
    """Return (R_min, R_max) — the smallest and largest R such that
    integral voltage violation stays ≤ threshold. Either is `None` if no
    feasible R exists in [1, R_hi]."""
    r_hi = _r_hi_for_spec(spec, shared)
    interior = _find_interior(spec, shared, mode=mode, total_duration_s=total_duration_s, r_hi=r_hi)
    if interior is None:
        return None, None

    # Probe the ends. If either end is itself feasible, it's the boundary.
    lo_result = _sim(spec, R_LO_INITIAL, shared, mode=mode, seed=SEARCH_SEED, total_duration_s=total_duration_s)
    hi_result = _sim(spec, r_hi, shared, mode=mode, seed=SEARCH_SEED, total_duration_s=total_duration_s)

    r_min = (
        R_LO_INITIAL
        if _is_feasible(lo_result)
        else _bisect_edge(
            spec, shared, mode=mode, total_duration_s=total_duration_s, lo=R_LO_INITIAL, hi=interior, direction="down"
        )
    )
    r_max = (
        r_hi
        if _is_feasible(hi_result)
        else _bisect_edge(
            spec, shared, mode=mode, total_duration_s=total_duration_s, lo=interior, hi=r_hi, direction="up"
        )
    )
    return r_min, r_max


def _search_and_verify(
    base_spec_label: str,
    dl_ms: int,
    mode: aic.Mode,
    total_duration_s: int,
    seeds: tuple[int, ...],
) -> list[Row]:
    """Run the full search + seed verification for one (deadline, mode) pair.

    Structured as a top-level function so it can be dispatched to a
    `ProcessPoolExecutor`. Each worker re-loads shared data from the
    per-spec cache (cheap — pure disk reads).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    shared = aic.load_shared_data()
    base = aic.SPECS[base_spec_label]
    spec = aic.restrict_spec_by_deadline(base, shared.logistic_models, dl_ms / 1000.0)
    apr_mw = aic.compute_achievable_power_range(
        deployments=[
            (
                ModelDeployment(spec=spec, initial_batch_size=max(spec.feasible_batch_sizes)),
                ReplicaSchedule(initial=1),
            )
        ],
        logistic_models=shared.logistic_models,
    )

    logger.info(
        "  -- [%s dl=%dms mode=%s] feasibility scan (int ≤ %.2f pu·s) --",
        base_spec_label,
        dl_ms,
        mode,
        FEASIBILITY_THRESHOLD_PU_S,
    )
    r_min, r_max = _find_feasible_interval(spec, shared, mode=mode, total_duration_s=total_duration_s)
    logger.info("  >>> feasible R [%s dl=%dms mode=%s] = [%s, %s]", base_spec_label, dl_ms, mode, r_min, r_max)

    rows: list[Row] = []
    for endpoint, r in (("r_min", r_min), ("r_max", r_max)):
        if r is None:
            continue
        for seed in seeds:
            result = _sim(spec, r, shared, mode=mode, seed=seed, total_duration_s=total_duration_s)
            rows.append(
                Row(
                    experiment="admissible_capacity",
                    variant=f"{base_spec_label}_dl{dl_ms}ms",
                    mode=mode,
                    seed=seed,
                    deadline_ms=dl_ms,
                    endpoint=endpoint,
                    num_replicas=r,
                    gpus_per_replica=spec.gpus_per_replica,
                    feasible_batch_min=min(spec.feasible_batch_sizes),
                    feasible_batch_max=max(spec.feasible_batch_sizes),
                    initial_batch=max(spec.feasible_batch_sizes),
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
                "    [%s dl=%dms mode=%s] verify %s R=%d seed=%d: viol=%.2fs int=%.4f vmin=%.4f vmax=%.4f thpt=%.1fk",
                base_spec_label,
                dl_ms,
                mode,
                endpoint,
                r,
                seed,
                result.voltage.violation_time_s,
                result.voltage.integral_violation_pu_s,
                result.voltage.worst_vmin,
                result.voltage.worst_vmax,
                result.performance.mean_throughput_tps / 1e3,
            )
    return rows


def main(
    seeds: tuple[int, ...] = VERIFY_SEEDS,
    total_duration_s: int = aic.TOTAL_DURATION_S,
    modes: tuple[
        Literal["baseline-no-tap", "baseline-tap-change", "ofo-no-tap", "ofo-tap-change"], ...
    ] = DEFAULT_MODES,
    gpu: Literal["both", "b200", "h100"] = "both",
    deadlines_ms: tuple[int, ...] = DEADLINES_MS,
    max_workers: int | None = None,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    outdir.mkdir(parents=True, exist_ok=True)

    targets = ["b200", "h100"] if gpu == "both" else [gpu]
    for gpu_tag in targets:
        base_spec_label = BASES_BY_GPU[gpu_tag]
        tasks = [(dl, mode) for dl in deadlines_ms for mode in modes]
        workers = max_workers if max_workers is not None else len(tasks)
        logger.info(
            "=== GPU=%s base=%s  dispatching %d (deadline, mode) tasks across %d workers ===",
            gpu_tag.upper(),
            base_spec_label,
            len(tasks),
            workers,
        )

        rows: list[Row] = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_search_and_verify, base_spec_label, dl, mode, total_duration_s, seeds): (dl, mode)
                for dl, mode in tasks
            }
            for fut in as_completed(futures):
                dl, mode = futures[fut]
                rows.extend(fut.result())
                logger.info("  ++ finished (dl=%dms, mode=%s): GPU=%s", dl, mode, gpu_tag.upper())

        out_path = outdir / f"admissible_capacity_{gpu_tag}.csv"
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
