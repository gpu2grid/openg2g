"""Build OpenG2G simulation artifacts from ML.ENERGY benchmark data.

Uses a JSON config that specifies exact model IDs, tasks, GPU counts,
and simulation labels.  No automatic ID derivation or guessing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from mlenergy_data.modeling import ITLMixtureModel, LogisticModel
from mlenergy_data.records import LLMRuns

from openg2g.datacenter.offline import PowerTraceStore

logger = logging.getLogger("build_mlenergy_data")


def _emit_trace_bank(*, tl: pd.DataFrame, dt_s: float, out_dir: Path) -> pd.DataFrame:
    if tl.empty:
        raise ValueError("No timeline rows extracted from selected runs")

    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    keys = [
        PowerTraceStore.MANIFEST_COL_MODEL_LABEL,
        PowerTraceStore.MANIFEST_COL_NUM_GPUS,
        PowerTraceStore.MANIFEST_COL_BATCH_SIZE,
    ]
    for key, g in tl.groupby(keys, dropna=False):
        assert isinstance(key, tuple)
        model_label, num_gpus, batch = str(key[0]), cast(int, key[1]), cast(int, key[2])
        series_list: list[tuple[np.ndarray, np.ndarray]] = []
        t_ends: list[float] = []

        for _run_index, rg in g.groupby("run_index"):
            rr = rg.sort_values("relative_time_s")
            t = rr["relative_time_s"].to_numpy(dtype=float)
            p = rr["value"].to_numpy(dtype=float)
            if t.size < 2:
                continue
            series_list.append((t, p))
            t_ends.append(float(t[-1]))

        if not series_list:
            continue

        t_end = float(np.median(np.asarray(t_ends, dtype=float)))
        grid = np.arange(0.0, t_end + 1e-12, float(dt_s), dtype=float)
        mats: list[np.ndarray] = []
        for t, p in series_list:
            mats.append(np.interp(grid, t, p, left=p[0], right=p[-1]))
        mat = np.vstack(mats)
        p_med = np.median(mat, axis=0)

        trace_name = f"{model_label}_num_gpus_{int(num_gpus)}_max_num_seqs_{int(batch)}.csv"
        pd.DataFrame({PowerTraceStore.TRACE_COL_TIME: grid, PowerTraceStore.TRACE_COL_POWER: p_med}).to_csv(
            traces_dir / trace_name,
            index=False,
        )

        summary_rows.append(
            {
                PowerTraceStore.MANIFEST_COL_MODEL_LABEL: str(model_label),
                PowerTraceStore.MANIFEST_COL_NUM_GPUS: int(num_gpus),
                PowerTraceStore.MANIFEST_COL_BATCH_SIZE: int(batch),
                "n_runs": int(len(series_list)),
                "duration_s_median": t_end,
                "power_total_w_median": float(np.median(p_med)),
                PowerTraceStore.MANIFEST_COL_TRACE_FILE: f"traces/{trace_name}",
            }
        )

    if not summary_rows:
        raise ValueError("No trace profiles emitted from timeline table")
    return pd.DataFrame(summary_rows).sort_values(["model_label", "num_gpus", "max_num_seqs"])


def _emit_logistic_fits(
    subsets_by_label: dict[str, LLMRuns],
    out_dir: Path,
    *,
    fit_exclude_batches: dict[str, set[int]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label, group in subsets_by_label.items():
        num_gpus = group[0].num_gpus
        by_batch: dict[int, list[tuple[float, float, float]]] = {}
        exclude = fit_exclude_batches.get(str(model_label), set())
        for r in group:
            if r.max_num_seqs in exclude:
                continue
            by_batch.setdefault(r.max_num_seqs, []).append(
                (r.avg_power_watts, r.mean_itl_ms / 1000.0, r.output_throughput_tokens_per_sec)
            )

        batches = sorted(by_batch.keys())
        if not batches:
            continue

        x = np.log2(np.array(batches, dtype=float).clip(min=1))
        for metric_name, idx in [("power", 0), ("latency", 1), ("throughput", 2)]:
            y = np.array([float(np.median([t[idx] for t in by_batch[b]])) for b in batches])
            fit = LogisticModel.fit(x, y)
            rows.append(
                {
                    "model_label": str(model_label),
                    "num_gpus": int(num_gpus),
                    "metric": metric_name,
                    "L": fit.L,
                    "x0": fit.x0,
                    "k": fit.k,
                    "b0": fit.b0,
                }
            )

    if not rows:
        raise ValueError("No logistic fit rows produced")
    df = pd.DataFrame(rows).sort_values(["model_label", "num_gpus", "metric"]).reset_index(drop=True)
    df.to_csv(out_dir / "logistic_fits.csv", index=False)
    return df


def _emit_latency_fits(
    itl: pd.DataFrame,
    out_dir: Path,
    *,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    if itl.empty:
        raise ValueError("No ITL samples provided")

    rows: list[dict[str, Any]] = []
    for key, g in itl.groupby(["model_label", "num_gpus", "max_num_seqs"], dropna=False):
        assert isinstance(key, tuple)
        model_label, num_gpus, batch = str(key[0]), cast(int, key[1]), cast(int, key[2])
        fit = ITLMixtureModel.fit(
            g["itl_s"].to_numpy(dtype=float),
            max_samples=max_samples,
            seed=seed,
        )
        rows.append(
            {
                "model_label": str(model_label),
                "num_gpus": int(num_gpus),
                "max_num_seqs": int(batch),
                "itl_dist": "lognormal_mixture_2",
                **{f"itl_mix_{k}": v for k, v in fit.to_dict().items()},
            }
        )

    df = pd.DataFrame(rows).sort_values(["model_label", "num_gpus", "max_num_seqs"]).reset_index(drop=True)
    df.to_csv(out_dir / "latency_fits.csv", index=False)
    return df


def _generate_plots(
    *,
    out_dir: Path,
    config: list[dict[str, Any]],
    summary_df: pd.DataFrame,
    logistic_fits_df: pd.DataFrame,
    latency_fits_df: pd.DataFrame,
    itl_samples_df: pd.DataFrame,
    fit_exclude_batches: dict[str, set[int]],
) -> None:
    """Generate data characterization plots."""
    from plotting import plot_itl_distributions, plot_logistic_fits, plot_power_trajectories

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    all_labels = [str(entry["label"]) for entry in config]
    num_gpus_by_model = {str(entry["label"]): int(entry["num_gpus"]) for entry in config}
    trace_dir = out_dir / "traces"

    llama_labels = [lbl for lbl in all_labels if lbl.startswith("Llama")]
    qwen_labels = [lbl for lbl in all_labels if lbl.startswith("Qwen")]

    # Batch sizes used in the simulation (powers of 2 only).
    sim_batch_sizes = [8, 16, 32, 64, 128, 256, 512]

    # Fig. 2: Combined power trajectories for large models (8-GPU).
    fig2_labels = [lbl for lbl in all_labels if num_gpus_by_model[lbl] >= 8]
    if fig2_labels:
        plot_power_trajectories(
            trace_dir=trace_dir,
            model_labels=fig2_labels,
            num_gpus_by_model=num_gpus_by_model,
            batch_sizes=sim_batch_sizes,
            xlim_s=(0, 60),
            save_path=plot_dir / "power_trajectories_combined.png",
        )
        logger.info("Saved power_trajectories_combined.png")

    # Per-model power trajectories (supplementary).
    for label in all_labels:
        plot_power_trajectories(
            trace_dir=trace_dir,
            model_labels=[label],
            num_gpus_by_model=num_gpus_by_model,
            batch_sizes=sim_batch_sizes,
            save_path=plot_dir / f"power_trajectories_{label}.png",
        )
        logger.info("Saved power_trajectories_%s.png", label)

    # Fig. 3: Logistic fits — Llama models overlaid
    if llama_labels:
        plot_logistic_fits(
            summary_df=summary_df,
            fits_df=logistic_fits_df,
            model_labels=llama_labels,
            fit_exclude_batches=fit_exclude_batches,
            save_path=plot_dir / "logistic_fits_llama.png",
        )
        logger.info("Saved logistic_fits_llama.png")

    # Fig. 9: Logistic fits — Qwen models overlaid
    if qwen_labels:
        plot_logistic_fits(
            summary_df=summary_df,
            fits_df=logistic_fits_df,
            model_labels=qwen_labels,
            fit_exclude_batches=fit_exclude_batches,
            save_path=plot_dir / "logistic_fits_qwen.png",
        )
        logger.info("Saved logistic_fits_qwen.png")

    # Fig. 4 / Fig. 10: ITL distributions — one per model
    for label in all_labels:
        if label not in itl_samples_df["model_label"].values:
            continue
        plot_itl_distributions(
            latency_fits_df=latency_fits_df,
            itl_samples_df=itl_samples_df,
            model_label=label,
            batch_sizes=sim_batch_sizes,
            save_path=plot_dir / f"itl_distributions_{label}.png",
        )
        logger.info("Saved itl_distributions_%s.png", label)

    logger.info("All plots saved to: %s", plot_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OpenG2G simulation artifacts from raw benchmark data")
    parser.add_argument(
        "--mlenergy-data-dir",
        default=None,
        help="Path to compiled mlenergy-data directory. If omitted, loads from Hugging Face Hub.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON config file specifying models (see data/models.json)",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--itl-sample-cap-per-run", type=int, default=2048)
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate data characterization plots (default: --plot).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(args.config).read_text())
    if not isinstance(config, list) or not config:
        raise ValueError("Config must be a non-empty JSON array of model entries")

    unique_tasks = {str(entry["task"]) for entry in config}

    if args.mlenergy_data_dir:
        logger.info("Loading runs from %s (tasks: %s)", args.mlenergy_data_dir, sorted(unique_tasks))
        all_runs = LLMRuns.from_directory(args.mlenergy_data_dir, stable_only=False).task(*unique_tasks)
    else:
        logger.info("Loading runs from Hugging Face Hub (tasks: %s)", sorted(unique_tasks))
        all_runs = LLMRuns.from_hf(stable_only=False).task(*unique_tasks)
    if not all_runs:
        raise ValueError("No runs found in benchmark root for the specified tasks")

    subsets_by_label: dict[str, LLMRuns] = {}
    tl_frames: list[pd.DataFrame] = []
    itl_frames: list[pd.DataFrame] = []
    run_frames: list[pd.DataFrame] = []

    for entry in config:
        model_id = str(entry["model_id"])
        num_gpus = int(entry["num_gpus"])
        gpu = str(entry["gpu"])
        label = str(entry["label"])
        batch_sizes: list[int] = [int(b) for b in entry["batch_sizes"]]

        subset = all_runs.model_id(model_id).gpu_model(gpu).num_gpus(num_gpus).max_num_seqs(*batch_sizes)
        if not subset:
            raise ValueError(
                f"Config entry matched zero runs: model_id={model_id!r}, "
                f"gpu={gpu!r}, num_gpus={num_gpus}, "
                f"batch_sizes={batch_sizes}"
            )
        subsets_by_label[label] = subset

        for run in subset:
            tl = run.timelines(metric="power.device_instant")
            tl["model_label"] = label
            tl["num_gpus"] = run.num_gpus
            tl["max_num_seqs"] = run.max_num_seqs
            tl["run_index"] = len(tl_frames)
            tl_frames.append(tl)

        itl = subset.inter_token_latencies()
        itl["model_label"] = label
        itl_frames.append(itl)

        rdf = subset.to_dataframe()
        rdf["model_label"] = label
        run_frames.append(rdf)

        logger.info(
            "  %s: %d runs (model_id=%s, gpu=%s, num_gpus=%d, batches=%s)",
            label,
            len(subset),
            model_id,
            gpu,
            num_gpus,
            sorted({r.max_num_seqs for r in subset}),
        )

    all_tl = pd.concat(tl_frames, ignore_index=True)
    itl_samples_df = pd.concat(itl_frames, ignore_index=True)
    run_df = pd.concat(run_frames, ignore_index=True)

    traces_summary = _emit_trace_bank(
        tl=all_tl,
        dt_s=float(args.dt_s),
        out_dir=out_dir,
    )
    fit_exclude_batches: dict[str, set[int]] = {}
    for entry in config:
        excl = entry.get("fit_exclude_batches")
        if excl:
            fit_exclude_batches[str(entry["label"])] = {int(b) for b in excl}

    logistic_fits_df = _emit_logistic_fits(
        subsets_by_label,
        out_dir,
        fit_exclude_batches=fit_exclude_batches,
    )

    latency_fits_df = _emit_latency_fits(
        itl_samples_df,
        out_dir,
        max_samples=int(args.itl_sample_cap_per_run),
        seed=int(args.seed),
    )

    run_summary = (
        run_df.groupby(["model_label", "num_gpus", "max_num_seqs"], dropna=False)
        .agg(
            n_runs=("model_label", "count"),
            avg_power_watts_median=("avg_power_watts", "median"),
            avg_itl_ms_median=("median_itl_ms", "median"),
            throughput_toks_s_median=("output_throughput_tokens_per_sec", "median"),
        )
        .reset_index()
        .sort_values(["model_label", "num_gpus", "max_num_seqs"])
    )
    run_summary.to_csv(out_dir / "summary.csv", index=False)
    traces_summary.to_csv(out_dir / "traces_summary.csv", index=False)

    logger.info("Wrote OpenG2G dataset to: %s", out_dir)
    logger.info("  traces: %s", out_dir / "traces")
    logger.info("  latency_fits: %s", out_dir / "latency_fits.csv")
    logger.info("  logistic_fits: %s", out_dir / "logistic_fits.csv")
    logger.info("  summary: %s", out_dir / "summary.csv")

    if args.plot:
        _generate_plots(
            out_dir=out_dir,
            config=config,
            summary_df=run_summary,
            logistic_fits_df=logistic_fits_df,
            latency_fits_df=latency_fits_df,
            itl_samples_df=itl_samples_df,
            fit_exclude_batches=fit_exclude_batches,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
