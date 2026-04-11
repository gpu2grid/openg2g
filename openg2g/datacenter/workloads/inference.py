"""Inference workload: power traces, templates, ITL fits, and augmentation."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from mlenergy_data.modeling import ITLMixtureModel
from mlenergy_data.records import LLMRuns
from pydantic import BaseModel, ConfigDict

import openg2g
from openg2g.common import ThreePhase
from openg2g.datacenter.config import InferenceModelSpec
from openg2g.datacenter.layout import ServerLayout

logger = logging.getLogger(__name__)


class MLEnergySource(BaseModel):
    """Per-model ML.ENERGY benchmark data extraction settings.

    Attributes:
        model_label: Simulation label for the model.
        task: Benchmark task name (e.g. `"lm-arena-chat"`, `"gpqa"`).
        gpu: GPU model name (e.g. `"H100"`).
        batch_sizes: Batch sizes to extract from the benchmark data.
        fit_exclude_batch_sizes: Batch sizes to exclude from logistic
            curve fitting (but still included in trace extraction).
    """

    model_config = ConfigDict(frozen=True)

    model_label: str
    task: str
    gpu: str
    batch_sizes: tuple[int, ...]
    fit_exclude_batch_sizes: tuple[int, ...] = ()


@dataclass(frozen=True)
class InferenceTrace:
    """A single power trace measurement.

    Attributes:
        t_s: Time vector (seconds), monotonically increasing.
        power_w: Total power vector (watts) across all measured GPUs,
            same length as `t_s`.
        measured_gpus: Number of GPUs used in the measurement.
    """

    t_s: np.ndarray
    power_w: np.ndarray
    measured_gpus: int

    def __post_init__(self) -> None:
        if len(self.t_s) != len(self.power_w):
            raise ValueError(f"t_s and power_w must have the same length, got {len(self.t_s)} and {len(self.power_w)}")
        if len(self.t_s) < 5:
            raise ValueError("Trace too short (need at least 5 samples).")
        if self.measured_gpus < 1:
            raise ValueError(f"measured_gpus must be >= 1, got {self.measured_gpus}")


def _build_per_gpu_power_template(
    trace: InferenceTrace,
    *,
    dt_s: Fraction | float,
    duration_s: Fraction | float,
    steady_skip_s: float = 0.0,
) -> np.ndarray:
    """Build a per-GPU power template over [0, duration_s] by periodic repetition.

    Args:
        trace: Source power trace (total power across measured GPUs).
        dt_s: Simulation timestep in seconds.
        duration_s: Total simulation duration in seconds.
        steady_skip_s: Skip this many seconds from the start of the trace
            to avoid warm-up transients.

    Returns:
        1-D array of per-GPU power values at each simulation timestep.
    """
    trace_t = np.asarray(trace.t_s, float)
    trace_p_total = np.asarray(trace.power_w, float)

    mg = max(trace.measured_gpus, 1)
    p_per_gpu = trace_p_total / mg
    p_per_gpu = np.clip(p_per_gpu, 0.0, None)

    if steady_skip_s > 0.0:
        idx0 = np.searchsorted(trace_t, trace_t[0] + float(steady_skip_s))
        if idx0 < trace_t.size - 5:
            trace_t = trace_t[idx0:] - trace_t[idx0]
            p_per_gpu = p_per_gpu[idx0:]

    trace_t = trace_t - trace_t[0]
    period = float(trace_t[-1] - trace_t[0])
    if period <= 0:
        raise ValueError("Non-positive trace duration.")

    n_steps = int(np.ceil(float(duration_s) / float(dt_s))) + 1
    t_grid = np.arange(n_steps, dtype=float) * float(dt_s)
    t_mod = np.mod(t_grid, period)

    template = np.interp(t_mod, trace_t, p_per_gpu, left=p_per_gpu[0], right=p_per_gpu[-1])
    return np.clip(template, 0.0, None)


class ITLFitStore:
    """Per-model, per-batch-size ITL mixture distributions.

    Indexed by `(model_label, batch_size)`. Provides:

    - [`load`][.load]: load fits from a CSV produced by the data pipeline
    - [`distributions`][.distributions]: access as a nested dict
    - [`sample_avg`][.sample_avg]: sample a fleet-average ITL value

    Attributes:
        COL_MODEL_LABEL: Column name for model label in the CSV.
        COL_BATCH_SIZE: Column name for batch size in the CSV.
    """

    COL_MODEL_LABEL = "model_label"
    COL_BATCH_SIZE = "max_num_seqs"

    def __init__(
        self,
        distributions: dict[str, dict[int, ITLMixtureModel]],
        approx_sampling_thresh: int = 30,
    ) -> None:
        self._distributions = {
            str(label): {int(b): m for b, m in per_batch.items()} for label, per_batch in distributions.items()
        }
        self._approx_sampling_thresh = int(approx_sampling_thresh)

    @property
    def distributions(self) -> dict[str, dict[int, ITLMixtureModel]]:
        """Nested dict: `model_label -> batch_size -> ITLMixtureModel`."""
        return self._distributions

    def sample_avg(
        self,
        model_label: str,
        batch_size: int,
        n_replicas: int,
        rng: np.random.Generator,
    ) -> float:
        """Sample a fleet-average ITL for the given model and batch size.

        Uses `ITLMixtureModel.sample_avg` under the hood, with the
        `approx_sampling_thresh` set at construction time.

        Args:
            model_label: Model label string.
            batch_size: Current batch size.
            n_replicas: Number of active replicas.
            rng: NumPy random generator for sampling.

        Returns:
            Fleet-average ITL in seconds.

        Raises:
            KeyError: If model or batch size is not in the store.
        """
        model_dists = self._distributions.get(model_label)
        if model_dists is None:
            raise KeyError(f"No ITL distributions for model={model_label!r}")
        params = model_dists.get(int(batch_size))
        if params is None:
            raise KeyError(
                f"No ITL distributions for model={model_label!r}, batch={batch_size}. "
                f"Available={sorted(model_dists.keys())}"
            )
        return params.sample_avg(
            n_replicas=n_replicas,
            rng=rng,
            exact_threshold=self._approx_sampling_thresh,
        )

    @classmethod
    def load(cls, csv_path: Path | str, approx_sampling_thresh: int = 30) -> ITLFitStore:
        """Load ITL mixture fits from a CSV.

        Expected columns: `model_label`, `max_num_seqs`, plus the
        `itl_mix_*` parameter columns produced by
        `ITLMixtureModel.to_dict()`.

        Args:
            csv_path: Path to the latency fits CSV.
            approx_sampling_thresh: Replica count above which sampling
                uses a CLT normal approximation instead of drawing
                individual samples.
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        required_cols = [cls.COL_MODEL_LABEL, cls.COL_BATCH_SIZE]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} missing columns: {missing}. Got: {list(df.columns)}")

        distributions: dict[str, dict[int, ITLMixtureModel]] = {}
        for row in df.to_dict(orient="records"):
            label = str(row[cls.COL_MODEL_LABEL]).strip()
            batch = int(row[cls.COL_BATCH_SIZE])
            distributions.setdefault(label, {})[batch] = ITLMixtureModel.from_dict(row)

        if not distributions:
            raise ValueError(f"No ITL mixture rows loaded from {csv_path}")
        return cls(distributions, approx_sampling_thresh=approx_sampling_thresh)

    def save(self, csv_path: Path) -> None:
        """Save ITL mixture fits to a CSV.

        Args:
            csv_path: Output CSV path.
        """
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for label in sorted(self._distributions):
            for batch in sorted(self._distributions[label]):
                model = self._distributions[label][batch]
                rows.append(
                    {
                        self.COL_MODEL_LABEL: label,
                        self.COL_BATCH_SIZE: batch,
                        "itl_dist": "lognormal_mixture_2",
                        **{f"itl_mix_{k}": v for k, v in model.to_dict().items()},
                    }
                )
        pd.DataFrame(rows).to_csv(csv_path, index=False)


class InferenceTemplateStore:
    """Pre-built per-GPU power templates for a specific simulation config.

    Created by [`InferenceTraceStore.build_templates`][..InferenceTraceStore.build_templates].
    Use [`template`][.template] to look up a template by model label and batch size.
    """

    def __init__(
        self,
        templates: dict[tuple[str, int], np.ndarray],
        batch_sizes_by_model: dict[str, list[int]],
    ) -> None:
        self._templates = templates
        self._batch_sizes_by_model = batch_sizes_by_model

    def template(self, model_label: str, batch_size: int) -> np.ndarray:
        """Return a pre-built per-GPU power template."""
        key = (str(model_label), int(batch_size))
        if key not in self._templates:
            raise KeyError(f"No template for model={model_label!r}, batch={batch_size}.")
        return self._templates[key]

    def batch_sizes(self, model_label: str) -> list[int]:
        """List of batch sizes available for a model."""
        sizes = self._batch_sizes_by_model.get(model_label)
        if sizes is None:
            raise KeyError(f"Unknown model: {model_label!r}")
        return list(sizes)


class InferenceTraceStore:
    """Manages raw power traces loaded from CSV files.

    Indexed by `(model_label, batch_size)`. Provides:

    - [`load`][.load]: load traces discovered via a manifest CSV
    - [`build_templates`][.build_templates]: build per-GPU power
      templates for a specific simulation config, returning a
      [`InferenceTemplateStore`][..InferenceTemplateStore]
    """

    MANIFEST_COL_MODEL_LABEL = "model_label"
    MANIFEST_COL_NUM_GPUS = "num_gpus"
    MANIFEST_COL_BATCH_SIZE = "max_num_seqs"
    MANIFEST_COL_TRACE_FILE = "trace_file"
    TRACE_COL_TIME = "relative_time_s"
    TRACE_COL_POWER = "power_total_W"

    def __init__(self, traces: dict[str, dict[int, InferenceTrace]]) -> None:
        self._traces = {str(label): {int(b): tr for b, tr in per_batch.items()} for label, per_batch in traces.items()}

    @classmethod
    def load(cls, manifest: Path) -> InferenceTraceStore:
        """Load traces discovered via a manifest CSV.

        Trace file paths in the manifest are resolved relative to the
        manifest file's parent directory.

        Args:
            manifest: Path to the manifest CSV (e.g. `traces_summary.csv`).
                Expected columns: `model_label`, `num_gpus`, `max_num_seqs`,
                `trace_file`.
        """
        manifest = Path(manifest)
        base_dir = manifest.parent
        df = pd.read_csv(manifest)

        required_cols = [
            cls.MANIFEST_COL_MODEL_LABEL,
            cls.MANIFEST_COL_NUM_GPUS,
            cls.MANIFEST_COL_BATCH_SIZE,
            cls.MANIFEST_COL_TRACE_FILE,
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Manifest {manifest} missing columns: {missing}. Got: {list(df.columns)}")

        traces: dict[str, dict[int, InferenceTrace]] = {}
        for row in df.to_dict(orient="records"):
            label = str(row[cls.MANIFEST_COL_MODEL_LABEL])
            num_gpus = int(row[cls.MANIFEST_COL_NUM_GPUS])
            batch = int(row[cls.MANIFEST_COL_BATCH_SIZE])
            trace_path = base_dir / str(row[cls.MANIFEST_COL_TRACE_FILE])

            if not trace_path.exists():
                raise FileNotFoundError(f"Trace file not found: {trace_path} (model={label}, batch={batch})")

            tdf = pd.read_csv(trace_path)
            if cls.TRACE_COL_TIME not in tdf.columns or cls.TRACE_COL_POWER not in tdf.columns:
                raise ValueError(
                    f"{trace_path} must contain {cls.TRACE_COL_TIME!r} and "
                    f"{cls.TRACE_COL_POWER!r}. Got: {list(tdf.columns)}"
                )

            t = tdf[cls.TRACE_COL_TIME].to_numpy(float)
            p = tdf[cls.TRACE_COL_POWER].to_numpy(float)
            if np.any(np.diff(t) < 0):
                idx = np.argsort(t)
                t, p = t[idx], p[idx]

            traces.setdefault(label, {})[batch] = InferenceTrace(
                t_s=t,
                power_w=p,
                measured_gpus=num_gpus,
            )

        return cls(traces)

    def build_templates(
        self,
        *,
        duration_s: Fraction | float,
        dt_s: Fraction | float,
        steady_skip_s: float = 0.0,
    ) -> InferenceTemplateStore:
        """Build per-GPU power templates for all traces.

        Args:
            duration_s: Total simulation duration (seconds).
            dt_s: Simulation timestep (seconds).
            steady_skip_s: Skip this many seconds from the start of each
                trace to avoid warm-up transients.

        Returns:
            A [`InferenceTemplateStore`][openg2g.datacenter.workloads.inference.InferenceTemplateStore]
                holding the built templates.
        """
        templates: dict[tuple[str, int], np.ndarray] = {}
        batch_sizes_by_model: dict[str, list[int]] = {}
        for label, per_batch in self._traces.items():
            batch_sizes_by_model[label] = sorted(per_batch.keys())
            for batch, tr in per_batch.items():
                tpl = _build_per_gpu_power_template(
                    tr,
                    dt_s=dt_s,
                    duration_s=duration_s,
                    steady_skip_s=steady_skip_s,
                )
                templates[(label, batch)] = tpl
        return InferenceTemplateStore(templates, batch_sizes_by_model)

    def save(self, out_dir: Path) -> None:
        """Save traces and manifest CSV to a directory.

        Writes individual trace CSVs to `out_dir/traces/` and a manifest
        CSV at `out_dir/traces_summary.csv`.

        Args:
            out_dir: Output directory.
        """
        out_dir = Path(out_dir)
        traces_dir = out_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: list[dict[str, Any]] = []
        for label in sorted(self._traces):
            for batch in sorted(self._traces[label]):
                tr = self._traces[label][batch]
                trace_name = f"{label}_num_gpus_{tr.measured_gpus}_max_num_seqs_{batch}.csv"
                pd.DataFrame(
                    {
                        self.TRACE_COL_TIME: tr.t_s,
                        self.TRACE_COL_POWER: tr.power_w,
                    }
                ).to_csv(traces_dir / trace_name, index=False)
                summary_rows.append(
                    {
                        self.MANIFEST_COL_MODEL_LABEL: label,
                        self.MANIFEST_COL_NUM_GPUS: tr.measured_gpus,
                        self.MANIFEST_COL_BATCH_SIZE: batch,
                        self.MANIFEST_COL_TRACE_FILE: f"traces/{trace_name}",
                    }
                )
        pd.DataFrame(summary_rows).to_csv(out_dir / "traces_summary.csv", index=False)


class InferenceData:
    """LLM inference workload with offline simulation data.

    Bundles model specifications with power templates and latency
    distributions. Validates that all models have matching data entries.

    Args:
        models: Model specifications as a tuple of
            [`InferenceModelSpec`][openg2g.datacenter.config.InferenceModelSpec].
        power_templates: Pre-built per-GPU power templates for all models
            and batch sizes, created via
            [`InferenceTraceStore.build_templates`][..InferenceTraceStore.build_templates].
        itl_fits: Per-model ITL mixture distributions. Required when using
            controllers that read observed latency (e.g.,
            `OFOBatchSizeController`). When omitted, NaN is reported for
            observed latency.
    """

    def __init__(
        self,
        models: tuple[InferenceModelSpec, ...],
        *,
        power_templates: InferenceTemplateStore,
        itl_fits: ITLFitStore | None = None,
    ) -> None:
        if isinstance(power_templates, InferenceTraceStore):
            raise TypeError(
                "Expected a InferenceTemplateStore, got InferenceTraceStore. "
                "Call InferenceTraceStore.build_templates() first to create a InferenceTemplateStore."
            )
        if not models:
            raise ValueError("models must not be empty.")
        labels = [ms.model_label for ms in models]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate model labels: {labels}")

        self._models = models
        self._power_templates: InferenceTemplateStore | None = power_templates
        self._trace_store: InferenceTraceStore | None = None
        self._itl_fit_store: ITLFitStore | None = None
        self._itl_fits = itl_fits
        self._itl_samples_df: pd.DataFrame | None = None

        for ms in self._models:
            try:
                power_templates.batch_sizes(ms.model_label)
            except KeyError:
                raise ValueError(
                    f"Power templates missing for model {ms.model_label!r}. "
                    f"Ensure InferenceTraceStore contains traces for all models."
                ) from None

            if itl_fits is not None and ms.model_label not in itl_fits.distributions:
                raise ValueError(
                    f"ITL fits missing for model {ms.model_label!r}. "
                    f"Available models in ITLFitStore: {sorted(itl_fits.distributions.keys())}"
                )

    def filter_models(
        self,
        models: tuple[InferenceModelSpec, ...],
    ) -> InferenceData:
        """Return a new InferenceData containing only the specified models."""
        labels = {ms.model_label for ms in models}

        # Filter power templates
        filtered_templates = {k: v for k, v in self._power_templates._templates.items() if k[0] in labels}
        filtered_batch_sizes = {k: v for k, v in self._power_templates._batch_sizes_by_model.items() if k in labels}
        new_templates = InferenceTemplateStore(filtered_templates, filtered_batch_sizes)

        # Filter ITL fits
        new_itl_fits = None
        if self._itl_fits is not None:
            filtered_dists = {k: v for k, v in self._itl_fits.distributions.items() if k in labels}
            new_itl_fits = ITLFitStore(filtered_dists)

        return InferenceData(models, power_templates=new_templates, itl_fits=new_itl_fits)

    @classmethod
    def generate(
        cls,
        models: tuple[InferenceModelSpec, ...],
        data_sources: dict[str, MLEnergySource],
        *,
        runs: Any = None,
        mlenergy_data_dir: Path | None = None,
        dt_s: float = 0.1,
        seed: int = 0,
        itl_sample_cap: int = 2048,
    ) -> InferenceData:
        """Generate inference data from ML.ENERGY benchmark data.

        Produces power traces and ITL mixture fits for all models and
        batch sizes specified in `data_sources`.

        Args:
            models: Model specifications.
            data_sources: Per-model benchmark data extraction settings,
                keyed by `model_label`.
            runs: Pre-loaded `LLMRuns` object. If `None`, loads from
                `mlenergy_data_dir` or the HuggingFace Hub.
            mlenergy_data_dir: Path to compiled mlenergy-data directory.
                Ignored if `runs` is provided.
            dt_s: Trace timestep (seconds).
            seed: Random seed for ITL fitting.
            itl_sample_cap: Maximum ITL samples per run for fitting.

        Returns:
            A new `InferenceData` with generated traces and ITL fits (no
            templates -- call `InferenceTraceStore.build_templates()` on the
            saved/loaded store to get templates).
        """
        if runs is None:
            unique_tasks = {src.task for src in data_sources.values()}
            if mlenergy_data_dir:
                logger.info("Loading runs from %s (tasks: %s)", mlenergy_data_dir, sorted(unique_tasks))
                runs = LLMRuns.from_directory(str(mlenergy_data_dir), stable_only=False).task(*unique_tasks)
            else:
                logger.info("Loading runs from Hugging Face Hub (tasks: %s)", sorted(unique_tasks))
                runs = LLMRuns.from_hf(stable_only=False).task(*unique_tasks)
        if not runs:
            raise ValueError("No runs found for the specified tasks")

        subsets_by_label: dict[str, Any] = {}
        tl_frames: list[pd.DataFrame] = []
        itl_frames: list[pd.DataFrame] = []

        for ms in models:
            src = data_sources.get(ms.model_label)
            if src is None:
                raise ValueError(f"No data source for model {ms.model_label!r}")
            model_id = ms.model_id
            if not model_id:
                raise ValueError(f"model_id is required for data generation (model={ms.model_label!r})")

            subset = (
                runs.model_id(model_id).gpu_model(src.gpu).num_gpus(ms.gpus_per_replica).max_num_seqs(*src.batch_sizes)
            )
            if not subset:
                raise ValueError(
                    f"Config matched zero runs: model_id={model_id!r}, "
                    f"gpu={src.gpu!r}, num_gpus={ms.gpus_per_replica}, "
                    f"batch_sizes={src.batch_sizes}"
                )
            subsets_by_label[ms.model_label] = subset
            logger.info(
                "%s: %d runs (model_id=%s, gpu=%s, num_gpus=%d, batches=%s)",
                ms.model_label,
                len(subset),
                model_id,
                src.gpu,
                ms.gpus_per_replica,
                sorted({r.max_num_seqs for r in subset}),
            )

        logger.info("Downloading raw result files for %d models ...", len(subsets_by_label))
        for subset in subsets_by_label.values():
            subset.download_raw_files(file="results")
        logger.info("Downloads complete. Extracting timelines and ITL samples ...")

        for label, subset in subsets_by_label.items():
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

        all_tl = pd.concat(tl_frames, ignore_index=True)
        itl_samples_df = pd.concat(itl_frames, ignore_index=True)
        logger.info("Building trace store (%d timeline rows) and fitting ITL models ...", len(all_tl))

        trace_store = _build_trace_store_from_timelines(all_tl, dt_s=dt_s)
        itl_fit_store = _build_itl_fit_store(itl_samples_df, max_samples=itl_sample_cap, seed=seed)

        return cls._from_stores(
            models,
            trace_store=trace_store,
            itl_fit_store=itl_fit_store,
            itl_samples_df=itl_samples_df,
        )

    @classmethod
    def _from_stores(
        cls,
        models: tuple[InferenceModelSpec, ...],
        *,
        trace_store: InferenceTraceStore,
        itl_fit_store: ITLFitStore,
        itl_samples_df: pd.DataFrame | None = None,
    ) -> InferenceData:
        """Create from raw stores (internal, used by generate)."""
        instance = object.__new__(cls)
        instance._models = models
        instance._trace_store = trace_store
        instance._itl_fit_store = itl_fit_store
        instance._power_templates = None
        instance._itl_fits = itl_fit_store
        instance._itl_samples_df = itl_samples_df
        return instance

    def save(self, out_dir: Path, *, plot: bool = False) -> None:
        """Save traces and ITL fits to a directory.

        Args:
            out_dir: Output directory.
            plot: If `True`, also write characterization plots (power
                trajectories, ITL distributions).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self._trace_store is not None:
            self._trace_store.save(out_dir)
        if self._itl_fits is not None:
            self._itl_fits.save(out_dir / "latency_fits.csv")

        (out_dir / "_manifest.json").write_text(
            json.dumps({"openg2g_version": openg2g.__version__}, indent=2, sort_keys=True)
        )

        if plot and self._trace_store is not None:
            _plot_power_trajectories(self._trace_store, self._models, out_dir)
            itl_samples = self._itl_samples_df
            if self._itl_fit_store is not None and itl_samples is not None:
                for ms in self._models:
                    _plot_itl_distributions(self._itl_fit_store, itl_samples, ms.model_label, out_dir)

    @classmethod
    def load(
        cls,
        data_dir: Path,
        models: tuple[InferenceModelSpec, ...],
        *,
        duration_s: float = 600.0,
        dt_s: float = 0.1,
        steady_skip_s: float = 0.0,
    ) -> InferenceData:
        """Load from a generated data directory.

        Loads traces from `traces_summary.csv`, builds templates, and
        loads ITL fits from `latency_fits.csv`.

        Args:
            data_dir: Directory containing generated data.
            models: Model specifications.
            duration_s: Simulation duration for template building.
            dt_s: Simulation timestep for template building.
            steady_skip_s: Skip seconds for template building.
        """
        data_dir = Path(data_dir)
        _check_version_stamp(data_dir, "InferenceData")
        store = InferenceTraceStore.load(data_dir / "traces_summary.csv")
        templates = store.build_templates(duration_s=duration_s, dt_s=dt_s, steady_skip_s=steady_skip_s)
        itl_fits = ITLFitStore.load(data_dir / "latency_fits.csv")
        return cls(models, power_templates=templates, itl_fits=itl_fits)

    @classmethod
    def ensure(
        cls,
        data_dir: Path,
        models: tuple[InferenceModelSpec, ...],
        data_sources: dict[str, MLEnergySource] | None = None,
        *,
        mlenergy_data_dir: Path | None = None,
        plot: bool = False,
        duration_s: float = 600.0,
        dt_s: float = 0.1,
        steady_skip_s: float = 0.0,
    ) -> InferenceData:
        """Load from `data_dir`, generating first if needed.

        If `data_dir/traces_summary.csv` does not exist, generates
        inference data from ML.ENERGY benchmark data and saves it.
        Then loads and returns.

        Args:
            data_dir: Data directory (generated files go here).
            models: Model specifications.
            data_sources: Per-model benchmark data extraction settings,
                keyed by `model_label`. Required when no cached data exists.
            mlenergy_data_dir: Path to compiled mlenergy-data directory.
            plot: If `True`, generate characterization plots on generation.
            duration_s: Simulation duration for template building.
            dt_s: Simulation timestep for template building.
            steady_skip_s: Skip seconds for template building.
        """
        data_dir = Path(data_dir)
        if not (data_dir / "traces_summary.csv").exists():
            if data_sources is None:
                raise ValueError("data_sources required for InferenceData generation (no cached data)")
            logger.info("Generating inference data to %s ...", data_dir)
            cls.generate(
                models,
                data_sources,
                mlenergy_data_dir=mlenergy_data_dir,
                dt_s=dt_s,
            ).save(data_dir, plot=plot)
        return cls.load(data_dir, models, duration_s=duration_s, dt_s=dt_s, steady_skip_s=steady_skip_s)

    @property
    def models(self) -> tuple[InferenceModelSpec, ...]:
        """The model specifications."""
        return self._models

    @property
    def power_templates(self) -> InferenceTemplateStore:
        if self._power_templates is None:
            raise RuntimeError("power_templates not available (generate-only instance). Load from disk first.")
        return self._power_templates

    @property
    def itl_fits(self) -> ITLFitStore | None:
        return self._itl_fits


def _check_version_stamp(data_dir: Path, label: str) -> None:
    """Log a warning if cached data was generated with a different openg2g version."""
    manifest_path = data_dir / "_manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    cached_version = manifest.get("openg2g_version", "unknown")
    if cached_version != openg2g.__version__:
        logger.warning(
            "%s: cached data generated with openg2g %s (current %s). Consider regenerating.",
            label,
            cached_version,
            openg2g.__version__,
        )


def _build_trace_store_from_timelines(tl: pd.DataFrame, *, dt_s: float) -> InferenceTraceStore:
    """Build an InferenceTraceStore from raw timeline data.

    Args:
        tl: Combined timeline dataframe with columns `model_label`,
            `num_gpus`, `max_num_seqs`, `run_index`, `relative_time_s`, `value`.
        dt_s: Resampling timestep.

    Returns:
        An InferenceTraceStore with median-aggregated traces.
    """
    if tl.empty:
        raise ValueError("No timeline rows extracted from selected runs")

    traces: dict[str, dict[int, InferenceTrace]] = {}
    keys = [
        InferenceTraceStore.MANIFEST_COL_MODEL_LABEL,
        InferenceTraceStore.MANIFEST_COL_NUM_GPUS,
        InferenceTraceStore.MANIFEST_COL_BATCH_SIZE,
    ]
    for key, g in tl.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            raise TypeError(f"Expected tuple groupby key, got {type(key).__name__}")
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

        traces.setdefault(model_label, {})[batch] = InferenceTrace(
            t_s=grid,
            power_w=p_med,
            measured_gpus=int(num_gpus),
        )

    if not traces:
        raise ValueError("No trace profiles extracted from timeline data")
    return InferenceTraceStore(traces)


def _build_itl_fit_store(
    itl: pd.DataFrame,
    *,
    max_samples: int,
    seed: int,
) -> ITLFitStore:
    """Build an ITLFitStore from raw ITL sample data.

    Args:
        itl: ITL sample dataframe with columns `model_label`, `num_gpus`,
            `max_num_seqs`, `itl_s`.
        max_samples: Maximum ITL samples per group for fitting.
        seed: Random seed for ITL fitting.

    Returns:
        An ITLFitStore with fitted mixture distributions.
    """
    if itl.empty:
        raise ValueError("No ITL samples provided")

    distributions: dict[str, dict[int, ITLMixtureModel]] = {}
    for key, g in itl.groupby(["model_label", "num_gpus", "max_num_seqs"], dropna=False):
        if not isinstance(key, tuple):
            raise TypeError(f"Expected tuple groupby key, got {type(key).__name__}")
        model_label, _num_gpus, batch = str(key[0]), cast(int, key[1]), cast(int, key[2])
        fit = ITLMixtureModel.fit(
            g["itl_s"].to_numpy(dtype=float),
            max_samples=max_samples,
            seed=seed,
        )
        distributions.setdefault(model_label, {})[batch] = fit

    if not distributions:
        raise ValueError("No ITL fits produced")
    return ITLFitStore(distributions)


@dataclass(frozen=True)
class InferenceAugmentedPower:
    """Result of inference power augmentation for one simulation timestep.

    Attributes:
        power_w: Three-phase inference power (watts), excluding base load.
        power_by_model_w: Per-model total active power (watts).
        active_replicas_by_model: Per-model active replica count.
    """

    power_w: ThreePhase
    power_by_model_w: dict[str, float] = field(default_factory=dict)
    active_replicas_by_model: dict[str, int] = field(default_factory=dict)


class InferencePowerAugmenter:
    """Scales per-GPU inference power through server layouts to three-phase power.

    Given per-GPU power values for each server (one value per server per
    model), applies per-server scaling, noise, activation masking, and
    phase summation to produce inference-level three-phase power.

    This class is backend-agnostic. The offline datacenter feeds it
    template-indexed values; the online datacenter can feed it
    live-measured values. The datacenter backend is responsible for
    adding facility base load on top of the returned inference power.

    Args:
        layouts: Per-model server layouts (physical topology + activation priority).
        seed: Random seed for noise RNG.
    """

    def __init__(
        self,
        layouts: dict[str, ServerLayout],
        seed: int = 0,
    ) -> None:
        self._layouts = layouts
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    def augment(
        self,
        per_gpu_by_model: dict[str, np.ndarray],
        replica_counts: dict[str, int],
    ) -> InferenceAugmentedPower:
        """Augment per-server per-GPU power to three-phase power.

        Args:
            per_gpu_by_model: Mapping of model label to per-GPU power
                array of shape `(num_servers,)`. Only models with active
                replicas should be included.
            replica_counts: Mapping of model label to effective replica
                count (schedule + runtime adjustments).

        Returns:
            Augmented inference power with three-phase totals, per-model
                power, and per-model active replica counts.
        """
        phase_power = np.zeros(3, dtype=float)
        power_by_model: dict[str, float] = {}
        active_replicas_by_model: dict[str, int] = {}

        for label, per_gpu in per_gpu_by_model.items():
            layout = self._layouts[label]

            server_powers = per_gpu * layout.gpus_per_server_list * layout.amplitude_scales
            if layout.noise_fraction > 0:
                levels = np.maximum(server_powers, 1.0)
                server_powers += self._rng.normal(0.0, 1.0, size=layout.num_servers) * layout.noise_fraction * levels
            server_powers = np.maximum(server_powers, 0.0)

            active_indices = layout.active_indices(replica_counts.get(label, 0))
            active_powers = server_powers[active_indices]
            active_phases = layout.phase_list[active_indices]

            model_phase_power = np.zeros(3, dtype=float)
            np.add.at(model_phase_power, active_phases, active_powers)
            phase_power += model_phase_power

            power_by_model[label] = float(np.sum(active_powers))
            active_gpus = int(np.sum(layout.gpus_per_server_list[active_indices]))
            active_replicas_by_model[label] = active_gpus // layout.gpus_per_replica

        return InferenceAugmentedPower(
            power_w=ThreePhase(
                a=float(phase_power[0]),
                b=float(phase_power[1]),
                c=float(phase_power[2]),
            ),
            power_by_model_w=power_by_model,
            active_replicas_by_model=active_replicas_by_model,
        )

    def reset(self) -> None:
        """Re-seed the noise RNG to its initial state."""
        self._rng = np.random.default_rng(self._seed)


def _lognorm_pdf(x: np.ndarray, sigma: float, scale: float) -> np.ndarray:
    """Standard lognormal PDF: f(x; sigma, scale) for x > 0."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask = x > 0
    xx = x[mask]
    out[mask] = (1.0 / (xx * sigma * np.sqrt(2.0 * np.pi))) * np.exp(-(np.log(xx / scale) ** 2) / (2.0 * sigma * sigma))
    return out


def _plot_power_trajectories(
    trace_store: InferenceTraceStore,
    models: tuple[InferenceModelSpec, ...],
    out_dir: Path,
    *,
    rolling_window: int = 10,
) -> None:
    """Plot total GPU power trajectories per batch size.

    One subplot per model. Each curve is a different batch size.
    Saves to `out_dir / "power_trajectories.png"`.
    """
    import matplotlib.pyplot as plt

    model_labels = [m.model_label for m in models]
    n_models = len(model_labels)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 5), dpi=160, squeeze=False)

    panel_labels = "abcdefghij"

    for row, model_label in enumerate(model_labels):
        ax = axes[row, 0]
        per_batch = trace_store._traces.get(model_label, {})
        if not per_batch:
            ax.set_title(f"{model_label} (no traces found)")
            continue

        batches = sorted(per_batch.keys())
        cmap = plt.get_cmap("tab10")

        for i, batch in enumerate(batches):
            tr = per_batch[batch]
            time_s = tr.t_s.copy()
            power_kw = tr.power_w.copy() / 1000.0

            if rolling_window > 1 and len(power_kw) >= rolling_window:
                kernel = np.ones(rolling_window) / rolling_window
                smoothed = np.convolve(power_kw, kernel, mode="same")
                half = rolling_window // 2
                smoothed[:half] = power_kw[:half]
                smoothed[-half:] = power_kw[-half:]
                power_kw = smoothed

            ax.plot(time_s, power_kw, label=f"batch={batch}", color=cmap(i))

        label_char = panel_labels[row] if row < len(panel_labels) else ""
        num_gpus = per_batch[batches[0]].measured_gpus
        gpu_suffix = "GPUs" if num_gpus > 1 else "GPU"
        ax.set_title(
            f"({label_char}) {model_label}: Total-GPU Power ({num_gpus} {gpu_suffix})",
            fontsize=13,
        )
        ax.set_ylabel("Power (kW)", fontsize=11)
        if row == 0:
            ax.legend(fontsize=9, ncol=len(batches), loc="lower center", frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    axes[-1, 0].set_xlabel("Time (seconds)", fontsize=11)
    fig.tight_layout()

    save_path = out_dir / "power_trajectories.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    logger.info("Saved power trajectories plot to %s", save_path)


def _plot_itl_distributions(
    itl_fit_store: ITLFitStore,
    itl_samples_df: pd.DataFrame,
    model_label: str,
    out_dir: Path,
    *,
    hist_bins: int = 120,
    hist_alpha: float = 0.12,
    x_lo_q: float = 0.5,
    x_hi_q: float = 99.5,
    grid_n: int = 1200,
) -> None:
    """Plot ITL mixture distribution overlay for one model.

    Shows the fitted mixture PDF for each batch size overlaid, with
    histograms and an inset showing steady/stall component decomposition
    for the largest batch size. Saves to `out_dir / "itl_distributions_{model_label}.png"`.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    model_dists = itl_fit_store.distributions.get(model_label, {})
    if not model_dists:
        logger.warning("No ITL distributions for model %s, skipping plot", model_label)
        return

    samples = itl_samples_df[itl_samples_df["model_label"] == model_label]
    batches = sorted(model_dists.keys())

    all_x = samples[samples["max_num_seqs"].isin(batches)]["itl_s"].to_numpy(dtype=float)
    if len(all_x) == 0:
        logger.warning("No ITL samples for model %s, skipping plot", model_label)
        return

    lo = float(np.percentile(all_x, x_lo_q))
    hi = float(np.percentile(all_x, x_hi_q))
    grid = np.linspace(lo, hi, grid_n)

    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=160)

    cmap = plt.get_cmap("tab10") if len(batches) <= 10 else plt.get_cmap("tab20")
    colors = {b: cmap(i % cmap.N) for i, b in enumerate(batches)}

    for b in batches:
        model = model_dists[b]
        params = model.to_dict()
        loc = float(params["loc"])
        pi = float(params["pi_steady"])
        s1 = float(params["sigma_steady"])
        sc1 = float(params["scale_steady"])
        s2 = float(params["sigma_stall"])
        sc2 = float(params["scale_stall"])

        shifted = grid - loc
        pdf_mix = pi * _lognorm_pdf(shifted, s1, sc1) + (1 - pi) * _lognorm_pdf(shifted, s2, sc2)

        c = colors[b]
        bsamp = samples[samples["max_num_seqs"] == b]["itl_s"].to_numpy(dtype=float)
        if len(bsamp) > 0:
            ax.hist(bsamp, bins=hist_bins, range=(lo, hi), density=True, alpha=hist_alpha, color=c)
        ax.plot(grid, pdf_mix, linewidth=2.2, color=c, label=f"batch={b}")

    ax.set_title(f"(a) {model_label}: ITL distribution vs batch size")
    ax.set_xlabel("Inter-token latency (seconds)")
    ax.set_ylabel("Density")
    ax.legend(ncol=4, fontsize=9, frameon=True)
    ax.set_xlim(lo, hi)

    inset_batch = max(batches)
    inset_model = model_dists[inset_batch]
    inset_params = inset_model.to_dict()
    loc = float(inset_params["loc"])
    pi = float(inset_params["pi_steady"])
    s1 = float(inset_params["sigma_steady"])
    sc1 = float(inset_params["scale_steady"])
    s2 = float(inset_params["sigma_stall"])
    sc2 = float(inset_params["scale_stall"])

    bsamp = samples[samples["max_num_seqs"] == inset_batch]["itl_s"].to_numpy(dtype=float)
    lo_i = float(np.percentile(bsamp, 0.5)) if len(bsamp) > 0 else lo
    hi_i = float(np.percentile(bsamp, 99.5)) if len(bsamp) > 0 else hi
    grid_i = np.linspace(lo_i, hi_i, 600)

    shifted_i = grid_i - loc
    pdf_steady = pi * _lognorm_pdf(shifted_i, s1, sc1)
    pdf_stall = (1 - pi) * _lognorm_pdf(shifted_i, s2, sc2)
    pdf_mix_i = pdf_steady + pdf_stall

    axins = inset_axes(
        ax,
        width="38%",
        height="55%",
        loc="lower right",
        bbox_to_anchor=(-0.1, 0.1, 1, 1),
        bbox_transform=ax.transAxes,
    )

    if len(bsamp) > 0:
        axins.hist(bsamp, bins=60, range=(lo_i, hi_i), density=True, alpha=0.20, color=colors[inset_batch])

    axins.plot(grid_i, pdf_mix_i, lw=2.0, color=colors[inset_batch], label="mixture")
    axins.plot(grid_i, pdf_steady, lw=1.6, ls="--", color="0.25", label="steady")
    axins.plot(grid_i, pdf_stall, lw=1.6, ls=":", color="0.25", label="stall")

    axins.set_title(f"(b) batch={inset_batch} components", fontsize=9)
    axins.set_xlim(lo_i, hi_i)
    axins.tick_params(axis="both", labelsize=8)
    axins.grid(True, alpha=0.25)
    axins.legend(fontsize=8, frameon=True, loc="upper right")

    fig.tight_layout()

    save_path = out_dir / f"itl_distributions_{model_label}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    logger.info("Saved ITL distributions plot for %s to %s", model_label, save_path)


class RequestsConfig(BaseModel):
    """Configuration for building per-model JSONL request files.

    Attributes:
        dataset: Dataset to sample prompts from (`"gpqa"` or `"lm-arena-chat"`).
        num_requests: Number of requests to sample per model.
        max_completion_tokens: Maximum output tokens per request.
        seed: Random seed for dataset shuffling and oversampling.
        system_prompt: System prompt prepended to every request.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str = "lm-arena-chat"
    num_requests: int = 1000
    max_completion_tokens: int = 512
    seed: int = 0
    system_prompt: str = "You are a helpful AI assistant."


class RequestStore:
    """Per-model request dicts for online load generation.

    Each model's requests are stored as a list of OpenAI Chat Completion
    streaming request dicts, suitable for submission to a vLLM server.

    Attributes:
        requests_by_model: Mapping from model label to request dicts.
    """

    def __init__(self, requests_by_model: dict[str, list[dict]]) -> None:
        self.requests_by_model = requests_by_model

    @classmethod
    def generate(
        cls,
        models: Sequence[InferenceModelSpec],
        config: RequestsConfig | None = None,
        *,
        extra_body_by_model: dict[str, dict] | None = None,
    ) -> RequestStore:
        """Sample prompts and build per-model request dicts.

        Requires `pip install datasets openai`.

        Args:
            models: Model specifications. Uses `model_id` for the API
                model field.
            config: Request generation config. Uses defaults if `None`.
            extra_body_by_model: Optional per-model extra fields merged
                into every request dict (e.g. `chat_template_kwargs`).
                Keyed by `model_label`.
        """
        import random as _random

        from datasets import load_dataset
        from openai.types.chat import (
            ChatCompletionAssistantMessageParam,
            ChatCompletionContentPartTextParam,
            ChatCompletionMessageParam,
            ChatCompletionSystemMessageParam,
            ChatCompletionUserMessageParam,
        )
        from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming

        if config is None:
            config = RequestsConfig()

        def _text_part(text: str) -> ChatCompletionContentPartTextParam:
            return ChatCompletionContentPartTextParam(type="text", text=text)

        def _prompt_to_messages(prompt: str | list[str]) -> list[ChatCompletionMessageParam]:
            if isinstance(prompt, str):
                return [ChatCompletionUserMessageParam(role="user", content=[_text_part(prompt)])]
            msgs: list[ChatCompletionMessageParam] = [
                ChatCompletionUserMessageParam(role="user", content=[_text_part(prompt[0])])
            ]
            for i, turn in enumerate(prompt[1:]):
                if i % 2 == 0:
                    msgs.append(ChatCompletionAssistantMessageParam(role="assistant", content=[_text_part(turn)]))
                else:
                    msgs.append(ChatCompletionUserMessageParam(role="user", content=[_text_part(turn)]))
            return msgs

        def _maybe_oversample(items: list, target: int, seed: int) -> None:
            if len(items) >= target:
                return
            rng = _random.Random(seed)
            original = list(items)
            while len(items) < target:
                items.append(rng.choice(original))

        def _sample_lm_arena_chat(num_requests: int, seed: int) -> list[str | list[str]]:
            data = load_dataset("lmarena-ai/arena-human-preference-100k", split="train").shuffle(seed=seed)
            prompts: list[str | list[str]] = []
            for item in data:
                num_turns = item["turn"]
                conversation = item["conversation_a"]
                for turns in range(num_turns):
                    if len(prompts) >= num_requests:
                        break
                    messages: list[str] = []
                    for message in conversation[: 2 * turns + 1]:
                        messages.append(message["content"])
                    prompts.append(messages if len(messages) > 1 else messages[0])
                if len(prompts) >= num_requests:
                    break
            _maybe_oversample(prompts, num_requests, seed)
            return prompts

        def _sample_gpqa(num_requests: int, seed: int) -> list[str | list[str]]:
            data = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train", streaming=True).shuffle(seed=seed)
            _random.seed(seed)
            prompts: list[str | list[str]] = []
            for item in data:
                if len(prompts) >= num_requests:
                    break
                choices = [
                    item["Incorrect Answer 1"].strip(),
                    item["Incorrect Answer 2"].strip(),
                    item["Incorrect Answer 3"].strip(),
                    item["Correct Answer"].strip(),
                ]
                _random.shuffle(choices)
                question = item["Question"]
                prompt = f"What is the correct answer to the following question: {question}\n\nChoices:"
                for letter, choice in zip("ABCD", choices, strict=True):
                    prompt += f"\n({letter}) {choice}"
                prompts.append(prompt)
            _maybe_oversample(prompts, num_requests, seed)
            return prompts

        samplers = {"lm-arena-chat": _sample_lm_arena_chat, "gpqa": _sample_gpqa}
        sampler = samplers.get(config.dataset)
        if sampler is None:
            raise ValueError(f"Unknown dataset: {config.dataset!r}. Available: {sorted(samplers)}")

        extra = extra_body_by_model or {}

        requests_by_model: dict[str, list[dict]] = {}
        for spec in models:
            label = spec.model_label
            model_id = spec.model_id

            logger.info("Sampling %d %s prompts for %s (%s)...", config.num_requests, config.dataset, label, model_id)
            prompts = sampler(num_requests=config.num_requests, seed=config.seed)

            system_msgs: list[ChatCompletionMessageParam] = []
            if config.system_prompt:
                system_msgs.append(ChatCompletionSystemMessageParam(role="system", content=config.system_prompt))

            template = CompletionCreateParamsStreaming(
                model=model_id,
                messages=system_msgs,
                max_completion_tokens=config.max_completion_tokens,
                stream=True,
                stream_options={"include_usage": True, "continuous_usage_stats": True},
            )
            if label in extra:
                template.update(extra[label])

            reqs: list[dict] = []
            for prompt in prompts:
                request = dict(template)
                request["messages"] = list(template["messages"]) + _prompt_to_messages(prompt)
                reqs.append(request)
            requests_by_model[label] = reqs

        return cls(requests_by_model)

    def save(self, out_dir: Path) -> None:
        """Write per-model JSONL files to `out_dir`.

        Args:
            out_dir: Output directory. Created if it doesn't exist.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for label, reqs in self.requests_by_model.items():
            out_path = out_dir / f"{label}.jsonl"
            with open(out_path, "w") as f:
                for req in reqs:
                    f.write(json.dumps(req) + "\n")
            logger.info("Wrote %d requests for %s to %s", len(reqs), label, out_path)

    @classmethod
    def load(cls, out_dir: Path) -> RequestStore:
        """Load per-model JSONL files from `out_dir`.

        Args:
            out_dir: Directory containing `{model_label}.jsonl` files.
        """
        requests_by_model: dict[str, list[dict]] = {}
        out_dir = Path(out_dir)
        for path in sorted(out_dir.glob("*.jsonl")):
            label = path.stem
            reqs: list[dict] = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        reqs.append(json.loads(line))
            requests_by_model[label] = reqs
            logger.info("Loaded %d requests for %s", len(reqs), label)
        return cls(requests_by_model)

    @classmethod
    def ensure(
        cls,
        out_dir: Path,
        models: Sequence[InferenceModelSpec] | None = None,
        config: RequestsConfig | None = None,
        *,
        extra_body_by_model: dict[str, dict] | None = None,
    ) -> RequestStore:
        """Load request files from `out_dir`, generating first if needed.

        Args:
            out_dir: Directory for JSONL files.
            models: Required if request files don't exist yet.
            config: Request generation config. Uses defaults if `None`.
            extra_body_by_model: Optional per-model extra fields for
                request generation. Keyed by `model_label`.
        """
        out_dir = Path(out_dir)
        if not out_dir.exists():
            if models is None:
                raise ValueError("models required (no cached request data)")
            logger.info("Generating request files to %s ...", out_dir)
            cls.generate(models, config, extra_body_by_model=extra_body_by_model).save(out_dir)
        return cls.load(out_dir)
