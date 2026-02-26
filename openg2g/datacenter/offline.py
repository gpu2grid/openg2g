"""Offline (trace-based) datacenter backend.

Loads power trace CSVs and serves per-timestep OfflineDatacenterState objects via step().
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from mlenergy_data.modeling import ITLMixtureModel

from openg2g.clock import SimulationClock
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter, LLMDatacenterState
from openg2g.datacenter.config import (
    DatacenterConfig,
    PowerAugmentationConfig,
    ServerRamp,
    ServerRampSchedule,
    TrainingRun,
    TrainingSchedule,
)
from openg2g.datacenter.layout import (
    ActivationStrategy,
    PowerAugmenter,
    RampActivationStrategy,
    ServerLayout,
)
from openg2g.datacenter.training_overlay import TrainingOverlayCache
from openg2g.events import EventEmitter
from openg2g.models.spec import LLMInferenceModelSpec, LLMInferenceWorkload
from openg2g.types import (
    DatacenterCommand,
    SetBatchSize,
    ThreePhase,
)

logger = logging.getLogger(__name__)

# Re-exports for backward compatibility (old names -> new names)


@dataclass(frozen=True)
class OfflineDatacenterState(LLMDatacenterState):
    """Extended state from the offline (trace-based) backend.

    Adds per-model power breakdown to
    [`LLMDatacenterState`][openg2g.datacenter.base.LLMDatacenterState].
    """

    power_by_model_w: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PowerTrace:
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
    trace: PowerTrace,
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


class PowerTemplateStore:
    """Pre-built per-GPU power templates for a specific simulation config.

    Created by [`PowerTraceStore.build_templates`][openg2g.datacenter.offline.PowerTraceStore.build_templates].
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


class PowerTraceStore:
    """Manages raw power traces loaded from CSV files.

    Indexed by `(model_label, batch_size)`. Provides:

    - [`load`][.load]: load traces discovered via a manifest CSV
    - [`from_traces`][.from_traces]: construct from pre-built
      `PowerTrace` objects
    - [`build_templates`][.build_templates]: build per-GPU power
      templates for a specific simulation config, returning a
      [`PowerTemplateStore`][openg2g.datacenter.offline.PowerTemplateStore]
    - [`trace`][.trace]: access the raw trace

    Attributes:
        MANIFEST_COL_MODEL_LABEL: Column name for model label in the manifest.
        MANIFEST_COL_NUM_GPUS: Column name for measured GPU count in the manifest.
        MANIFEST_COL_BATCH_SIZE: Column name for batch size in the manifest.
        MANIFEST_COL_TRACE_FILE: Column name for trace file path in the manifest.
        TRACE_COL_TIME: Column name for time in trace CSVs.
        TRACE_COL_POWER: Column name for power in trace CSVs.
    """

    MANIFEST_COL_MODEL_LABEL = "model_label"
    MANIFEST_COL_NUM_GPUS = "num_gpus"
    MANIFEST_COL_BATCH_SIZE = "max_num_seqs"
    MANIFEST_COL_TRACE_FILE = "trace_file"
    TRACE_COL_TIME = "relative_time_s"
    TRACE_COL_POWER = "power_total_W"

    def __init__(self, traces: dict[str, dict[int, PowerTrace]]) -> None:
        self._traces = {str(label): {int(b): tr for b, tr in per_batch.items()} for label, per_batch in traces.items()}

    @classmethod
    def load(cls, manifest: Path) -> PowerTraceStore:
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

        traces: dict[str, dict[int, PowerTrace]] = {}
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

            traces.setdefault(label, {})[batch] = PowerTrace(
                t_s=t,
                power_w=p,
                measured_gpus=num_gpus,
            )

        return cls(traces)

    @classmethod
    def from_traces(cls, traces: dict[str, dict[int, PowerTrace]]) -> PowerTraceStore:
        """Construct from pre-built [`PowerTrace`][...PowerTrace] objects.

        Args:
            traces: Mapping of `model_label -> batch_size -> PowerTrace`.
        """
        return cls(traces)

    def trace(self, model_label: str, batch_size: int) -> PowerTrace:
        """Return the raw power trace for a model and batch size."""
        per_batch = self._traces.get(model_label)
        if per_batch is None:
            raise KeyError(f"Unknown model: {model_label!r}")
        tr = per_batch.get(int(batch_size))
        if tr is None:
            raise KeyError(
                f"No trace for model={model_label!r}, batch={batch_size}. "
                f"Available batch sizes: {sorted(per_batch.keys())}"
            )
        return tr

    def build_templates(
        self,
        *,
        duration_s: Fraction | float,
        dt_s: Fraction | float,
        steady_skip_s: float = 0.0,
    ) -> PowerTemplateStore:
        """Build per-GPU power templates for all traces.

        Args:
            duration_s: Total simulation duration (seconds).
            dt_s: Simulation timestep (seconds).
            steady_skip_s: Skip this many seconds from the start of each
                trace to avoid warm-up transients.

        Returns:
            A [`PowerTemplateStore`][openg2g.datacenter.offline.PowerTemplateStore]
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
        return PowerTemplateStore(templates, batch_sizes_by_model)

    @property
    def model_labels(self) -> list[str]:
        """List of model labels in the store."""
        return list(self._traces.keys())


class OfflineInferenceData:
    """LLM inference workload with offline simulation data.

    Bundles model specifications with power templates and latency
    distributions. Validates that all models have matching data entries.

    Args:
        models: Model specifications. Accepts a tuple of
            [`LLMInferenceModelSpec`][openg2g.models.spec.LLMInferenceModelSpec]
            or a pre-built
            [`LLMInferenceWorkload`][openg2g.models.spec.LLMInferenceWorkload].
        power_templates: Pre-built per-GPU power templates for all models
            and batch sizes, created via
            [`PowerTraceStore.build_templates`][..PowerTraceStore.build_templates].
        itl_fits: Per-model ITL mixture distributions. Required when using
            controllers that read observed latency (e.g.,
            `OFOBatchSizeController`). When omitted, NaN is reported for
            observed latency.
    """

    def __init__(
        self,
        models: tuple[LLMInferenceModelSpec, ...] | LLMInferenceWorkload,
        *,
        power_templates: PowerTemplateStore,
        itl_fits: ITLFitStore | None = None,
    ) -> None:
        if isinstance(power_templates, PowerTraceStore):
            raise TypeError(
                "Expected a PowerTemplateStore, got PowerTraceStore. "
                "Call PowerTraceStore.build_templates() first to create a PowerTemplateStore."
            )
        if isinstance(models, LLMInferenceWorkload):
            self._workload = models
        else:
            self._workload = LLMInferenceWorkload(models=models)

        self._power_templates = power_templates
        self._itl_fits = itl_fits

        for ms in self._workload.models:
            try:
                power_templates.batch_sizes(ms.model_label)
            except KeyError:
                raise ValueError(
                    f"Power templates missing for model {ms.model_label!r}. "
                    f"Ensure PowerTraceStore contains traces for all models."
                ) from None

            if itl_fits is not None and ms.model_label not in itl_fits.distributions:
                raise ValueError(
                    f"ITL fits missing for model {ms.model_label!r}. "
                    f"Available models in ITLFitStore: {sorted(itl_fits.distributions.keys())}"
                )

    @property
    def workload(self) -> LLMInferenceWorkload:
        """The underlying inference workload specification."""
        return self._workload

    @property
    def power_templates(self) -> PowerTemplateStore:
        return self._power_templates

    @property
    def itl_fits(self) -> ITLFitStore | None:
        return self._itl_fits


class OfflineWorkload:
    """Complete offline simulation workload.

    Bundles inference data with optional training overlays and server ramp
    events. Normalizes flexible input types internally.

    Args:
        inference_data: LLM inference workload with offline simulation
            data (model specs, power templates, ITL fits).
        training: Training workload window(s). `None` disables training
            overlay.
        server_ramps: Server ramp event(s). `None` keeps all servers
            active.
    """

    def __init__(
        self,
        inference_data: OfflineInferenceData,
        training: TrainingRun | TrainingSchedule | None = None,
        server_ramps: ServerRamp | ServerRampSchedule | None = None,
    ) -> None:
        self._inference_data = inference_data

        if training is None:
            self._training = TrainingSchedule(entries=())
        elif isinstance(training, TrainingRun):
            self._training = TrainingSchedule(entries=(training,))
        else:
            self._training = training

        if server_ramps is None:
            self._server_ramps = ServerRampSchedule(entries=())
        elif isinstance(server_ramps, ServerRamp):
            self._server_ramps = ServerRampSchedule(entries=(server_ramps,))
        else:
            self._server_ramps = server_ramps

    @property
    def inference_data(self) -> OfflineInferenceData:
        return self._inference_data

    @property
    def training(self) -> TrainingSchedule:
        return self._training

    @property
    def server_ramps(self) -> ServerRampSchedule:
        return self._server_ramps


class OfflineDatacenter(LLMBatchSizeControlledDatacenter[OfflineDatacenterState]):
    """Trace-based datacenter simulation with step-by-step interface.

    Each `step` call computes one timestep of power output by indexing
    into pre-built per-GPU templates, applying per-server amplitude
    scaling and noise, and summing across active servers per phase.

    Batch size changes via `apply_control` take effect on the next
    `step` call.

    If `workload.server_ramps` is set, it is automatically wrapped in a
    [`RampActivationStrategy`][openg2g.datacenter.layout.RampActivationStrategy].

    Args:
        datacenter: Facility configuration (GPUs per server, base load).
        workload: Offline workload configuration bundling inference data,
            training overlays, and server ramp events.
        dt_s: Simulation timestep (seconds).
        seed: Random seed for layout generation, noise, and latency
            sampling. Sub-seeds are derived deterministically.
        power_augmentation: Per-server amplitude scaling and noise
            settings.
    """

    def __init__(
        self,
        datacenter: DatacenterConfig,
        workload: OfflineWorkload,
        *,
        dt_s: Fraction,
        seed: int = 0,
        power_augmentation: PowerAugmentationConfig | None = None,
    ) -> None:
        if power_augmentation is None:
            power_augmentation = PowerAugmentationConfig()
        inference = workload.inference_data
        models = list(inference.workload.models)

        activation_strategy: ActivationStrategy | None = None
        if workload.server_ramps:
            activation_strategy = RampActivationStrategy(workload.server_ramps)

        training_overlays: list[tuple[TrainingOverlayCache, TrainingRun]] = []
        for tr in workload.training:
            overlay = TrainingOverlayCache(
                tr.trace,
                target_peak_W_per_gpu=tr.target_peak_W_per_gpu,
            )
            training_overlays.append((overlay, tr))

        self._dt_s = dt_s
        self._template_store = inference.power_templates
        self._models = models
        self._gpus_per_server = int(datacenter.gpus_per_server)
        self._seed = int(seed)
        self._amplitude_scale_range = (
            float(power_augmentation.amplitude_scale_range[0]),
            float(power_augmentation.amplitude_scale_range[1]),
        )
        self._noise_fraction = float(power_augmentation.noise_fraction)

        self._layout_rng = np.random.default_rng(self._seed)

        self._activation_strategy = activation_strategy
        self._base_W_per_phase = float(datacenter.base_kw_per_phase) * 1e3

        self._training_overlays = training_overlays
        self._itl_fits = inference.itl_fits

        self._batch_by_model: dict[str, int] = {ms.model_label: ms.initial_batch_size for ms in models}

        self._layouts: dict[str, ServerLayout] = {}
        self._build_all_layouts()
        self._augmenter = PowerAugmenter(
            layouts=self._layouts,
            base_w_per_phase=self._base_W_per_phase,
            seed=self._seed + 12345,
        )

        self._global_step: int = 0
        self._latency_rng = np.random.default_rng(self._seed + 54321)
        self._events: EventEmitter | None = None
        self._state: OfflineDatacenterState | None = None
        self._history: list[OfflineDatacenterState] = []

        logger.info(
            "OfflineDatacenter: %d models, dt=%s s, seed=%d",
            len(models),
            dt_s,
            seed,
        )
        for ms in models:
            logger.info(
                "  %s: %d replicas, %d GPUs/replica, batch=%d",
                ms.model_label,
                ms.num_replicas,
                ms.gpus_per_replica,
                ms.initial_batch_size,
            )

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    @property
    def models(self) -> list[LLMInferenceModelSpec]:
        return list(self._models)

    @property
    def batch_by_model(self) -> dict[str, int]:
        return dict(self._batch_by_model)

    @property
    def state(self) -> OfflineDatacenterState:
        if self._state is None:
            raise RuntimeError("OfflineDatacenter.state accessed before first step().")
        return self._state

    def history(self, n: int | None = None) -> list[OfflineDatacenterState]:
        if n is None:
            return list(self._history)
        if n <= 0:
            return []
        return list(self._history[-int(n) :])

    def step(self, clock: SimulationClock) -> OfflineDatacenterState:
        t_now = clock.time_s

        per_gpu_by_model: dict[str, np.ndarray] = {}
        for ms in self._models:
            label = ms.model_label
            if ms.num_replicas <= 0:
                continue
            if label not in self._batch_by_model:
                raise KeyError(f"Missing required batch size for model {label!r}")
            batch = int(self._batch_by_model[label])

            layout = self._layouts[label]
            template = self._template_store.template(label, batch)
            L = len(template)
            indices = (self._global_step + layout.stagger_offsets) % L
            per_gpu_by_model[label] = template[indices]

        aug = self._augmenter.step(per_gpu_by_model, t_now)

        power_by_model = dict(aug.power_by_model_w)
        active_replicas_by_model = dict(aug.active_replicas_by_model)
        for ms in self._models:
            power_by_model.setdefault(ms.model_label, 0.0)
            active_replicas_by_model.setdefault(ms.model_label, 0)

        phase_power = np.array([aug.power_w.a, aug.power_w.b, aug.power_w.c])

        # Training overlay
        t_arr = np.asarray(t_now, dtype=float)
        for overlay, tr in self._training_overlays:
            training_power_w = float(
                overlay.eval_total_on_grid(
                    t_arr,
                    t_add_start=tr.t_start,
                    t_add_end=tr.t_end,
                    n_train_gpus=tr.n_gpus,
                )
            )
            phase_power += training_power_w / 3.0

        # ITL sampling
        observed_itl_s_by_model: dict[str, float] = {}
        for ms in self._models:
            label = ms.model_label
            n_rep = active_replicas_by_model.get(label, 0)
            if self._itl_fits is None or n_rep <= 0:
                observed_itl_s_by_model[label] = float("nan")
                continue
            batch = int(self._batch_by_model[label])
            observed_itl_s_by_model[label] = self._itl_fits.sample_avg(
                model_label=label,
                batch_size=batch,
                n_replicas=n_rep,
                rng=self._latency_rng,
            )

        state = OfflineDatacenterState(
            time_s=float(t_now),
            power_w=ThreePhase(
                a=float(phase_power[0]),
                b=float(phase_power[1]),
                c=float(phase_power[2]),
            ),
            power_by_model_w=power_by_model,
            active_replicas_by_model=active_replicas_by_model,
            batch_size_by_model=dict(self._batch_by_model),
            observed_itl_s_by_model=observed_itl_s_by_model,
        )
        self._global_step += 1
        self._state = state
        self._history.append(state)
        return state

    @functools.singledispatchmethod
    def apply_control(self, command: DatacenterCommand) -> None:
        """Apply a control command. Dispatches on command type."""
        raise TypeError(f"OfflineDatacenter does not support {type(command).__name__}")

    @apply_control.register
    def apply_control_set_batch_size(self, command: SetBatchSize) -> None:
        """Record new batch sizes. Changes take effect on the next step."""
        if command.ramp_up_rate_by_model:
            raise ValueError(
                f"OfflineDatacenter does not support ramp_up_rate_by_model (got {command.ramp_up_rate_by_model}). "
                f"Batch size changes are always immediate in trace-based simulation."
            )
        for label, b in command.batch_size_by_model.items():
            b_int = int(b)
            if b_int <= 0:
                raise ValueError(f"Batch size must be positive for model {label!r}, got {b_int}.")
            old = self._batch_by_model.get(str(label))
            self._batch_by_model[str(label)] = b_int
            if old != b_int:
                logger.info("Batch size %s: %s -> %d", label, old, b_int)
        if self._events is not None:
            self._events.emit(
                "datacenter.batch_size.updated",
                {"batch_size_by_model": dict(self._batch_by_model)},
            )

    def reset(self) -> None:
        self._state = None
        self._history = []
        self._global_step = 0
        self._batch_by_model = {ms.model_label: ms.initial_batch_size for ms in self._models}
        self._layout_rng = np.random.default_rng(self._seed)
        self._layouts = {}
        self._build_all_layouts()
        self._augmenter = PowerAugmenter(
            layouts=self._layouts,
            base_w_per_phase=self._base_W_per_phase,
            seed=self._seed + 12345,
        )
        self._latency_rng = np.random.default_rng(self._seed + 54321)

    def bind_event_emitter(self, emitter: EventEmitter) -> None:
        self._events = emitter

    def _build_all_layouts(self) -> None:
        """Eagerly build layouts for all models with replicas > 0."""
        default_strategy = RampActivationStrategy(ServerRampSchedule(entries=()))
        strategy = self._activation_strategy or default_strategy
        for ms in self._models:
            if ms.num_replicas > 0:
                any_batch = self.batch_sizes(ms.model_label)[0]
                tpl_len = len(self._template_store.template(ms.model_label, any_batch))
                self._layouts[ms.model_label] = ServerLayout.build(
                    ms,
                    gpus_per_server=self._gpus_per_server,
                    stagger_range=tpl_len,
                    activation_strategy=strategy,
                    amplitude_scale_range=self._amplitude_scale_range,
                    noise_fraction=self._noise_fraction,
                    rng=self._layout_rng,
                )

    def batch_sizes(self, model_label: str) -> list[int]:
        """Delegate to the template store for available batch sizes."""
        return self._template_store.batch_sizes(model_label)

    @property
    def phase_share_by_model(self) -> dict[str, np.ndarray]:
        """Per-model phase share vectors derived from server placement.

        Returns:
            Mapping of model label to a 3-element array `[frac_A, frac_B, frac_C]`
                representing the fraction of servers on each phase.
        """
        shares: dict[str, np.ndarray] = {}
        for label, layout in self._layouts.items():
            counts = np.bincount(layout.phase_list, minlength=3).astype(float)
            total = counts.sum()
            if total > 0:
                shares[label] = counts / total
            else:
                shares[label] = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        return shares
