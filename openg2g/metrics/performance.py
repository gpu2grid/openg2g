"""Inference performance metrics: throughput and latency (ITL).

Companion to [`VoltageStats`][openg2g.metrics.voltage.VoltageStats]. Where
voltage metrics capture grid-side quality, these metrics capture
datacenter-side quality of service -- how many tokens per second the
fleet serves and how often observed inter-token latency crossed each
model's deadline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openg2g.datacenter.base import LLMDatacenterState


@dataclass
class PerformanceStats:
    """Summary throughput and ITL statistics over a simulation run.

    Throughput is derived per step from the observed ITL:
    `tokens/s_per_replica = batch_size / observed_itl_s`, then summed
    across active replicas. This uses the simulator's sampled ITL rather
    than a fitted curve, so it reflects the actual latency realised in
    the run.

    Attributes:
        mean_throughput_tps: Time-averaged total throughput across all
            models and sites (tokens/s).
        integrated_throughput_tokens: Total tokens produced over the run
            (tokens).
        throughput_by_model_tps: Per-model mean throughput (tokens/s).
        mean_batch_by_model: Per-model mean batch size.
        mean_replicas_by_model: Per-model mean active replica count.
        mean_itl_s_by_model: Per-model mean observed ITL (seconds).
        itl_deadline_fraction: Fraction of (timestep, model) samples where
            observed ITL exceeded the model's deadline, across all models.
        itl_deadline_fraction_by_model: Same fraction, per model.
    """

    mean_throughput_tps: float
    integrated_throughput_tokens: float
    throughput_by_model_tps: dict[str, float] = field(default_factory=dict)
    mean_batch_by_model: dict[str, float] = field(default_factory=dict)
    mean_replicas_by_model: dict[str, float] = field(default_factory=dict)
    mean_itl_s_by_model: dict[str, float] = field(default_factory=dict)
    itl_deadline_fraction: float = 0.0
    itl_deadline_fraction_by_model: dict[str, float] = field(default_factory=dict)


def compute_performance_stats(
    dc_states: list[LLMDatacenterState],
    *,
    itl_deadline_s_by_model: dict[str, float],
) -> PerformanceStats:
    """Compute throughput and ITL statistics from datacenter state snapshots.

    Per-replica throughput is estimated as `batch / observed_itl_s`
    (tokens/s). Total throughput at each step is the sum across models
    and sites of `throughput_per_replica * active_replicas`.

    Args:
        dc_states: All datacenter state snapshots (flat across sites).
            Typically `SimulationLog.dc_states`.
        itl_deadline_s_by_model: Per-model ITL deadline (seconds). A
            sample counts as a deadline miss when observed ITL > this.

    Returns:
        A [`PerformanceStats`][.PerformanceStats] with aggregated metrics.

    Raises:
        ValueError: If `dc_states` is empty or snapshot times are not
            monotonically non-decreasing.
    """
    if not dc_states:
        raise ValueError("compute_performance_stats requires at least one DC state.")

    times = np.asarray([float(s.time_s) for s in dc_states], dtype=float)
    duration_s = float(times.max() - times.min())
    if duration_s <= 0.0:
        duration_s = 1.0

    per_model_tps_sum: dict[str, float] = {}
    per_model_tps_count: dict[str, int] = {}
    per_model_batch_sum: dict[str, float] = {}
    per_model_reps_sum: dict[str, float] = {}
    per_model_itl_sum: dict[str, float] = {}
    per_model_itl_count: dict[str, int] = {}
    per_model_itl_viol: dict[str, int] = {}

    total_tps_sum = 0.0
    total_tps_count = 0
    total_itl_viol = 0
    total_itl_count = 0

    for state in dc_states:
        step_tps = 0.0
        for label, batch in state.batch_size_by_model.items():
            n_rep = state.active_replicas_by_model.get(label, 0)
            itl = state.observed_itl_s_by_model.get(label, float("nan"))

            per_model_batch_sum[label] = per_model_batch_sum.get(label, 0.0) + float(batch)
            per_model_reps_sum[label] = per_model_reps_sum.get(label, 0.0) + float(n_rep)
            per_model_tps_count[label] = per_model_tps_count.get(label, 0) + 1

            if n_rep <= 0 or batch <= 0 or not np.isfinite(itl) or itl <= 0.0:
                continue
            tps_per_replica = float(batch) / float(itl)
            tps_model = tps_per_replica * float(n_rep)
            per_model_tps_sum[label] = per_model_tps_sum.get(label, 0.0) + tps_model
            step_tps += tps_model

            per_model_itl_sum[label] = per_model_itl_sum.get(label, 0.0) + float(itl)
            per_model_itl_count[label] = per_model_itl_count.get(label, 0) + 1
            total_itl_count += 1
            deadline = float(itl_deadline_s_by_model[label])
            if itl > deadline:
                per_model_itl_viol[label] = per_model_itl_viol.get(label, 0) + 1
                total_itl_viol += 1

        total_tps_sum += step_tps
        total_tps_count += 1

    # Throughput metrics are fleet-totals (summed across sites), averaged over time.
    # `total_tps_count` and `per_model_tps_count` are counts of (site, timestep) samples,
    # so normalise by the number of unique timesteps to avoid reporting per-site averages.
    n_timesteps = len({float(s.time_s) for s in dc_states})
    mean_total_tps = total_tps_sum / n_timesteps if n_timesteps > 0 else 0.0
    integrated_tokens = mean_total_tps * duration_s

    throughput_by_model = {label: per_model_tps_sum[label] / n_timesteps for label in per_model_tps_count}
    # `mean_batch_by_model`, `mean_replicas_by_model`, and `mean_itl_s_by_model` stay per
    # (site, timestep) sample; those quantities aren't summable across sites in a meaningful way.
    mean_batch = {label: per_model_batch_sum[label] / per_model_tps_count[label] for label in per_model_tps_count}
    mean_reps = {label: per_model_reps_sum[label] / per_model_tps_count[label] for label in per_model_tps_count}
    mean_itl = {label: per_model_itl_sum[label] / per_model_itl_count[label] for label in per_model_itl_count}
    itl_deadline_fraction_by_model = {
        label: per_model_itl_viol.get(label, 0) / per_model_itl_count[label] for label in per_model_itl_count
    }
    itl_deadline_fraction = total_itl_viol / total_itl_count if total_itl_count > 0 else 0.0

    return PerformanceStats(
        mean_throughput_tps=mean_total_tps,
        integrated_throughput_tokens=integrated_tokens,
        throughput_by_model_tps=throughput_by_model,
        mean_batch_by_model=mean_batch,
        mean_replicas_by_model=mean_reps,
        mean_itl_s_by_model=mean_itl,
        itl_deadline_fraction=itl_deadline_fraction,
        itl_deadline_fraction_by_model=itl_deadline_fraction_by_model,
    )
