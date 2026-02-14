"""Logistic fit models for power, latency, and throughput as functions of log2(batch_size)."""

from __future__ import annotations

from mlenergy_data.modeling.logistic import (
    LogisticFit,
    LogisticParams,
    load_logistic_fits,
    load_logistic_fits_merged,
)

__all__ = [
    "LogisticFit",
    "LogisticParams",
    "load_logistic_fits",
    "load_logistic_fits_merged",
]
