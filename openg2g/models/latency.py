"""Compatibility wrappers for latency modeling.

OpenG2G now uses the ML.ENERGY data toolkit as the source of latency profile
implementations.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from mlenergy_data_toolkit.modeling.latency import ITLMixture2Params, LatencyFitTable
except ImportError as exc:
    workspace_toolkit = Path(__file__).resolve().parents[2] / "mlenergy-data-toolkit"
    if workspace_toolkit.exists():
        sys.path.insert(0, str(workspace_toolkit))
        from mlenergy_data_toolkit.modeling.latency import ITLMixture2Params, LatencyFitTable
    else:
        raise ImportError(
            "mlenergy_data_toolkit is required. Install it with "
            "`pip install mlenergy-data-toolkit` or, in this workspace, "
            "`pip install -e mlenergy-data-toolkit`."
        ) from exc

__all__ = [
    "ITLMixture2Params",
    "LatencyFitTable",
]
