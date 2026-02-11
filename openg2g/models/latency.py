"""ITL mixture model and latency sampling table."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ITLMixture2Params:
    """ITL = loc + LogNormal, mixture of two components (steady + stall).

    Y ~ LogNormal(meanlog=ln(scale), sdlog=sigma)
    X = loc + Y
    """

    loc: float
    pi_steady: float
    sigma_steady: float
    scale_steady: float
    pi_stall: float
    sigma_stall: float
    scale_stall: float

    def _lognormal_mean_var(self, sigma: float, scale: float) -> tuple[float, float]:
        s2 = float(sigma) ** 2
        ey = float(scale) * math.exp(0.5 * s2)
        vy = (float(scale) ** 2) * (math.exp(s2) - 1.0) * math.exp(s2)
        return ey, vy

    def mean_var(self) -> tuple[float, float]:
        """Mixture mean and variance."""
        p1 = float(self.pi_steady)
        p2 = float(self.pi_stall)
        ps = max(p1 + p2, 1e-12)
        p1 /= ps
        p2 /= ps

        m1, v1 = self._lognormal_mean_var(self.sigma_steady, self.scale_steady)
        m2, v2 = self._lognormal_mean_var(self.sigma_stall, self.scale_stall)

        m1x = float(self.loc) + m1
        m2x = float(self.loc) + m2

        mx = p1 * m1x + p2 * m2x

        ex2 = p1 * (v1 + m1x * m1x) + p2 * (v2 + m2x * m2x)
        vx = max(ex2 - mx * mx, 0.0)
        return mx, vx

    def sample_one(self, rng: np.random.Generator) -> float:
        """Draw a single ITL sample."""
        p1 = float(self.pi_steady)
        p2 = float(self.pi_stall)
        ps = max(p1 + p2, 1e-12)
        p1 /= ps

        if rng.random() < p1:
            y = rng.lognormal(
                mean=math.log(max(self.scale_steady, 1e-15)),
                sigma=max(self.sigma_steady, 0.0),
            )
        else:
            y = rng.lognormal(
                mean=math.log(max(self.scale_stall, 1e-15)),
                sigma=max(self.sigma_stall, 0.0),
            )
        return float(self.loc + y)


class LatencyFitTable:
    """Loads per-(model, batch_size) ITL mixture parameters and provides sampling.

    Args:
        csv_path: Path to CSV with columns: model_label, max_num_seqs,
            itl_dist, itl_mix_loc, itl_mix_pi_steady, itl_mix_sigma_steady,
            itl_mix_scale_steady, itl_mix_pi_stall, itl_mix_sigma_stall,
            itl_mix_scale_stall.
        strict: If True, raise on missing or unsupported entries.
        model_label_map: Optional renaming map for model labels in the CSV.
    """

    def __init__(
        self,
        csv_path: str,
        *,
        strict: bool = True,
        model_label_map: dict[str, str] | None = None,
    ):
        self.csv_path = str(csv_path)
        self.strict = bool(strict)
        self.model_label_map = dict(model_label_map or {})

        df = pd.read_csv(self.csv_path)

        need = {
            "model_label",
            "max_num_seqs",
            "itl_dist",
            "itl_mix_loc",
            "itl_mix_pi_steady",
            "itl_mix_sigma_steady",
            "itl_mix_scale_steady",
            "itl_mix_pi_stall",
            "itl_mix_sigma_stall",
            "itl_mix_scale_stall",
        }
        missing = sorted(list(need - set(df.columns)))
        if missing:
            raise ValueError(f"{self.csv_path} missing columns: {missing}")

        self._params: dict[tuple[str, int], ITLMixture2Params] = {}

        for _, row in df.iterrows():
            mdl_raw = str(row["model_label"]).strip()
            mdl = self.model_label_map.get(mdl_raw, mdl_raw)
            b = int(row["max_num_seqs"])  # type: ignore[arg-type]

            dist = str(row["itl_dist"]).strip().lower()
            if dist != "lognormal_mixture_2":
                if self.strict:
                    raise ValueError(f"Unsupported itl_dist={dist!r} for model={mdl} batch={b}")
                else:
                    continue

            p = ITLMixture2Params(
                loc=float(row["itl_mix_loc"]),  # type: ignore[arg-type]
                pi_steady=float(row["itl_mix_pi_steady"]),  # type: ignore[arg-type]
                sigma_steady=float(row["itl_mix_sigma_steady"]),  # type: ignore[arg-type]
                scale_steady=float(row["itl_mix_scale_steady"]),  # type: ignore[arg-type]
                pi_stall=float(row["itl_mix_pi_stall"]),  # type: ignore[arg-type]
                sigma_stall=float(row["itl_mix_sigma_stall"]),  # type: ignore[arg-type]
                scale_stall=float(row["itl_mix_scale_stall"]),  # type: ignore[arg-type]
            )
            self._params[(mdl, b)] = p

        if self.strict and not self._params:
            raise ValueError("No ITL mixture rows loaded. Check model labels / CSV content.")

    def _get(self, model_label: str, batch_size: int) -> ITLMixture2Params:
        key = (str(model_label), int(batch_size))
        if key in self._params:
            return self._params[key]

        candidates = [b for (m, b) in self._params if m == str(model_label)]
        if not candidates:
            raise KeyError(f"No ITL params for model_label={model_label!r}.")
        b_near = min(candidates, key=lambda bb: abs(int(bb) - int(batch_size)))
        if self.strict:
            raise KeyError(
                f"No ITL params for model={model_label!r}, batch={batch_size}. "
                f"Available batches: {sorted(candidates)}"
            )
        return self._params[(str(model_label), int(b_near))]

    def sample_itl_s(
        self,
        *,
        model_label: str,
        batch_size: int,
        rng: np.random.Generator,
    ) -> float:
        """Sample a single ITL value in seconds."""
        p = self._get(model_label, batch_size)
        return p.sample_one(rng)

    def sample_avg_itl_s(
        self,
        *,
        model_label: str,
        batch_size: int,
        n_replicas: int,
        rng: np.random.Generator,
        exact_threshold: int = 30,
    ) -> float:
        """Sample the average ITL across *n_replicas* active replicas.

        If n_replicas <= exact_threshold: draw n i.i.d. samples and average.
        Otherwise: approximate as Normal(mean, var/n).
        """
        n = int(n_replicas)
        if n <= 0:
            return float("nan")

        p = self._get(model_label, batch_size)

        if n <= int(exact_threshold):
            vals = [p.sample_one(rng) for _ in range(n)]
            return float(np.mean(vals))

        mu, var = p.mean_var()
        sd = math.sqrt(max(var / float(n), 0.0))
        x = float(rng.normal(mu, sd))
        return float(max(x, 0.0))
