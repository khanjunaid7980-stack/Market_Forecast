"""Rationality Gate — compares the market-implied growth rate against the
company's own 5-year empirical growth distribution.

Verdict scale:
  Rational      z ≤ 1.0
  Stretched     1.0 < z ≤ 2.0
  Speculative   z > 2.0
  Insufficient  fewer than 3 data points
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RationalityResult:
    implied_growth: float
    historical_mean: float
    historical_std: float
    z_score: float
    verdict: str           # Rational | Stretched | Speculative | Insufficient Data
    verdict_color: str     # green | orange | red | gray
    analyst_growth: float | None
    analyst_delta: float | None    # implied - analyst (positive = market expects MORE)

    @property
    def verdict_emoji(self) -> str:
        return {
            "Rational": "✅",
            "Stretched": "⚠️",
            "Speculative": "🚨",
            "Insufficient Data": "ℹ️",
        }.get(self.verdict, "")


def _robust_stats(xs: list[float]) -> tuple[float, float]:
    """Winsorised mean and std (clip at 2 std before computing std)."""
    arr = np.array([x for x in xs if np.isfinite(x)], dtype=float)
    if len(arr) < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    std_raw = float(np.std(arr, ddof=1))
    # Winsorise at ±3 raw std then recompute
    clipped = np.clip(arr, mean - 3 * std_raw, mean + 3 * std_raw)
    return float(np.mean(clipped)), float(np.std(clipped, ddof=1))


def rationality_check(
    implied_growth: float,
    rev_growth_history: Sequence[float],
    eps_growth_history: Sequence[float],
    analyst_growth_5y: float | None = None,
) -> RationalityResult:
    pool = list(rev_growth_history) + list(eps_growth_history)
    finite = [x for x in pool if np.isfinite(x)]

    if len(finite) < 3:
        return RationalityResult(
            implied_growth=implied_growth,
            historical_mean=float("nan"),
            historical_std=float("nan"),
            z_score=0.0,
            verdict="Insufficient Data",
            verdict_color="#8b9ab5",
            analyst_growth=analyst_growth_5y,
            analyst_delta=None,
        )

    mean, std = _robust_stats(finite)
    if std < 1e-6:
        std = abs(mean) * 0.10 + 0.01  # floor at 1 pp

    z = (implied_growth - mean) / std

    if z > 2.0:
        verdict, color = "Speculative", "#ef4444"
    elif z > 1.0:
        verdict, color = "Stretched", "#f59e0b"
    else:
        verdict, color = "Rational", "#22c55e"

    analyst_delta = (
        (implied_growth - analyst_growth_5y)
        if analyst_growth_5y is not None else None
    )

    return RationalityResult(
        implied_growth=implied_growth,
        historical_mean=mean,
        historical_std=std,
        z_score=float(z),
        verdict=verdict,
        verdict_color=color,
        analyst_growth=analyst_growth_5y,
        analyst_delta=analyst_delta,
    )


__all__ = ["RationalityResult", "rationality_check"]
