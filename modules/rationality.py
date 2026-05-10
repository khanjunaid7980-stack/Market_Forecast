"""Rationality Check — classify market-implied growth vs. history.

Rule:
    z = (g_implied − μ_hist) / σ_hist
    z >  +2  → "Speculative"     (market expects far above realised)
    z <  −2  → "Pessimistic"     (market expects far below realised)
    else      → "Rational"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RationalityVerdict:
    implied_growth: float
    hist_mean: float
    hist_std: float
    z_score: float
    verdict: str
    rationality_gap: float | None
    color: str


def historical_growth_stats(revenue: pd.Series) -> tuple[float | None, float | None, list[float]]:
    s = revenue.dropna().astype(float)
    if len(s) < 3:
        return None, None, []
    growths = s.pct_change().dropna().tolist()
    if not growths:
        return None, None, []
    mean = float(np.mean(growths))
    std = float(np.std(growths, ddof=1)) if len(growths) > 1 else 0.0
    return mean, std, growths


def assess(implied_growth: float | None, revenue: pd.Series) -> RationalityVerdict:
    nan = float("nan")
    if implied_growth is None:
        return RationalityVerdict(nan, nan, nan, nan, "Insufficient Data", None, "#9ca3af")

    mean, std, _ = historical_growth_stats(revenue)
    if mean is None:
        return RationalityVerdict(implied_growth, nan, nan, nan, "Insufficient Data", None, "#9ca3af")

    z = (implied_growth - mean) / std if std and std > 0 else 0.0
    gap = implied_growth - mean

    if z > 2.0:
        verdict, color = "Speculative", "#ef4444"
    elif z < -2.0:
        verdict, color = "Pessimistic", "#f59e0b"
    else:
        verdict, color = "Rational", "#22c55e"

    return RationalityVerdict(implied_growth, mean, std or 0.0, z, verdict, gap, color)


__all__ = ["RationalityVerdict", "historical_growth_stats", "assess"]
