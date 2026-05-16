"""Rationality Check — classify the market-implied growth vs. company history.

Primary rule (z-score on annual YoY revenue growth):
    z = (g_implied − μ_YoY) / σ_YoY
    z > +2.0  →  "Speculative"   market demands far above historical pace
    z < −2.0  →  "Pessimistic"   market assumes far below historical pace
    else       →  "Rational"      implied growth is within the historical band

Secondary reference:
    Compare implied g also against the 3-year and 5-year realised CAGRs,
    which are more stable than YoY mean for volatile revenue series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RationalityVerdict:
    implied_growth: float           # market-implied CAGR (decimal)
    hist_mean: float                # mean of annual YoY growth rates
    hist_std: float                 # std-dev of annual YoY growth rates
    hist_cagr_3yr: float | None     # realised 3-year CAGR
    hist_cagr_5yr: float | None     # realised 5-year CAGR
    z_score: float                  # (implied − μ) / σ
    verdict: str                    # Speculative | Rational | Pessimistic | Insufficient Data
    rationality_gap: float | None   # implied − hist_mean (percentage points if × 100)
    color: str                      # hex colour for UI badge


def _cagr(s: pd.Series, n: int) -> float | None:
    s = s.dropna().astype(float)
    if len(s) < n + 1:
        return None
    end, start = float(s.iloc[-1]), float(s.iloc[-(n + 1)])
    if start <= 0 or end <= 0:
        return None
    return (end / start) ** (1.0 / n) - 1.0


def historical_growth_stats(revenue: pd.Series) -> tuple[float | None, float | None, list[float]]:
    """Returns (mean_yoy, std_yoy, list_of_yoy_rates)."""
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

    if implied_growth is None or not np.isfinite(implied_growth):
        return RationalityVerdict(
            nan, nan, nan, None, None, nan,
            "Insufficient Data", None, "#6b7280",
        )

    mean, std, _ = historical_growth_stats(revenue)
    cagr_3 = _cagr(revenue, 3)
    cagr_5 = _cagr(revenue, 5)

    if mean is None:
        return RationalityVerdict(
            implied_growth, nan, nan, cagr_3, cagr_5, nan,
            "Insufficient Data", None, "#6b7280",
        )

    z = (implied_growth - mean) / std if (std and std > 0) else 0.0
    gap = implied_growth - mean

    if z > 2.0:
        verdict, color = "Speculative", "#ef4444"
    elif z > 1.0:
        verdict, color = "Elevated", "#f59e0b"
    elif z < -2.0:
        verdict, color = "Pessimistic", "#f59e0b"
    elif z < -1.0:
        verdict, color = "Cautious", "#60a5fa"
    else:
        verdict, color = "Rational", "#22c55e"

    return RationalityVerdict(
        implied_growth=implied_growth,
        hist_mean=mean,
        hist_std=std or 0.0,
        hist_cagr_3yr=cagr_3,
        hist_cagr_5yr=cagr_5,
        z_score=z,
        verdict=verdict,
        rationality_gap=gap,
        color=color,
    )


__all__ = ["RationalityVerdict", "historical_growth_stats", "assess"]
