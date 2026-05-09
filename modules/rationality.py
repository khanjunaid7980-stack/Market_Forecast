"""Rationality Gate — compares the market-implied growth rate against the
company's own historical growth distribution.

Verdict ladder:
  Rational      z ≤ 1.0
  Stretched     1.0 < z ≤ 2.0
  Speculative   2.0 < z ≤ 3.0
  Bubble-like   z > 3.0
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
    historical_min: float
    historical_max: float
    historical_n: int
    z_score: float
    verdict: str
    verdict_color: str
    rationale: str
    analyst_growth: float | None
    analyst_delta: float | None

    @property
    def emoji(self) -> str:
        return {
            "Rational": "✅", "Stretched": "⚠️",
            "Speculative": "🚨", "Bubble-like": "🔥",
            "Insufficient Data": "ℹ️",
        }.get(self.verdict, "")


def _winsorised_stats(xs: list[float]) -> tuple[float, float]:
    arr = np.array([x for x in xs if np.isfinite(x)], dtype=float)
    if len(arr) < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    std_raw = float(np.std(arr, ddof=1))
    if std_raw == 0:
        return mean, 0.0
    clipped = np.clip(arr, mean - 3 * std_raw, mean + 3 * std_raw)
    return float(np.mean(clipped)), float(np.std(clipped, ddof=1))


def rationality_check(
    implied_growth: float,
    rev_growth_history: Sequence[float],
    eps_growth_history: Sequence[float],
    fcf_growth_history: Sequence[float] = (),
    analyst_growth_5y: float | None = None,
) -> RationalityResult:
    pool = [x for x in list(rev_growth_history) + list(eps_growth_history)
            + list(fcf_growth_history) if np.isfinite(x)]

    if len(pool) < 3:
        return RationalityResult(
            implied_growth=implied_growth, historical_mean=float("nan"),
            historical_std=float("nan"), historical_min=float("nan"),
            historical_max=float("nan"), historical_n=len(pool),
            z_score=0.0, verdict="Insufficient Data",
            verdict_color="#7d8693",
            rationale="Fewer than 3 historical growth observations — z-score not meaningful.",
            analyst_growth=analyst_growth_5y, analyst_delta=None,
        )

    mean, std = _winsorised_stats(pool)
    if std < 1e-6:
        std = abs(mean) * 0.10 + 0.01
    z = (implied_growth - mean) / std

    if z > 3.0:
        verdict, color = "Bubble-like", "#dc2626"
        rationale = (
            f"Market-implied growth ({implied_growth*100:.1f}%) exceeds the "
            f"5-year historical mean ({mean*100:.1f}%) by more than 3σ. "
            "Today's price requires a regime change well beyond anything the "
            "company has demonstrated."
        )
    elif z > 2.0:
        verdict, color = "Speculative", "#ef4444"
        rationale = (
            f"Implied growth sits >2σ above the historical mean. "
            "The market is pricing a meaningfully better future than the past."
        )
    elif z > 1.0:
        verdict, color = "Stretched", "#f59e0b"
        rationale = (
            "Implied growth is >1σ above historical — elevated but not extreme. "
            "Justifiable for high-quality compounders."
        )
    else:
        verdict, color = "Rational", "#22c55e"
        rationale = (
            "Implied growth is within ±1σ of the historical mean. "
            "Today's price aligns with what the company has actually delivered."
        )

    analyst_delta = (
        implied_growth - analyst_growth_5y
        if analyst_growth_5y is not None else None
    )
    return RationalityResult(
        implied_growth=implied_growth,
        historical_mean=mean, historical_std=std,
        historical_min=float(min(pool)), historical_max=float(max(pool)),
        historical_n=len(pool),
        z_score=float(z), verdict=verdict, verdict_color=color,
        rationale=rationale,
        analyst_growth=analyst_growth_5y, analyst_delta=analyst_delta,
    )


__all__ = ["RationalityResult", "rationality_check"]
