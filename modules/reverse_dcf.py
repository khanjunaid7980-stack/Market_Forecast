"""Reverse-DCF: solve for the revenue growth rate the market is pricing in.

Given the current market cap M, base revenue R0, FCF margin m, WACC r, and
terminal growth g_T, find the growth rate g such that a two-stage DCF projects
back to the market's equity value:

    DCF(g) = Σ_{t=1..N} R0·(1+g)^t · m / (1+r)^t
           + R0·(1+g)^N · m · (1+g_T) / (r − g_T) / (1+r)^N
           − NetDebt
           = M

Solved with Brent's method on g ∈ [-30%, +50%].
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import brentq


@dataclass
class ImpliedResult:
    implied_growth: float | None
    forecast_years: int
    fcf_margin: float
    wacc: float
    terminal_growth: float
    market_cap: float
    base_revenue: float
    net_debt: float
    converged: bool


def dcf_value(
    g: float,
    base_revenue: float,
    fcf_margin: float,
    wacc: float,
    terminal_growth: float,
    years: int,
    net_debt: float = 0.0,
) -> float:
    """Two-stage DCF equity value at growth rate g."""
    if wacc <= terminal_growth:
        return float("inf")
    pv = 0.0
    for t in range(1, years + 1):
        rev_t = base_revenue * (1.0 + g) ** t
        pv += rev_t * fcf_margin / (1.0 + wacc) ** t
    rev_N = base_revenue * (1.0 + g) ** years
    fcf_terminal = rev_N * fcf_margin * (1.0 + terminal_growth)
    tv = fcf_terminal / (wacc - terminal_growth)
    pv += tv / (1.0 + wacc) ** years
    return pv - net_debt


def solve_implied_growth(
    market_cap: float,
    base_revenue: float,
    fcf_margin: float,
    wacc: float,
    terminal_growth: float = 0.025,
    years: int = 10,
    net_debt: float = 0.0,
) -> ImpliedResult:
    """Brent's method on f(g) = DCF(g) − M."""
    def f(g: float) -> float:
        return dcf_value(g, base_revenue, fcf_margin, wacc, terminal_growth, years, net_debt) - market_cap

    base = ImpliedResult(
        implied_growth=None, forecast_years=years, fcf_margin=fcf_margin, wacc=wacc,
        terminal_growth=terminal_growth, market_cap=market_cap, base_revenue=base_revenue,
        net_debt=net_debt, converged=False,
    )
    try:
        f_lo, f_hi = f(-0.30), f(0.50)
        if f_lo * f_hi > 0:
            return base
        g_solved = brentq(f, -0.30, 0.50, xtol=1e-5, maxiter=200)
        return ImpliedResult(
            implied_growth=float(g_solved), forecast_years=years, fcf_margin=fcf_margin,
            wacc=wacc, terminal_growth=terminal_growth, market_cap=market_cap,
            base_revenue=base_revenue, net_debt=net_debt, converged=True,
        )
    except (ValueError, RuntimeError):
        return base


__all__ = ["ImpliedResult", "dcf_value", "solve_implied_growth"]
