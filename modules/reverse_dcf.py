"""Reverse-DCF: solve for the revenue CAGR the market is pricing into the stock.

Enterprise Value basis (academically correct):

    EV  =  Σ_{t=1}^{N}  FCFF_t / (1+WACC)^t   +   TV / (1+WACC)^N
    EV  =  Market Cap  +  Net Debt

Where:
    FCFF_t  =  Revenue₀ × (1+g)^t  ×  fcf_margin
    TV      =  FCFF_N   × (1+g_T)  /  (WACC − g_T)   [Gordon Growth]

We solve for g (the implied revenue CAGR) such that the DCF equals the
observed enterprise value. Root-finding via Brent's method.

fcf_margin = FCFF / Revenue = steady-state free-cash-flow-to-revenue ratio.
Callers should derive this from the 3-year median of historical (CFO − CapEx) / Revenue,
or from the NOPAT-based approach when CFO data is sparse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq


@dataclass
class ImpliedResult:
    # ── Primary output ──────────────────────────────────────────
    implied_growth: float | None   # solved CAGR (decimal). None if not converged.
    converged: bool

    # ── EV decomposition (populated when converged=True) ────────
    enterprise_value: float = 0.0   # market_cap + net_debt
    pv_explicit: float = 0.0        # PV of explicit-period FCFFs
    pv_terminal: float = 0.0        # PV of terminal value
    tv_pct: float = 0.0             # pv_terminal / enterprise_value  (0–1)
    implied_revenue_final: float = 0.0  # revenue at end of forecast horizon

    # ── Inputs echoed for downstream use ────────────────────────
    forecast_years: int = 10
    fcf_margin: float = 0.0
    wacc: float = 0.0
    terminal_growth: float = 0.0
    market_cap: float = 0.0
    base_revenue: float = 0.0
    net_debt: float = 0.0

    @property
    def failure_reason(self) -> str:
        if self.converged:
            return ""
        if self.fcf_margin <= 0:
            return "FCF margin ≤ 0 — company is cash-negative; DCF requires positive free cash flow."
        if self.wacc <= self.terminal_growth:
            return f"WACC ({self.wacc*100:.1f}%) must exceed terminal growth ({self.terminal_growth*100:.1f}%)."
        if self.enterprise_value <= 0:
            return "Enterprise value ≤ 0 (market cap < net debt); DCF basis undefined."
        return "Implied growth lies outside the solvable bracket. Adjust inputs."


def _pv_breakdown(
    g: float,
    base_revenue: float,
    fcf_margin: float,
    wacc: float,
    terminal_growth: float,
    years: int,
) -> tuple[float, float]:
    """(pv_explicit_period, pv_terminal_value). Both pre-net-debt."""
    pv_ex = sum(
        base_revenue * (1.0 + g) ** t * fcf_margin / (1.0 + wacc) ** t
        for t in range(1, years + 1)
    )
    rev_N = base_revenue * (1.0 + g) ** years
    tv = rev_N * fcf_margin * (1.0 + terminal_growth) / (wacc - terminal_growth)
    pv_tv = tv / (1.0 + wacc) ** years
    return pv_ex, pv_tv


def dcf_value(
    g: float,
    base_revenue: float,
    fcf_margin: float,
    wacc: float,
    terminal_growth: float,
    years: int,
    net_debt: float = 0.0,
) -> float:
    """Equity value = PV(all FCFFs) − Net Debt at revenue growth rate g."""
    if wacc <= terminal_growth:
        return float("inf")
    pv_ex, pv_tv = _pv_breakdown(g, base_revenue, fcf_margin, wacc, terminal_growth, years)
    return pv_ex + pv_tv - net_debt


def solve_implied_growth(
    market_cap: float,
    base_revenue: float,
    fcf_margin: float,
    wacc: float,
    terminal_growth: float = 0.025,
    years: int = 10,
    net_debt: float = 0.0,
) -> ImpliedResult:
    """Find the revenue CAGR g such that DCF(g) = market_cap.

    Uses Brent's method with progressively wider brackets to maximise
    the chance of finding a solution for extreme valuations.
    """
    ev = market_cap + net_debt

    base_kw = dict(
        forecast_years=years, fcf_margin=fcf_margin, wacc=wacc,
        terminal_growth=terminal_growth, market_cap=market_cap,
        base_revenue=base_revenue, net_debt=net_debt,
        enterprise_value=ev,
    )
    failed = ImpliedResult(implied_growth=None, converged=False, **base_kw)

    # Pre-flight checks — each is a reason the DCF has no solution.
    if fcf_margin <= 0.0:
        return failed
    if wacc <= terminal_growth:
        return failed
    if base_revenue <= 0.0:
        return failed
    if ev <= 0.0:
        return failed

    def f(g: float) -> float:
        return dcf_value(g, base_revenue, fcf_margin, wacc, terminal_growth, years, net_debt) - market_cap

    # Try progressively wider brackets; real-world g almost never leaves [-40%, +80%].
    for lo, hi in [(-0.30, 0.50), (-0.40, 0.70), (-0.50, 0.90)]:
        try:
            f_lo, f_hi = f(lo), f(hi)
            if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
                continue
            if f_lo * f_hi >= 0:
                continue  # no sign change in this bracket
            g_solved = brentq(f, lo, hi, xtol=1e-6, maxiter=400)
            pv_ex, pv_tv = _pv_breakdown(g_solved, base_revenue, fcf_margin, wacc, terminal_growth, years)
            total_pv = pv_ex + pv_tv
            tv_pct = pv_tv / total_pv if total_pv > 0 else 0.0
            return ImpliedResult(
                implied_growth=float(g_solved),
                converged=True,
                pv_explicit=pv_ex,
                pv_terminal=pv_tv,
                tv_pct=tv_pct,
                implied_revenue_final=base_revenue * (1.0 + g_solved) ** years,
                **base_kw,
            )
        except (ValueError, RuntimeError):
            continue

    return failed


__all__ = ["ImpliedResult", "dcf_value", "solve_implied_growth"]
