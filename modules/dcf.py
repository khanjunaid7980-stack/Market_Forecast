"""Reverse-DCF and Intrinsic-DCF engines.

Reverse-DCF:  bisection solver that finds the high-growth rate g_h the
              market implicitly requires to justify today's enterprise value.

Intrinsic-DCF: standard revenue-driven two-stage model that returns a
               fair value per share given user-supplied assumptions.

Both use a linear fade from the explicit high-growth rate to the terminal
rate to prevent the terminal value from being artificially inflated.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# assumptions
# ---------------------------------------------------------------------------

@dataclass
class WaccInputs:
    beta: float = 1.0
    risk_free_rate: float = 0.045
    equity_risk_premium: float = 0.055
    cost_of_debt_pretax: float = 0.05
    tax_rate: float = 0.21
    debt_weight: float = 0.20

    @property
    def cost_of_equity(self) -> float:
        return self.risk_free_rate + self.beta * self.equity_risk_premium

    @property
    def after_tax_kd(self) -> float:
        return self.cost_of_debt_pretax * (1 - self.tax_rate)

    @property
    def wacc(self) -> float:
        we = 1 - self.debt_weight
        return we * self.cost_of_equity + self.debt_weight * self.after_tax_kd


@dataclass
class ReverseDcfInputs:
    fcf_base: float          # base-year FCF (3y avg preferred)
    net_debt: float          # total debt - cash
    shares: float
    wacc: float
    terminal_g: float = 0.025
    high_growth_years: int = 5
    fade_years: int = 5


@dataclass
class IntrinsicInputs:
    revenue_base: float
    net_debt: float
    shares: float
    revenue_growth: float    = 0.08
    operating_margin: float  = 0.20
    tax_rate: float          = 0.21
    da_pct_revenue: float    = 0.03
    capex_pct_revenue: float = 0.04
    nwc_pct_revenue: float   = 0.02
    wacc: float              = 0.09
    terminal_g: float        = 0.025
    projection_years: int    = 10

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# two-stage enterprise PV
# ---------------------------------------------------------------------------

def enterprise_pv(growth: float, inp: ReverseDcfInputs) -> float:
    """PV of enterprise FCFs given an explicit high-growth rate."""
    if inp.wacc <= inp.terminal_g:
        return float("inf")
    pv = 0.0
    fcf = inp.fcf_base
    t = 0
    # Stage 1: constant high growth
    for _ in range(inp.high_growth_years):
        fcf *= 1 + growth
        t += 1
        pv += fcf / (1 + inp.wacc) ** t
    # Stage 2: linear fade to terminal g
    for i in range(1, inp.fade_years + 1):
        g_fade = growth + (inp.terminal_g - growth) * i / inp.fade_years
        fcf *= 1 + g_fade
        t += 1
        pv += fcf / (1 + inp.wacc) ** t
    # Terminal value
    tv = fcf * (1 + inp.terminal_g) / (inp.wacc - inp.terminal_g)
    pv += tv / (1 + inp.wacc) ** t
    return pv


# ---------------------------------------------------------------------------
# reverse-DCF
# ---------------------------------------------------------------------------

@dataclass
class ReverseDcfResult:
    implied_growth: float
    converged: bool
    ev_target: float
    ev_model: float
    iterations: int
    wacc: float
    terminal_g: float


def reverse_dcf(market_cap: float, inp: ReverseDcfInputs) -> ReverseDcfResult:
    """Bisection: find g_h s.t. enterprise_pv(g_h) == market_cap + net_debt."""
    ev_target = market_cap + inp.net_debt

    lo, hi = -0.30, 1.50

    # Sanity: if even 150% growth can't justify the price, cap it
    ev_hi = enterprise_pv(hi, inp)
    if ev_hi < ev_target:
        return ReverseDcfResult(
            implied_growth=hi, converged=False,
            ev_target=ev_target, ev_model=ev_hi,
            iterations=0, wacc=inp.wacc, terminal_g=inp.terminal_g,
        )

    ev_lo = enterprise_pv(lo, inp)
    if ev_lo > ev_target:
        return ReverseDcfResult(
            implied_growth=lo, converged=False,
            ev_target=ev_target, ev_model=ev_lo,
            iterations=0, wacc=inp.wacc, terminal_g=inp.terminal_g,
        )

    for itr in range(120):
        mid = (lo + hi) / 2
        if enterprise_pv(mid, inp) < ev_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break

    g_solved = (lo + hi) / 2
    return ReverseDcfResult(
        implied_growth=g_solved, converged=True,
        ev_target=ev_target, ev_model=enterprise_pv(g_solved, inp),
        iterations=itr + 1, wacc=inp.wacc, terminal_g=inp.terminal_g,
    )


# ---------------------------------------------------------------------------
# intrinsic DCF
# ---------------------------------------------------------------------------

@dataclass
class IntrinsicResult:
    projections: pd.DataFrame
    terminal_value: float
    pv_terminal: float
    enterprise_value: float
    equity_value: float
    fair_value_per_share: float
    current_price: float | None
    upside_pct: float | None
    tv_pct_of_ev: float
    wacc_used: float


def intrinsic_dcf(
    inp: IntrinsicInputs,
    current_price: float | None = None,
) -> IntrinsicResult:
    """Revenue-driven two-stage DCF (FCFF-based)."""
    wacc = inp.wacc
    g = min(inp.terminal_g, wacc - 0.005)
    rows: list[dict] = []
    rev = inp.revenue_base

    for t in range(1, inp.projection_years + 1):
        rev_t = rev * (1 + inp.revenue_growth)
        ebit = rev_t * inp.operating_margin
        nopat = ebit * (1 - inp.tax_rate)
        da = rev_t * inp.da_pct_revenue
        capex = rev_t * inp.capex_pct_revenue
        dnwc = (rev_t - rev) * inp.nwc_pct_revenue
        fcff = nopat + da - capex - dnwc
        df = (1 + wacc) ** t
        rows.append({
            "Year": t,
            "Revenue": rev_t,
            "EBIT": ebit,
            "NOPAT": nopat,
            "D&A": da,
            "CapEx": capex,
            "ΔWorking Capital": dnwc,
            "FCFF": fcff,
            "Discount Factor": df,
            "PV FCFF": fcff / df,
        })
        rev = rev_t

    proj = pd.DataFrame(rows).set_index("Year")
    last_fcff = float(proj["FCFF"].iloc[-1])
    tv = last_fcff * (1 + g) / (wacc - g)
    pv_tv = tv / (1 + wacc) ** inp.projection_years
    pv_explicit = float(proj["PV FCFF"].sum())
    ev = pv_explicit + pv_tv
    equity = ev - inp.net_debt
    fv = equity / inp.shares if inp.shares > 0 else float("nan")
    upside = (fv / current_price - 1) if (current_price and np.isfinite(fv) and current_price > 0) else None
    tv_pct = pv_tv / ev if ev > 0 else 0.0
    return IntrinsicResult(
        projections=proj, terminal_value=tv, pv_terminal=pv_tv,
        enterprise_value=ev, equity_value=equity,
        fair_value_per_share=fv, current_price=current_price,
        upside_pct=upside, tv_pct_of_ev=tv_pct, wacc_used=wacc,
    )


# ---------------------------------------------------------------------------
# sensitivity grid (for heatmap)
# ---------------------------------------------------------------------------

def sensitivity_grid(
    inp: IntrinsicInputs,
    current_price: float | None = None,
    wacc_steps: tuple[float, ...] = (-0.02, -0.01, 0.0, +0.01, +0.02),
    g_steps: tuple[float, ...] = (-0.01, -0.005, 0.0, +0.005, +0.01),
) -> pd.DataFrame:
    """Fair value per share over a WACC x terminal-g grid."""
    rows: dict[str, dict[str, float]] = {}
    for dw in wacc_steps:
        w = inp.wacc + dw
        row: dict[str, float] = {}
        for dg in g_steps:
            tg = inp.terminal_g + dg
            if w - tg <= 0.005:
                row[f"{tg:.2%}"] = float("nan")
                continue
            i2 = IntrinsicInputs(**{**inp.to_dict(), "wacc": w, "terminal_g": tg})
            r = intrinsic_dcf(i2, current_price)
            row[f"{tg:.2%}"] = round(r.fair_value_per_share, 2)
        rows[f"{w:.2%}"] = row
    df = pd.DataFrame(rows).T
    df.index.name = "WACC"
    df.columns.name = "Terminal g"
    return df


# ---------------------------------------------------------------------------
# fair-value vs growth curve (for sensitivity line chart)
# ---------------------------------------------------------------------------

def fair_value_curve(
    inp: ReverseDcfInputs,
    shares: float,
    current_price: float | None = None,
    n_points: int = 40,
) -> pd.DataFrame:
    """Fair value per share at each growth rate from -10% to +80%."""
    rows = []
    for g in np.linspace(-0.10, 0.80, n_points):
        ev = enterprise_pv(float(g), inp)
        equity = ev - inp.net_debt
        fv = equity / shares if shares > 0 else float("nan")
        rows.append({"growth": float(g), "fair_value": fv})
    df = pd.DataFrame(rows)
    if current_price:
        df["price"] = current_price
    return df


__all__ = [
    "WaccInputs", "ReverseDcfInputs", "IntrinsicInputs",
    "ReverseDcfResult", "IntrinsicResult",
    "enterprise_pv", "reverse_dcf", "intrinsic_dcf",
    "sensitivity_grid", "fair_value_curve",
]
