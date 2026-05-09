"""Reverse-DCF and Intrinsic-DCF engines.

Reverse-DCF: bisection solver for the high-growth rate gʰ that makes the
             two-stage enterprise PV equal the observed EV (mkt cap + net debt).

Intrinsic-DCF: revenue-driven FCFF model for the manual sensitivity tab.

Monte-Carlo wrapper: runs the reverse-DCF over a basket of FCF-base scenarios
(TTM, 3y avg, 5y avg) and returns the mean implied growth + a 1σ band.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# WACC inputs
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


# ---------------------------------------------------------------------------
# DCF inputs
# ---------------------------------------------------------------------------

@dataclass
class ReverseDcfInputs:
    fcf_base: float
    net_debt: float
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
# two-stage enterprise PV with linear fade
# ---------------------------------------------------------------------------

def enterprise_pv(growth: float, inp: ReverseDcfInputs) -> float:
    if inp.wacc <= inp.terminal_g:
        return float("inf")
    pv, fcf, t = 0.0, inp.fcf_base, 0
    for _ in range(inp.high_growth_years):
        fcf *= 1 + growth
        t += 1
        pv += fcf / (1 + inp.wacc) ** t
    for i in range(1, inp.fade_years + 1):
        g = growth + (inp.terminal_g - growth) * i / inp.fade_years
        fcf *= 1 + g
        t += 1
        pv += fcf / (1 + inp.wacc) ** t
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
    """Bisection: find gʰ ∈ [-30%, +200%] s.t. enterprise_pv(gʰ) == EV."""
    ev_target = market_cap + inp.net_debt
    lo, hi = -0.30, 2.00

    ev_hi = enterprise_pv(hi, inp)
    if ev_hi < ev_target:
        return ReverseDcfResult(hi, False, ev_target, ev_hi, 0,
                                inp.wacc, inp.terminal_g)
    ev_lo = enterprise_pv(lo, inp)
    if ev_lo > ev_target:
        return ReverseDcfResult(lo, False, ev_target, ev_lo, 0,
                                inp.wacc, inp.terminal_g)

    itr = 0
    for itr in range(150):
        mid = (lo + hi) / 2
        ev = enterprise_pv(mid, inp)
        if ev < ev_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    g = (lo + hi) / 2
    return ReverseDcfResult(
        implied_growth=g, converged=True, ev_target=ev_target,
        ev_model=enterprise_pv(g, inp), iterations=itr + 1,
        wacc=inp.wacc, terminal_g=inp.terminal_g,
    )


# ---------------------------------------------------------------------------
# Monte-Carlo wrapper over FCF-base scenarios
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    point_estimate: float        # implied growth at primary FCF base
    scenario_growths: dict[str, float]   # label -> implied growth
    mean: float
    std: float
    range_low: float             # mean - 1σ
    range_high: float            # mean + 1σ


def monte_carlo_implied_growth(
    market_cap: float, base_inp: ReverseDcfInputs,
    fcf_scenarios: dict[str, float],
) -> MonteCarloResult:
    """Run reverse-DCF for several FCF base scenarios; return mean ± 1σ."""
    growths: dict[str, float] = {}
    for label, fcf in fcf_scenarios.items():
        if not fcf or fcf <= 0:
            continue
        inp = ReverseDcfInputs(
            fcf_base=fcf, net_debt=base_inp.net_debt, shares=base_inp.shares,
            wacc=base_inp.wacc, terminal_g=base_inp.terminal_g,
            high_growth_years=base_inp.high_growth_years, fade_years=base_inp.fade_years,
        )
        r = reverse_dcf(market_cap, inp)
        if r.converged:
            growths[label] = r.implied_growth
    if not growths:
        return MonteCarloResult(
            point_estimate=float("nan"), scenario_growths={},
            mean=float("nan"), std=float("nan"),
            range_low=float("nan"), range_high=float("nan"),
        )
    arr = np.array(list(growths.values()))
    mean, std = float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    primary = next(iter(growths.values()))
    return MonteCarloResult(
        point_estimate=primary, scenario_growths=growths,
        mean=mean, std=std,
        range_low=mean - std, range_high=mean + std,
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
    inp: IntrinsicInputs, current_price: float | None = None,
) -> IntrinsicResult:
    wacc = inp.wacc
    g = min(inp.terminal_g, wacc - 0.005)
    rev, rows = inp.revenue_base, []
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
            "Year": t, "Revenue": rev_t, "EBIT": ebit, "NOPAT": nopat,
            "D&A": da, "CapEx": capex, "ΔNWC": dnwc,
            "FCFF": fcff, "DiscountFactor": df, "PV FCFF": fcff / df,
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
    upside = (
        (fv / current_price - 1)
        if current_price and np.isfinite(fv) and current_price > 0 else None
    )
    tv_pct = pv_tv / ev if ev > 0 else 0.0
    return IntrinsicResult(
        projections=proj, terminal_value=tv, pv_terminal=pv_tv,
        enterprise_value=ev, equity_value=equity,
        fair_value_per_share=fv, current_price=current_price,
        upside_pct=upside, tv_pct_of_ev=tv_pct, wacc_used=wacc,
    )


# ---------------------------------------------------------------------------
# sensitivity grid (heatmap)
# ---------------------------------------------------------------------------

def sensitivity_grid(
    inp: IntrinsicInputs, current_price: float | None = None,
    wacc_steps: tuple[float, ...] = (-0.02, -0.01, 0.0, +0.01, +0.02),
    g_steps: tuple[float, ...] = (-0.01, -0.005, 0.0, +0.005, +0.01),
) -> pd.DataFrame:
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
# fair-value vs growth curve
# ---------------------------------------------------------------------------

def fair_value_curve(
    inp: ReverseDcfInputs, shares: float,
    current_price: float | None = None, n_points: int = 50,
) -> pd.DataFrame:
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
    "ReverseDcfResult", "IntrinsicResult", "MonteCarloResult",
    "enterprise_pv", "reverse_dcf", "intrinsic_dcf",
    "monte_carlo_implied_growth",
    "sensitivity_grid", "fair_value_curve",
]
