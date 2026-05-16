"""Forward-DCF: compute intrinsic equity value from user-supplied assumptions.

FCFF model:
    FCFF_t = Revenue_t × operating_margin × (1 − tax_rate) − Revenue_t × capex_pct
    Revenue_t = Revenue_{t-1} × (1 + revenue_growth)

Terminal value (Gordon Growth):
    TV = FCFF_N × (1 + terminal_growth) / (WACC − terminal_growth)

Equity value = PV(FCFFs) + PV(TV) − Net Debt
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class IntrinsicResult:
    intrinsic_equity: float          # EV − net debt
    intrinsic_per_share: float | None
    margin_of_safety: float | None   # (intrinsic − price) / price
    projections: pd.DataFrame        # columns: Revenue, FCFF, PV_FCFF
    enterprise_value: float
    pv_explicit: float
    pv_terminal: float
    tv_pct: float


def intrinsic_value(
    base_revenue: float,
    revenue_growth: float,
    operating_margin: float,
    tax_rate: float,
    capex_pct: float,
    wacc: float,
    terminal_growth: float,
    years: int,
    shares_out: float | None = None,
    current_price: float | None = None,
    net_debt: float = 0.0,
) -> IntrinsicResult:
    """Two-stage revenue-driven DCF returning equity intrinsic value."""
    # Guard: terminal growth must be below WACC
    term_g = min(terminal_growth, wacc - 0.005)

    rows = []
    rev = base_revenue
    pv_ex = 0.0
    for t in range(1, years + 1):
        rev = rev * (1.0 + revenue_growth)
        fcff = rev * operating_margin * (1.0 - tax_rate) - rev * capex_pct
        df = (1.0 + wacc) ** t
        pv = fcff / df
        pv_ex += pv
        rows.append({"Year": t, "Revenue": rev, "FCFF": fcff, "PV_FCFF": pv})

    proj = pd.DataFrame(rows).set_index("Year")

    # Terminal value on the last-period FCFF
    last_fcff = float(proj["FCFF"].iloc[-1])
    tv = last_fcff * (1.0 + term_g) / (wacc - term_g)
    pv_tv = tv / (1.0 + wacc) ** years

    ev = pv_ex + pv_tv
    equity = ev - net_debt

    per_share: float | None = None
    if shares_out and shares_out > 0 and np.isfinite(equity):
        per_share = equity / shares_out

    mos: float | None = None
    if per_share is not None and current_price and current_price > 0:
        mos = (per_share - current_price) / current_price

    tv_pct = pv_tv / ev if ev > 0 else 0.0

    return IntrinsicResult(
        intrinsic_equity=equity,
        intrinsic_per_share=per_share,
        margin_of_safety=mos,
        projections=proj,
        enterprise_value=ev,
        pv_explicit=pv_ex,
        pv_terminal=pv_tv,
        tv_pct=tv_pct,
    )


__all__ = ["IntrinsicResult", "intrinsic_value"]
