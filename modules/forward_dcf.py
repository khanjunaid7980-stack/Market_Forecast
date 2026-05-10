"""Forward DCF — Manual Sensitivity Mode.

User defines forward assumptions (revenue growth, operating margin, tax,
CapEx %, WACC, terminal g) and the engine computes intrinsic equity value,
per-share value, and margin of safety vs. the current market price.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class IntrinsicResult:
    intrinsic_equity: float
    intrinsic_per_share: float | None
    margin_of_safety: float | None
    projections: pd.DataFrame


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
    """FCFF = Rev × OpMargin × (1−tax) − Rev × capex_pct."""
    fcf_margin = operating_margin * (1.0 - tax_rate) - capex_pct
    rows: list[dict] = []
    pv_total = 0.0
    for t in range(1, years + 1):
        rev_t = base_revenue * (1.0 + revenue_growth) ** t
        fcf_t = rev_t * fcf_margin
        pv = fcf_t / (1.0 + wacc) ** t
        pv_total += pv
        rows.append({"Year": t, "Revenue": rev_t, "FCFF": fcf_t, "PV_FCFF": pv})

    rev_N = base_revenue * (1.0 + revenue_growth) ** years
    fcf_term = rev_N * fcf_margin * (1.0 + terminal_growth)
    tv = fcf_term / (wacc - terminal_growth) if wacc > terminal_growth else 0.0
    pv_tv = tv / (1.0 + wacc) ** years

    enterprise_value = pv_total + pv_tv
    equity = enterprise_value - net_debt
    per_share = (equity / shares_out) if (shares_out and shares_out > 0) else None
    mos = ((per_share - current_price) / current_price) if (per_share and current_price) else None

    df = pd.DataFrame(rows).set_index("Year")
    return IntrinsicResult(
        intrinsic_equity=equity,
        intrinsic_per_share=per_share,
        margin_of_safety=mos,
        projections=df,
    )


__all__ = ["IntrinsicResult", "intrinsic_value"]
