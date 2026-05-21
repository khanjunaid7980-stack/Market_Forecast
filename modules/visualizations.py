"""Plotly visualisations for the RvM Forecast Terminal.

All charts share the same Bloomberg-dark base layout. Pure functions —
no Streamlit dependencies, easy to test in isolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .rationality import RationalityResult


# Palette
BG = "#0b0d10"
PANEL = "#13171c"
GRID = "#1f2630"
FG = "#d8dde3"
MUTED = "#7d8693"
AMBER = "#ffb000"
CYAN = "#27e0c5"
MAGENTA = "#ff5fa2"
GREEN = "#22c55e"
RED = "#ef4444"
BLUE = "#4f8cff"
PURPLE = "#a78bfa"

_FONT = dict(color=FG, family="JetBrains Mono, IBM Plex Mono, monospace", size=12)


def _layout(fig: go.Figure, title: str = "", height: int = 360, **kw) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=FG, size=13)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=_FONT,
        height=height,
        margin=dict(l=50, r=24, t=46, b=50),
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=GRID,
            font=dict(size=11, color=FG),
            orientation="h", y=-0.22, yanchor="top",
        ),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=11)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=11)),
        **kw,
    )
    return fig


# ===========================================================================
# 1. Rationality gauge
# ===========================================================================

def rationality_gauge(r: RationalityResult) -> go.Figure:
    z_clip = float(np.clip(r.z_score, -3, 5))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=z_clip,
        number=dict(suffix=" σ", font=dict(color=r.verdict_color, size=42)),
        gauge=dict(
            axis=dict(
                range=[-3, 5],
                tickvals=[-3, -1, 0, 1, 2, 3, 5],
                ticktext=["-3σ", "-1σ", "0", "+1σ", "+2σ", "+3σ", "+5σ"],
                tickfont=dict(size=10, color=MUTED),
            ),
            bar=dict(color=r.verdict_color, thickness=0.45),
            bgcolor=PANEL, borderwidth=1, bordercolor=GRID,
            steps=[
                dict(range=[-3, 1], color="rgba(34,197,94,0.18)"),
                dict(range=[1, 2], color="rgba(245,158,11,0.18)"),
                dict(range=[2, 3], color="rgba(239,68,68,0.18)"),
                dict(range=[3, 5], color="rgba(220,38,38,0.28)"),
            ],
            threshold=dict(
                line=dict(color=AMBER, width=3),
                thickness=0.85, value=z_clip,
            ),
        ),
        title=dict(
            text=f"Rationality — <b style='color:{r.verdict_color}'>{r.verdict}</b>",
            font=dict(size=13, color=FG),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=PANEL, font=_FONT,
        height=280, margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ===========================================================================
# 2. Implied growth vs historical distribution
# ===========================================================================

def growth_distribution(r: RationalityResult) -> go.Figure:
    fig = go.Figure()
    if not np.isfinite(r.historical_mean) or not np.isfinite(r.historical_std) or r.historical_std == 0:
        fig.add_annotation(
            text="Insufficient historical data",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color=MUTED, size=14),
        )
        return _layout(fig, "Implied Growth vs. Historical Distribution", height=320)

    mu, sig = r.historical_mean, r.historical_std
    x_lo = min(mu - 4 * sig, r.implied_growth - 0.05)
    x_hi = max(mu + 4 * sig, r.implied_growth + 0.05)
    xs = np.linspace(x_lo, x_hi, 240)
    ys = np.exp(-0.5 * ((xs - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    xs_pct = xs * 100

    def band(lo: float, hi: float, color: str, name: str) -> go.Scatter:
        mask = (xs >= lo) & (xs <= hi)
        return go.Scatter(
            x=np.concatenate([xs_pct[mask], xs_pct[mask][::-1]]),
            y=np.concatenate([ys[mask], np.zeros_like(ys[mask])]),
            fill="toself", fillcolor=color, line=dict(width=0),
            name=name, hoverinfo="skip",
        )

    fig.add_trace(band(mu - sig, mu + sig, "rgba(34,197,94,0.18)", "±1σ Rational"))
    fig.add_trace(band(mu + sig, mu + 2 * sig, "rgba(245,158,11,0.20)", "1–2σ Stretched"))
    fig.add_trace(band(mu + 2 * sig, x_hi, "rgba(239,68,68,0.18)", ">2σ Speculative"))
    fig.add_trace(go.Scatter(
        x=xs_pct, y=ys, mode="lines",
        line=dict(color=BLUE, width=2.2), name="Historical PDF",
    ))
    fig.add_vline(
        x=r.implied_growth * 100, line_color=AMBER, line_width=2.5,
        annotation_text=f"Implied {r.implied_growth*100:.1f}%",
        annotation_font_color=AMBER, annotation_position="top right",
    )
    if r.analyst_growth is not None:
        fig.add_vline(
            x=r.analyst_growth * 100, line_color=CYAN, line_dash="dash",
            annotation_text=f"Analyst {r.analyst_growth*100:.1f}%",
            annotation_font_color=CYAN, annotation_position="top left",
        )
    fig.add_vline(x=mu * 100, line_color=MUTED, line_dash="dot",
                  annotation_text=f"μ {mu*100:.1f}%",
                  annotation_font_color=MUTED, annotation_position="bottom left")

    fig = _layout(fig, "Implied Growth vs. 5Y Historical Distribution", height=340)
    fig.update_xaxes(title_text="Annual growth rate (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Density", showticklabels=False)
    return fig


# ===========================================================================
# 3. Fair value vs growth curve
# ===========================================================================

def fair_value_curve_chart(
    curve: pd.DataFrame, implied_growth: float,
    current_price: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["growth"] * 100, y=curve["fair_value"],
        mode="lines", line=dict(color=CYAN, width=2.5),
        name="DCF fair value", fill="tozeroy",
        fillcolor="rgba(39,224,197,0.06)",
    ))
    if current_price:
        fig.add_hline(
            y=current_price, line_color=AMBER, line_width=2, line_dash="dash",
            annotation_text=f"Market ${current_price:,.2f}",
            annotation_font_color=AMBER, annotation_position="top right",
        )
    fig.add_vline(
        x=implied_growth * 100, line_color=RED, line_width=2, line_dash="dot",
        annotation_text=f"Implied {implied_growth*100:.1f}%",
        annotation_font_color=RED, annotation_position="top right",
    )
    fig = _layout(fig, "Fair Value per Share vs. Assumed Growth Rate", height=400)
    fig.update_xaxes(title_text="Growth Rate (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Fair Value / share ($)", tickprefix="$")
    return fig


# ===========================================================================
# 4. Sensitivity heatmap
# ===========================================================================

def sensitivity_heatmap(
    grid: pd.DataFrame, current_price: float | None = None,
) -> go.Figure:
    z = grid.values.astype(float)
    finite = z[np.isfinite(z)]
    zmid = current_price if (current_price and finite.size and
                              finite.min() <= current_price <= finite.max()) else None
    text = np.where(np.isfinite(z),
                    np.vectorize(lambda v: f"${v:,.0f}")(z), "N/A")
    title = "Sensitivity: Fair Value / Share (WACC × Terminal g)"
    if current_price:
        title += f" — market ${current_price:,.2f}"

    fig = go.Figure(data=go.Heatmap(
        z=z, x=list(grid.columns), y=list(grid.index),
        colorscale="RdYlGn", zmid=zmid,
        colorbar=dict(title="$/share", tickfont=dict(size=10)),
        text=text, texttemplate="%{text}",
        textfont=dict(size=11, color=FG, family="JetBrains Mono, monospace"),
        hovertemplate="WACC=%{y} · g=%{x}<br>Fair value=%{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=FG, size=13)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=_FONT,
        height=380, margin=dict(l=70, r=24, t=50, b=50),
        xaxis=dict(title="Terminal g", tickfont=dict(size=11)),
        yaxis=dict(title="WACC", tickfont=dict(size=11), autorange="reversed"),
    )
    return fig


# ===========================================================================
# 5. Historical growth bars vs implied / analyst lines
# ===========================================================================

def historical_growth_bars(
    rev_growth: pd.Series, eps_growth: pd.Series,
    implied_growth: float | None = None,
    analyst_growth: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    if not rev_growth.empty:
        fig.add_trace(go.Bar(
            x=rev_growth.index.astype(str),
            y=(rev_growth.values * 100),
            name="Revenue YoY (%)", marker_color=CYAN, opacity=0.85,
        ))
    if not eps_growth.empty:
        fig.add_trace(go.Bar(
            x=eps_growth.index.astype(str),
            y=(eps_growth.values * 100),
            name="NetIncome YoY (%)", marker_color=MAGENTA, opacity=0.85,
        ))
    if implied_growth is not None:
        fig.add_hline(
            y=implied_growth * 100, line_color=AMBER, line_width=2.5,
            annotation_text=f"Implied {implied_growth*100:.1f}%",
            annotation_font_color=AMBER, annotation_position="top right",
        )
    if analyst_growth is not None:
        fig.add_hline(
            y=analyst_growth * 100, line_color=BLUE, line_dash="dash",
            annotation_text=f"Analyst {analyst_growth*100:.1f}%",
            annotation_font_color=BLUE, annotation_position="bottom right",
        )
    fig.add_hline(y=0, line_color=GRID, line_width=1)
    fig = _layout(fig, "Historical Growth (YoY) vs. Market Expectation", height=360)
    fig.update_xaxes(title_text="Fiscal year")
    fig.update_yaxes(title_text="YoY growth (%)", ticksuffix="%")
    fig.update_layout(barmode="group")
    return fig


# ===========================================================================
# 6. DCF projection chart
# ===========================================================================

def dcf_projection_chart(proj: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=proj.index.astype(str), y=proj["FCFF"] / 1e9,
        name="FCFF ($B)", marker_color=BLUE, opacity=0.78,
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=proj.index.astype(str), y=proj["PV FCFF"] / 1e9,
        name="PV of FCFF ($B)", marker_color=PURPLE, opacity=0.6,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=proj.index.astype(str), y=proj["Revenue"] / 1e9,
        name="Revenue ($B)", mode="lines+markers",
        line=dict(color=AMBER, width=2.5), marker=dict(size=6),
    ), secondary_y=True)
    fig.update_layout(
        title=dict(text="DCF Projections — explicit period", font=dict(color=FG, size=13)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=_FONT,
        height=380, margin=dict(l=50, r=50, t=46, b=58),
        hovermode="x unified", barmode="group",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.24),
        xaxis=dict(gridcolor=GRID, title="Year"),
    )
    fig.update_yaxes(title_text="Cash flow ($B)", gridcolor=GRID, secondary_y=False)
    fig.update_yaxes(title_text="Revenue ($B)", secondary_y=True, gridcolor=GRID)
    return fig


# ===========================================================================
# 7. Price history with fair-value & MoS bands
# ===========================================================================

def price_history_chart(
    hist: pd.DataFrame, fair_value: float | None = None,
    mos_low: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["Date"], y=hist["Close"], mode="lines",
            line=dict(color=BLUE, width=1.8), name="Close",
            fill="tozeroy", fillcolor="rgba(79,140,255,0.08)",
        ))
    if fair_value and np.isfinite(fair_value):
        fig.add_hline(
            y=fair_value, line_color=GREEN, line_width=2,
            annotation_text=f"DCF fair ${fair_value:,.2f}",
            annotation_font_color=GREEN,
        )
    if mos_low and np.isfinite(mos_low):
        fig.add_hline(
            y=mos_low, line_color=CYAN, line_dash="dot", line_width=1.5,
            annotation_text=f"15% MoS ${mos_low:,.2f}",
            annotation_font_color=CYAN, annotation_position="bottom right",
        )
    fig = _layout(fig, "Price History vs. DCF Fair Value", height=380)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price ($)", tickprefix="$")
    return fig


# ===========================================================================
# 8. Revenue & margin trends
# ===========================================================================

def revenue_and_margins(fin: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "Revenue" in fin.index:
        rev = fin.loc["Revenue"].astype(float)
        fig.add_trace(go.Bar(
            x=rev.index.astype(str), y=rev.values / 1e9,
            name="Revenue ($B)", marker_color=BLUE, opacity=0.75,
        ), secondary_y=False)
    if "GrossProfit" in fin.index and "Revenue" in fin.index:
        gm = (fin.loc["GrossProfit"].astype(float)
              / fin.loc["Revenue"].astype(float).replace(0, np.nan)) * 100
        fig.add_trace(go.Scatter(
            x=gm.index.astype(str), y=gm.values, name="Gross margin (%)",
            mode="lines+markers", line=dict(color=GREEN, width=2.2),
        ), secondary_y=True)
    if "OperatingIncome" in fin.index and "Revenue" in fin.index:
        om = (fin.loc["OperatingIncome"].astype(float)
              / fin.loc["Revenue"].astype(float).replace(0, np.nan)) * 100
        fig.add_trace(go.Scatter(
            x=om.index.astype(str), y=om.values, name="Operating margin (%)",
            mode="lines+markers", line=dict(color=AMBER, width=2.5),
        ), secondary_y=True)
    fig.update_layout(
        title=dict(text="Revenue & Profit Margins", font=dict(color=FG, size=13)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=_FONT,
        height=380, margin=dict(l=50, r=50, t=46, b=58),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.24),
        xaxis=dict(gridcolor=GRID),
    )
    fig.update_yaxes(title_text="Revenue ($B)", gridcolor=GRID, secondary_y=False)
    fig.update_yaxes(title_text="Margin (%)", ticksuffix="%",
                     gridcolor=GRID, secondary_y=True)
    return fig


# ===========================================================================
# 9. FCF history & FCF margin
# ===========================================================================

def fcf_quality_chart(
    fcf_history: pd.Series, rev_history: pd.Series,
) -> go.Figure:
    fig = go.Figure()
    if fcf_history.empty:
        fig.add_annotation(
            text="No FCF history available",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color=MUTED, size=14),
        )
        return _layout(fig, "FCF History & FCF Margin", height=340)
    yrs = [str(y) for y in fcf_history.index]
    fcf_b = (fcf_history.values / 1e9).astype(float)
    colors = [GREEN if v >= 0 else RED for v in fcf_b]
    fig.add_trace(go.Bar(
        x=yrs, y=fcf_b, name="FCF ($B)", marker_color=colors,
    ))
    if not rev_history.empty:
        common = fcf_history.index.intersection(rev_history.index)
        if len(common) > 0:
            margin = (fcf_history.loc[common] / rev_history.loc[common].replace(0, np.nan)) * 100
            fig.add_trace(go.Scatter(
                x=[str(y) for y in margin.index], y=margin.values,
                mode="lines+markers", name="FCF margin (%)",
                line=dict(color=CYAN, width=2.2), yaxis="y2",
            ))
    fig.update_layout(
        title=dict(text="FCF History & FCF Margin", font=dict(color=FG, size=13)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=_FONT,
        height=360, margin=dict(l=50, r=60, t=46, b=58),
        yaxis=dict(title="FCF ($B)", gridcolor=GRID),
        yaxis2=dict(
            title="FCF Margin (%)", ticksuffix="%", overlaying="y",
            side="right", gridcolor=GRID,
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.24),
        xaxis=dict(gridcolor=GRID, title="Fiscal year"),
    )
    return fig


# ===========================================================================
# 10. Valuation waterfall (DCF bridge)
# ===========================================================================

def valuation_waterfall(
    pv_explicit: float, pv_terminal: float, net_debt: float,
    market_cap: float | None = None,
) -> go.Figure:
    labels = ["Explicit PV", "Terminal PV", "− Net Debt", "= Equity Value"]
    values = [pv_explicit / 1e9, pv_terminal / 1e9,
              -net_debt / 1e9, 0]
    measures = ["relative", "relative", "relative", "total"]
    fig = go.Figure(go.Waterfall(
        x=labels, measure=measures, y=values,
        text=[f"${v:,.1f}B" for v in values[:-1]] + [f"${(pv_explicit+pv_terminal-net_debt)/1e9:,.1f}B"],
        textposition="outside",
        connector=dict(line=dict(color=GRID, dash="dot")),
        increasing=dict(marker=dict(color=GREEN)),
        decreasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=AMBER)),
    ))
    if market_cap:
        fig.add_hline(
            y=market_cap / 1e9, line_color=CYAN, line_dash="dash", line_width=1.5,
            annotation_text=f"Market cap ${market_cap/1e9:,.1f}B",
            annotation_font_color=CYAN,
        )
    fig = _layout(fig, "DCF Valuation Bridge", height=380)
    fig.update_yaxes(title_text="$ Billions", gridcolor=GRID)
    return fig


# ===========================================================================
# 11. Implied vs comparables bar
# ===========================================================================

def growth_comparison_bar(r: RationalityResult) -> go.Figure:
    labels, values, colors = [], [], []
    if np.isfinite(r.historical_min):
        labels.append("5Y min")
        values.append(r.historical_min * 100); colors.append(MUTED)
    if np.isfinite(r.historical_mean):
        labels.append("5Y mean")
        values.append(r.historical_mean * 100); colors.append(BLUE)
    if np.isfinite(r.historical_max):
        labels.append("5Y max")
        values.append(r.historical_max * 100); colors.append(CYAN)
    if r.analyst_growth is not None:
        labels.append("Analyst 5Y")
        values.append(r.analyst_growth * 100); colors.append(PURPLE)
    labels.append("Market-implied")
    values.append(r.implied_growth * 100); colors.append(r.verdict_color)

    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:+.1f}%" for v in values],
        textposition="outside",
    ))
    fig = _layout(fig, "Growth Rate Comparison — Implied vs. History vs. Analyst", height=320)
    fig.update_yaxes(title_text="Annual growth (%)", ticksuffix="%")
    fig.update_xaxes(title_text="")
    return fig


# ===========================================================================
# 12. ROIC vs WACC
# ===========================================================================

def roic_vs_wacc(fin: pd.DataFrame, wacc: float) -> go.Figure:
    fig = go.Figure()
    if "OperatingIncome" not in fin.index:
        fig.add_annotation(
            text="OperatingIncome unavailable",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color=MUTED, size=14),
        )
        return _layout(fig, "ROIC vs. WACC — Value Creation", height=340)
    ebit = fin.loc["OperatingIncome"].astype(float)
    debt = pd.Series(0.0, index=ebit.index)
    for k in ("LongTermDebt", "ShortTermDebt"):
        if k in fin.index:
            debt = debt.add(fin.loc[k].astype(float).fillna(0), fill_value=0)
    equity = (fin.loc["TotalEquity"].astype(float).fillna(0)
              if "TotalEquity" in fin.index else pd.Series(0.0, index=ebit.index))
    cash = (fin.loc["CashAndEquivalents"].astype(float).fillna(0)
            if "CashAndEquivalents" in fin.index else pd.Series(0.0, index=ebit.index))
    invested = (equity + debt - cash).replace(0, np.nan)
    tax = 0.21
    if "IncomeTaxExpense" in fin.index and "PreTaxIncome" in fin.index:
        ratios = (fin.loc["IncomeTaxExpense"].astype(float)
                  / fin.loc["PreTaxIncome"].astype(float)).replace([np.inf, -np.inf], np.nan).dropna()
        ratios = ratios[(ratios > 0) & (ratios < 0.5)]
        if not ratios.empty:
            tax = float(ratios.tail(3).mean())
    nopat = ebit * (1 - tax)
    roic = (nopat / invested) * 100
    roic = roic.replace([np.inf, -np.inf], np.nan).dropna()
    if not roic.empty:
        colors = [GREEN if v >= wacc * 100 else RED for v in roic.values]
        fig.add_trace(go.Bar(
            x=roic.index.astype(str), y=roic.values,
            name="ROIC (%)", marker_color=colors,
        ))
        fig.add_hline(
            y=wacc * 100, line_dash="dash", line_color=AMBER,
            annotation_text=f"WACC = {wacc*100:.2f}%",
            annotation_font_color=AMBER, annotation_position="top left",
        )
    fig = _layout(fig, "ROIC vs. WACC — Economic Value Creation", height=340)
    fig.update_yaxes(title_text="ROIC (%)", ticksuffix="%")
    return fig


__all__ = [
    "rationality_gauge", "growth_distribution", "fair_value_curve_chart",
    "sensitivity_heatmap", "historical_growth_bars", "dcf_projection_chart",
    "price_history_chart", "revenue_and_margins", "fcf_quality_chart",
    "valuation_waterfall", "growth_comparison_bar", "roic_vs_wacc",
]
