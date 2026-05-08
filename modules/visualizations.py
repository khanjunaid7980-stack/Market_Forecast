"""Plotly visualisations for the RvM Forecast Terminal.

All charts share a Bloomberg-dark base layout so they look
consistent across tabs without per-chart boilerplate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .rationality import RationalityResult


# ---------------------------------------------------------------------------
# shared theme
# ---------------------------------------------------------------------------

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


def _base(title: str = "", height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(color=FG, size=13)),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=FG, family="JetBrains Mono, IBM Plex Mono, monospace"),
        height=height,
        margin=dict(l=50, r=20, t=44, b=44),
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=GRID,
            font=dict(size=11), orientation="h", y=-0.22,
        ),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=11)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=11)),
    )
    return fig


# ---------------------------------------------------------------------------
# 1. Rationality gauge
# ---------------------------------------------------------------------------

def rationality_gauge(result: RationalityResult) -> go.Figure:
    z = min(max(result.z_score, -3), 4)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=z,
        number=dict(suffix=" σ", font=dict(color=result.verdict_color, size=36)),
        delta=dict(
            reference=0,
            increasing=dict(color=RED),
            decreasing=dict(color=GREEN),
        ),
        gauge=dict(
            axis=dict(
                range=[-3, 4],
                tickvals=[-3, -2, -1, 0, 1, 2, 3, 4],
                ticktext=["-3σ", "-2σ", "-1σ", "0", "+1σ", "+2σ", "+3σ", "+4σ"],
                tickfont=dict(size=10, color=MUTED),
            ),
            bar=dict(color=result.verdict_color, thickness=0.5),
            bgcolor=PANEL,
            borderwidth=1,
            bordercolor=GRID,
            steps=[
                dict(range=[-3, 1], color="#0e2e1a"),
                dict(range=[1, 2], color="#2e2200"),
                dict(range=[2, 4], color="#2e0e0e"),
            ],
            threshold=dict(
                line=dict(color=AMBER, width=3),
                thickness=0.85,
                value=z,
            ),
        ),
        title=dict(
            text=f"Rationality z-score — <b>{result.verdict}</b>",
            font=dict(size=13, color=FG),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=PANEL, font=dict(color=FG),
        height=280, margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Implied growth vs historical distribution
# ---------------------------------------------------------------------------

def growth_distribution(result: RationalityResult) -> go.Figure:
    fig = _base("Implied Growth vs. 5Y Historical Distribution", height=300)

    if not np.isfinite(result.historical_mean):
        fig.add_annotation(text="Insufficient historical data",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color=MUTED, size=14))
        return fig

    mu, sig = result.historical_mean, result.historical_std
    xs = np.linspace(mu - 4 * sig, max(mu + 4 * sig, result.implied_growth + 0.05), 200)
    ys = np.exp(-0.5 * ((xs - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

    # shaded bands
    def band(lo, hi, color, name):
        mask = (xs >= lo) & (xs <= hi)
        return go.Scatter(
            x=np.concatenate([xs[mask], xs[mask][::-1]]),
            y=np.concatenate([ys[mask], np.zeros_like(ys[mask])]),
            fill="toself", fillcolor=color,
            line=dict(width=0), name=name, showlegend=True,
        )

    fig.add_trace(band(mu - sig, mu + sig, "rgba(34,197,94,0.15)", "±1σ (Rational)"))
    fig.add_trace(band(mu + sig, mu + 2 * sig, "rgba(245,158,11,0.15)", "1–2σ (Stretched)"))
    fig.add_trace(band(mu + 2 * sig, xs[-1], "rgba(239,68,68,0.12)", ">2σ (Speculative)"))

    fig.add_trace(go.Scatter(
        x=xs * 100, y=ys, mode="lines",
        line=dict(color=BLUE, width=2), name="Historical distribution", showlegend=False,
    ))

    # Fix: x axis should be in percent
    fig2 = _base("Implied Growth vs. 5Y Historical Distribution", height=300)
    xs_pct = xs * 100
    fig2.add_trace(band(mu * 100 - sig * 100, mu * 100 + sig * 100, "rgba(34,197,94,0.15)", "±1σ Rational"))
    fig2.add_trace(band(mu * 100 + sig * 100, mu * 100 + 2 * sig * 100, "rgba(245,158,11,0.15)", "1–2σ Stretched"))
    fig2.add_trace(band(mu * 100 + 2 * sig * 100, xs_pct[-1], "rgba(239,68,68,0.12)", ">2σ Speculative"))
    fig2.add_trace(go.Scatter(
        x=xs_pct, y=ys, mode="lines",
        line=dict(color=BLUE, width=2), name="5Y empirical distribution",
    ))
    fig2.add_vline(x=result.implied_growth * 100, line_color=AMBER, line_width=2.5,
                   annotation_text=f"Implied {result.implied_growth*100:.1f}%",
                   annotation_font_color=AMBER, annotation_position="top right")
    if result.analyst_growth is not None:
        fig2.add_vline(x=result.analyst_growth * 100, line_color=CYAN, line_dash="dash",
                       annotation_text=f"Analyst {result.analyst_growth*100:.1f}%",
                       annotation_font_color=CYAN, annotation_position="top left")
    fig2.update_xaxes(title_text="Growth Rate (%)", ticksuffix="%")
    fig2.update_yaxes(title_text="Density", showticklabels=False)
    return fig2


# ---------------------------------------------------------------------------
# 3. Fair-value vs growth curve
# ---------------------------------------------------------------------------

def fair_value_curve_chart(
    curve_df: pd.DataFrame,
    implied_growth: float,
    current_price: float | None = None,
) -> go.Figure:
    fig = _base("Fair Value per Share vs. Assumed Growth Rate", height=380)
    fig.add_trace(go.Scatter(
        x=curve_df["growth"] * 100,
        y=curve_df["fair_value"],
        mode="lines",
        line=dict(color=CYAN, width=2.5),
        name="DCF fair value",
    ))
    if current_price:
        fig.add_hline(y=current_price, line_color=AMBER, line_width=1.8,
                      line_dash="dash",
                      annotation_text=f"Market ${current_price:,.2f}",
                      annotation_font_color=AMBER)
    fig.add_vline(x=implied_growth * 100, line_color=RED, line_width=1.5,
                  line_dash="dot",
                  annotation_text=f"Implied {implied_growth*100:.1f}%",
                  annotation_font_color=RED, annotation_position="top right")
    fig.update_xaxes(title_text="Growth Rate (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Fair Value / Share ($)", tickprefix="$")
    return fig


# ---------------------------------------------------------------------------
# 4. Sensitivity heatmap
# ---------------------------------------------------------------------------

def sensitivity_heatmap(grid: pd.DataFrame, current_price: float | None = None) -> go.Figure:
    z = grid.values.astype(float)
    text = np.where(np.isfinite(z), np.vectorize(lambda v: f"${v:,.1f}")(z), "N/A")
    title = "Sensitivity: Fair Value / Share (WACC × Terminal g)"
    if current_price:
        title += f" — market ${current_price:,.2f}"
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=list(grid.columns),
        y=list(grid.index),
        colorscale="RdYlGn",
        zmid=current_price,
        colorbar=dict(title="$/share", tickfont=dict(size=10)),
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="WACC=%{y} terminal_g=%{x}<br>Fair value=%{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=FG, size=13)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=FG, family="JetBrains Mono, IBM Plex Mono, monospace"),
        height=360, margin=dict(l=60, r=20, t=48, b=40),
        xaxis=dict(title="Terminal g", tickfont=dict(size=11)),
        yaxis=dict(title="WACC", tickfont=dict(size=11)),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Historical growth bar chart
# ---------------------------------------------------------------------------

def historical_growth_bars(
    rev_growth: list[float],
    eps_growth: list[float],
    implied_growth: float | None = None,
    analyst_growth: float | None = None,
) -> go.Figure:
    fig = _base("Historical Revenue & Earnings Growth (YoY)", height=340)
    years_rev = [f"Y-{len(rev_growth)-i}" for i in range(len(rev_growth))]
    years_eps = [f"Y-{len(eps_growth)-i}" for i in range(len(eps_growth))]
    if rev_growth:
        fig.add_trace(go.Bar(
            x=years_rev, y=[v * 100 for v in rev_growth],
            name="Revenue growth (%)", marker_color=CYAN, opacity=0.85,
        ))
    if eps_growth:
        fig.add_trace(go.Bar(
            x=years_eps, y=[v * 100 for v in eps_growth],
            name="EPS / NI growth (%)", marker_color=MAGENTA, opacity=0.85,
        ))
    if implied_growth is not None:
        fig.add_hline(y=implied_growth * 100, line_color=AMBER, line_width=2,
                      annotation_text=f"Implied {implied_growth*100:.1f}%",
                      annotation_font_color=AMBER)
    if analyst_growth is not None:
        fig.add_hline(y=analyst_growth * 100, line_color=BLUE, line_dash="dash",
                      annotation_text=f"Analyst {analyst_growth*100:.1f}%",
                      annotation_font_color=BLUE, annotation_position="bottom right")
    fig.add_hline(y=0, line_color=GRID, line_width=1)
    fig.update_xaxes(title_text="Historical Period")
    fig.update_yaxes(title_text="YoY Growth (%)", ticksuffix="%")
    fig.update_layout(barmode="group")
    return fig


# ---------------------------------------------------------------------------
# 6. DCF projection chart
# ---------------------------------------------------------------------------

def dcf_projection_chart(proj: pd.DataFrame, wacc: float) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=proj.index.astype(str),
        y=proj["FCFF"] / 1e9,
        name="FCFF ($B)", marker_color=BLUE, opacity=0.8,
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=proj.index.astype(str),
        y=proj["PV FCFF"] / 1e9,
        name="PV of FCFF ($B)", marker_color=CYAN, opacity=0.6,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=proj.index.astype(str),
        y=proj["Revenue"] / 1e9,
        name="Revenue ($B)", mode="lines+markers",
        line=dict(color=AMBER, width=2.5),
    ), secondary_y=True)
    fig.update_layout(
        title=dict(text="DCF Projections — Explicit Period", font=dict(color=FG, size=13)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=FG, family="JetBrains Mono, IBM Plex Mono, monospace"),
        height=380, margin=dict(l=50, r=50, t=48, b=60),
        hovermode="x unified", barmode="group",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.28),
        xaxis=dict(gridcolor=GRID, title="Year"),
    )
    fig.update_yaxes(title_text="Cash Flow ($B)", gridcolor=GRID, secondary_y=False)
    fig.update_yaxes(title_text="Revenue ($B)", secondary_y=True, gridcolor=GRID)
    return fig


# ---------------------------------------------------------------------------
# 7. Price history with fair-value band
# ---------------------------------------------------------------------------

def price_history_chart(
    hist: pd.DataFrame,
    fair_value: float | None = None,
    mos_low: float | None = None,
) -> go.Figure:
    fig = _base("Price History vs. DCF Fair Value", height=360)
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["Date"], y=hist["Close"],
            mode="lines", line=dict(color=BLUE, width=1.8),
            name="Close", fill="tozeroy",
            fillcolor="rgba(79,140,255,0.06)",
        ))
    if fair_value and np.isfinite(fair_value):
        fig.add_hline(y=fair_value, line_color=GREEN, line_width=2,
                      annotation_text=f"DCF Fair Value ${fair_value:,.2f}",
                      annotation_font_color=GREEN)
    if mos_low and np.isfinite(mos_low):
        fig.add_hline(y=mos_low, line_color=CYAN, line_dash="dot", line_width=1.5,
                      annotation_text=f"15% MoS ${mos_low:,.2f}",
                      annotation_font_color=CYAN, annotation_position="bottom right")
    fig.update_yaxes(title_text="Price ($)", tickprefix="$")
    return fig


# ---------------------------------------------------------------------------
# 8. Revenue & margin trends
# ---------------------------------------------------------------------------

def revenue_and_margins(
    rev_history: list[float],
    op_margin_history: list[float] | None = None,
    gross_margin_history: list[float] | None = None,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    n = len(rev_history)
    years = [f"Y-{n-1-i}" for i in range(n)]
    fig.add_trace(go.Bar(
        x=years, y=[v / 1e9 for v in rev_history],
        name="Revenue ($B)", marker_color=BLUE, opacity=0.75,
    ), secondary_y=False)
    if op_margin_history:
        fig.add_trace(go.Scatter(
            x=years[:len(op_margin_history)],
            y=[v * 100 for v in op_margin_history],
            name="Operating margin", mode="lines+markers",
            line=dict(color=AMBER, width=2.5),
        ), secondary_y=True)
    if gross_margin_history:
        fig.add_trace(go.Scatter(
            x=years[:len(gross_margin_history)],
            y=[v * 100 for v in gross_margin_history],
            name="Gross margin", mode="lines+markers",
            line=dict(color=GREEN, width=2),
        ), secondary_y=True)
    fig.update_layout(
        title=dict(text="Revenue & Profit Margins", font=dict(color=FG, size=13)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=FG, family="JetBrains Mono, IBM Plex Mono, monospace"),
        height=360, margin=dict(l=50, r=50, t=48, b=60),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.28),
        xaxis=dict(gridcolor=GRID),
    )
    fig.update_yaxes(title_text="Revenue ($B)", gridcolor=GRID, secondary_y=False)
    fig.update_yaxes(title_text="Margin (%)", ticksuffix="%", gridcolor=GRID, secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# 9. FCF quality chart
# ---------------------------------------------------------------------------

def fcf_quality_chart(fcf_history: list[float], rev_history: list[float]) -> go.Figure:
    fig = _base("FCF History & FCF Yield", height=320)
    n = min(len(fcf_history), len(rev_history))
    if n == 0:
        return fig
    years = [f"Y-{n-1-i}" for i in range(n)]
    fcf_b = [v / 1e9 for v in fcf_history[-n:]]
    colors = [GREEN if v >= 0 else RED for v in fcf_b]
    fig.add_trace(go.Bar(
        x=years, y=fcf_b,
        name="FCF ($B)", marker_color=colors,
    ))
    rev_b = rev_history[-n:]
    fcf_margin = [f / r * 100 if r > 0 else float("nan")
                  for f, r in zip(fcf_history[-n:], rev_b)]
    fig.add_trace(go.Scatter(
        x=years, y=fcf_margin,
        mode="lines+markers", name="FCF margin (%)",
        line=dict(color=CYAN, width=2), yaxis="y2",
    ))
    fig.update_layout(
        yaxis=dict(title="FCF ($B)", gridcolor=GRID),
        yaxis2=dict(
            title="FCF Margin (%)", ticksuffix="%",
            overlaying="y", side="right", gridcolor=GRID,
        ),
        barmode="relative",
    )
    return fig


__all__ = [
    "rationality_gauge",
    "growth_distribution",
    "fair_value_curve_chart",
    "sensitivity_heatmap",
    "historical_growth_bars",
    "dcf_projection_chart",
    "price_history_chart",
    "revenue_and_margins",
    "fcf_quality_chart",
]
