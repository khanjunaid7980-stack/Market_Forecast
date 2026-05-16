"""Bloomberg-Terminal-styled Plotly charts for the RvM tool.

Design principles:
  - Dark palette (#0b0f19 background, #141a2a card)
  - Monospace font throughout (matches terminal aesthetic)
  - hovermode set per chart type (unified only on time-series lines)
  - All annotations positioned to avoid overlap
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Palette ──────────────────────────────────────────────────────────────────
BG     = "#0b0f19"
CARD   = "#141a2a"
BORDER = "#1f2937"
TXT    = "#e6edf3"
MUTED  = "#6b7280"
ACCENT = "#4f8cff"
GREEN  = "#22c55e"
RED    = "#ef4444"
AMBER  = "#f59e0b"
PURPLE = "#a78bfa"

_FONT = dict(color=TXT, family="ui-monospace, SFMono-Regular, Menlo, monospace", size=12)


def _base_layout(fig: go.Figure, title: str, hovermode: str = "closest") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TXT)),
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=_FONT,
        hovermode=hovermode,
        legend=dict(
            orientation="h", y=-0.22,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=55, r=25, t=65, b=55),
    )
    fig.update_xaxes(
        gridcolor=BORDER, zerolinecolor=BORDER,
        tickfont=dict(size=11), title_font=dict(size=12),
    )
    fig.update_yaxes(
        gridcolor=BORDER, zerolinecolor=BORDER,
        tickfont=dict(size=11), title_font=dict(size=12),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: Rationality Gap — historical growth bars + implied line
# ─────────────────────────────────────────────────────────────────────────────
def rationality_gap_chart(
    implied: float,
    revenue_series: pd.Series,       # indexed by fiscal year (int)
    hist_mean: float,
    hist_std: float,
) -> go.Figure:
    """Bar chart of each year's YoY growth, with the market-implied CAGR overlaid."""
    growths = revenue_series.pct_change().dropna()

    fig = go.Figure()

    if not growths.empty:
        x_labels = [str(int(y)) for y in growths.index]
        y_vals = growths.values * 100

        # Colour each bar by distance from mean
        bar_colors = []
        for g_val in y_vals:
            if not (np.isfinite(hist_mean) and np.isfinite(hist_std) and hist_std > 0):
                bar_colors.append(ACCENT)
            else:
                z = abs((g_val / 100 - hist_mean) / hist_std)
                bar_colors.append(GREEN if z < 1 else AMBER if z < 2 else RED)

        fig.add_trace(go.Bar(
            x=x_labels, y=y_vals,
            name="Historical YoY Growth",
            marker_color=bar_colors,
            hovertemplate="FY %{x}<br>YoY: %{y:.1f}%<extra></extra>",
        ))

    # ±2σ rational band
    if np.isfinite(hist_mean) and np.isfinite(hist_std) and hist_std > 0:
        fig.add_hrect(
            y0=(hist_mean - 2 * hist_std) * 100,
            y1=(hist_mean + 2 * hist_std) * 100,
            fillcolor=GREEN, opacity=0.08, line_width=0,
        )
        fig.add_hline(
            y=hist_mean * 100,
            line_dash="dot", line_color=GREEN, line_width=1.5,
            annotation_text=f"Hist. avg {hist_mean*100:.1f}%",
            annotation_position="bottom left",
            annotation_font_color=GREEN,
            annotation_font_size=11,
        )

    # Implied growth — the KEY number, drawn last so it's on top
    fig.add_hline(
        y=implied * 100,
        line_color=RED, line_width=2.5,
        annotation_text=f"  Mkt-Implied CAGR: {implied*100:.2f}%",
        annotation_position="top left",
        annotation_font_color=RED,
        annotation_font_size=12,
    )

    _base_layout(fig, "Revenue Growth History vs. Market-Implied CAGR", hovermode="x unified")
    fig.update_yaxes(title_text="YoY Revenue Growth (%)")
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: Revenue Projection — historical actuals vs implied trajectory
# ─────────────────────────────────────────────────────────────────────────────
def revenue_projection_chart(
    revenue_hist: pd.Series,
    implied_g: float,
    years: int,
) -> go.Figure:
    """Grouped bar: historical revenue (solid blue) + implied projection (red)."""
    fig = go.Figure()

    # Historical
    x_hist = [str(int(y)) for y in revenue_hist.index]
    y_hist = revenue_hist.values / 1e9
    fig.add_trace(go.Bar(
        x=x_hist, y=y_hist,
        name="Historical Revenue",
        marker_color=ACCENT,
        hovertemplate="FY %{x}<br>Revenue: $%{y:.1f}B<extra></extra>",
    ))

    # Implied projection
    base = float(revenue_hist.iloc[-1])
    last_yr = int(revenue_hist.index[-1])
    x_proj = [str(last_yr + t) for t in range(1, years + 1)]
    y_proj = [base * (1.0 + implied_g) ** t / 1e9 for t in range(1, years + 1)]
    fig.add_trace(go.Bar(
        x=x_proj, y=y_proj,
        name=f"Implied @ {implied_g*100:.1f}% p.a.",
        marker_color=RED, opacity=0.75,
        hovertemplate="FY %{x}<br>Projected: $%{y:.1f}B<extra></extra>",
    ))

    # Connector line between last actual and first projection
    fig.add_trace(go.Scatter(
        x=[x_hist[-1], x_proj[0]],
        y=[y_hist[-1], y_proj[0]],
        mode="lines",
        line=dict(color=RED, dash="dot", width=1),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Annotation: final projected revenue
    fig.add_annotation(
        x=x_proj[-1], y=y_proj[-1],
        text=f"${y_proj[-1]:.0f}B",
        showarrow=True, arrowhead=2, arrowcolor=RED,
        arrowwidth=1.5, ax=0, ay=-35,
        font=dict(color=RED, size=11),
    )

    _base_layout(
        fig,
        f"Revenue: Actual vs. Market-Implied Projection ({years}-yr horizon)",
        hovermode="x unified",
    )
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(title_text="Revenue ($B)")
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3: Price vs. Intrinsic (manual mode)
# ─────────────────────────────────────────────────────────────────────────────
def price_vs_intrinsic(
    price: float,
    intrinsic: float | None,
    implied_g: float | None = None,
    user_g: float | None = None,
) -> go.Figure:
    """Side-by-side bars: market price vs. user DCF intrinsic value."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Market Price"], y=[price],
        name="Market Price",
        marker_color=ACCENT,
        width=0.4,
        hovertemplate="Market Price: $%{y:,.2f}<extra></extra>",
    ))

    if intrinsic is not None:
        c = GREEN if intrinsic >= price else RED
        fig.add_trace(go.Bar(
            x=["Your DCF Value"], y=[intrinsic],
            name="Your DCF Value",
            marker_color=c,
            width=0.4,
            hovertemplate="Intrinsic Value: $%{y:,.2f}<extra></extra>",
        ))

        gap_pct = (intrinsic - price) / price * 100
        label = f"{'Undervalued' if intrinsic > price else 'Overvalued'}: {gap_pct:+.1f}%"
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=1.08,
            text=label, showarrow=False,
            font=dict(size=13, color=GREEN if intrinsic > price else RED),
        )

    # Show which growth rate produced which value
    if implied_g is not None and user_g is not None:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=-0.18,
            text=(
                f"Market implies {implied_g*100:.1f}% CAGR  |  "
                f"Your model uses {user_g*100:.1f}% CAGR"
            ),
            showarrow=False,
            font=dict(size=10, color=MUTED),
        )

    _base_layout(fig, "Market Price vs. Your Intrinsic Value (per share)")
    fig.update_yaxes(title_text="$ per share")
    fig.update_layout(barmode="group", bargap=0.3)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4: Growth Distribution — histogram of historical YoY with implied line
# ─────────────────────────────────────────────────────────────────────────────
def historical_growth_distribution(
    growths: list[float],
    implied: float | None,
) -> go.Figure:
    """Histogram of YoY revenue growth rates, with implied CAGR marked."""
    fig = go.Figure()

    if growths:
        fig.add_trace(go.Histogram(
            x=[g * 100 for g in growths],
            nbinsx=max(6, len(growths) // 2),
            marker_color=ACCENT,
            marker_line=dict(width=1, color=BORDER),
            name="Annual YoY Revenue Growth",
            hovertemplate="Growth: %{x:.1f}%<br>Count: %{y}<extra></extra>",
        ))

    if implied is not None and np.isfinite(implied):
        fig.add_vline(
            x=implied * 100,
            line_color=RED, line_width=2.5,
            annotation_text=f"Implied {implied*100:.1f}%",
            annotation_position="top right",
            annotation_font_color=RED,
            annotation_font_size=11,
        )

    _base_layout(fig, "Distribution of Historical YoY Revenue Growth")
    fig.update_xaxes(title_text="Year-over-Year Revenue Growth (%)")
    fig.update_yaxes(title_text="Frequency (years)")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5: Sensitivity heatmap — implied growth across WACC × terminal g grid
# ─────────────────────────────────────────────────────────────────────────────
def implied_growth_sensitivity(
    wacc_range: np.ndarray,
    term_g_range: np.ndarray,
    market_cap: float,
    base_rev: float,
    fcf_margin: float,
    years: int,
    net_debt: float,
    solver: Callable,
) -> go.Figure:
    """Heatmap of implied growth rate across a WACC × terminal-g grid."""
    grid = np.full((len(wacc_range), len(term_g_range)), np.nan)
    for i, w in enumerate(wacc_range):
        for j, tg in enumerate(term_g_range):
            res = solver(market_cap, base_rev, fcf_margin, w, tg, years, net_debt)
            if res.converged and res.implied_growth is not None:
                grid[i, j] = round(res.implied_growth * 100, 2)

    fig = go.Figure(go.Heatmap(
        z=grid,
        x=[f"{g*100:.1f}%" for g in term_g_range],
        y=[f"{w*100:.1f}%" for w in wacc_range],
        colorscale="RdYlGn_r",
        colorbar=dict(
            title=dict(text="Implied g (%)", font=dict(size=11, color=TXT)),
            tickfont=dict(color=TXT),
        ),
        text=np.where(np.isnan(grid), "N/A", np.char.add(np.round(grid, 1).astype(str), "%")),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="WACC: %{y}<br>Term. g: %{x}<br>Implied g: %{z:.2f}%<extra></extra>",
    ))

    _base_layout(fig, "Implied Revenue CAGR Sensitivity (WACC × Terminal Growth)")
    fig.update_xaxes(title_text="Terminal Growth Rate")
    fig.update_yaxes(title_text="WACC")
    return fig


__all__ = [
    "rationality_gap_chart",
    "revenue_projection_chart",
    "price_vs_intrinsic",
    "historical_growth_distribution",
    "implied_growth_sensitivity",
]
