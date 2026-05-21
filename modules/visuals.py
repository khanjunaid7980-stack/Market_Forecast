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


# ─────────────────────────────────────────────────────────────────────────────
# Chart 6: Earnings — LM Sentiment Breakdown (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
def earnings_sentiment_chart(analysis: "EarningsAnalysis") -> go.Figure:  # type: ignore[name-defined]
    """Horizontal bar: LM category counts per 1 000 words."""
    categories  = ["Positive", "Negative", "Uncertainty", "Litigious"]
    per_1k      = [
        analysis.pos_per_1k,
        analysis.neg_per_1k,
        analysis.unc_per_1k,
        analysis.lm_litigious / max(analysis.word_count, 1) * 1000,
    ]
    colors = [GREEN, RED, AMBER, PURPLE]

    fig = go.Figure(go.Bar(
        x=per_1k,
        y=categories,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in per_1k],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f} per 1k words<extra></extra>",
    ))
    _base_layout(fig, "Loughran-McDonald Sentiment (per 1 000 words)")
    fig.update_xaxes(title_text="Frequency per 1 000 words")
    fig.update_yaxes(tickfont=dict(size=12))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 7: Earnings — Sentiment timeline across the call
# ─────────────────────────────────────────────────────────────────────────────
def earnings_tone_timeline(para_tones: list[float]) -> go.Figure:
    """Line chart: rolling tone score by 200-word segment."""
    if not para_tones:
        return go.Figure()

    x = list(range(1, len(para_tones) + 1))
    y = para_tones

    colors = [GREEN if v >= 0 else RED for v in y]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(color=colors, size=7, line=dict(color=BORDER, width=1)),
        name="Tone score",
        hovertemplate="Segment %{x}<br>Tone: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED, line_width=1)
    fig.add_hrect(y0=0, y1=1,  fillcolor=GREEN, opacity=0.04, line_width=0)
    fig.add_hrect(y0=-1, y1=0, fillcolor=RED,   opacity=0.04, line_width=0)

    _base_layout(fig, "Sentiment Flow — 200-Word Segments", hovermode="x unified")
    fig.update_yaxes(title_text="Tone Score (LM)", range=[-1.05, 1.05])
    fig.update_xaxes(title_text="Segment (≈ 200 words each)")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 8: Earnings — Topic coverage
# ─────────────────────────────────────────────────────────────────────────────
def earnings_topic_chart(topic_hits: dict[str, int]) -> go.Figure:
    """Horizontal bar: topic mention frequency."""
    items = sorted(topic_hits.items(), key=lambda kv: kv[1])
    topics = [k for k, _ in items]
    counts = [v for _, v in items]

    fig = go.Figure(go.Bar(
        x=counts, y=topics,
        orientation="h",
        marker=dict(
            color=counts,
            colorscale=[[0, MUTED], [0.5, ACCENT], [1, GREEN]],
            showscale=False,
        ),
        text=counts,
        textposition="outside",
        hovertemplate="%{y}: %{x} mentions<extra></extra>",
    ))
    _base_layout(fig, "Key Topic Coverage (keyword mentions)")
    fig.update_xaxes(title_text="Mention count")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 9: Monte Carlo — Intrinsic value distribution
# ─────────────────────────────────────────────────────────────────────────────
def mc_intrinsic_histogram(iv_per_share: np.ndarray, price: float | None) -> go.Figure:
    """Histogram of simulated intrinsic values with market price overlay."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=iv_per_share,
        nbinsx=60,
        marker_color=ACCENT,
        marker_line=dict(width=0.5, color=BORDER),
        opacity=0.85,
        name="Simulated IV",
        hovertemplate="IV: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))
    if price is not None:
        fig.add_vline(
            x=price, line_color=RED, line_width=2.5,
            annotation_text=f"  Market ${price:,.2f}",
            annotation_position="top right",
            annotation_font_color=RED, annotation_font_size=11,
        )
        # Shade undervalued region
        x_max = float(np.percentile(iv_per_share, 99))
        if x_max > price:
            fig.add_vrect(
                x0=price, x1=x_max,
                fillcolor=GREEN, opacity=0.07, line_width=0,
            )
    _base_layout(fig, "Monte Carlo: Simulated Intrinsic Value per Share")
    fig.update_xaxes(title_text="Intrinsic Value ($ per share)")
    fig.update_yaxes(title_text="Frequency")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 10: Monte Carlo — Implied CAGR distribution
# ─────────────────────────────────────────────────────────────────────────────
def mc_cagr_histogram(cagr_arr: np.ndarray, base_implied: float | None) -> go.Figure:
    """Histogram of implied CAGRs across WACC × FCF-margin draws."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=cagr_arr * 100,
        nbinsx=50,
        marker_color=PURPLE,
        marker_line=dict(width=0.5, color=BORDER),
        opacity=0.85,
        name="Implied CAGR",
        hovertemplate="CAGR: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    ))
    if base_implied is not None and np.isfinite(base_implied):
        fig.add_vline(
            x=base_implied * 100, line_color=RED, line_width=2,
            annotation_text=f"  Base {base_implied*100:.1f}%",
            annotation_position="top right",
            annotation_font_color=RED,
        )
    _base_layout(fig, "Monte Carlo: Implied Revenue CAGR Sensitivity")
    fig.update_xaxes(title_text="Implied CAGR (%)")
    fig.update_yaxes(title_text="Frequency")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 11: Monte Carlo — Tornado / OAT sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def mc_tornado(
    sensitivity: dict[str, tuple[float, float, float]],
    shares: float | None,
) -> go.Figure:
    """Horizontal tornado bar: each parameter's low→high IV swing."""
    scale = shares if (shares and shares > 0) else 1e9

    params, lo_vals, hi_vals, base_vals = [], [], [], []
    for name, (lo, base, hi) in sorted(
        sensitivity.items(),
        key=lambda kv: abs(kv[1][2] - kv[1][0]),
    ):
        params.append(name)
        lo_vals.append(lo / scale)
        hi_vals.append(hi / scale)
        base_vals.append(base / scale)

    fig = go.Figure()
    for i, name in enumerate(params):
        lo, hi, base = lo_vals[i], hi_vals[i], base_vals[i]
        fig.add_trace(go.Bar(
            name=f"↓ {name}",
            y=[name],
            x=[lo - base],
            base=[base],
            orientation="h",
            marker_color=RED, opacity=0.8,
            showlegend=i == 0,
            hovertemplate=f"{name} (low 1σ): ${{x:.2f}}/share<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name=f"↑ {name}",
            y=[name],
            x=[hi - base],
            base=[base],
            orientation="h",
            marker_color=GREEN, opacity=0.8,
            showlegend=i == 0,
            hovertemplate=f"{name} (high 1σ): ${{x:.2f}}/share<extra></extra>",
        ))

    label = "$ per share" if shares else "$ equity ($B)"
    _base_layout(fig, "Sensitivity Tornado: Which Input Drives IV Uncertainty Most?")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text=label)
    fig.update_yaxes(title_text="Input parameter")
    return fig


__all__ = [
    "rationality_gap_chart",
    "revenue_projection_chart",
    "price_vs_intrinsic",
    "historical_growth_distribution",
    "implied_growth_sensitivity",
    "earnings_sentiment_chart",
    "earnings_tone_timeline",
    "earnings_topic_chart",
    "mc_intrinsic_histogram",
    "mc_cagr_histogram",
    "mc_tornado",
]
