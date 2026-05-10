"""Bloomberg-Terminal-styled Plotly charts for the RvM tool."""

from __future__ import annotations

from typing import Callable

import numpy as np
import plotly.graph_objects as go


BG = "#0b0f19"
CARD = "#141a2a"
TXT = "#e6edf3"
ACCENT = "#4f8cff"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
GREY = "#6b7280"


def _layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=TXT, family="ui-monospace, SFMono-Regular, Menlo, monospace", size=12),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def rationality_gap_chart(
    implied: float,
    hist_growths: list[float],
    hist_mean: float,
    hist_std: float,
) -> go.Figure:
    fig = go.Figure()
    if hist_growths:
        fig.add_trace(go.Box(
            y=[g * 100 for g in hist_growths],
            name="Historical YoY",
            boxmean="sd",
            marker_color=ACCENT,
            line_color=ACCENT,
        ))
    fig.add_hline(
        y=implied * 100, line_color=RED, line_width=3,
        annotation_text=f"Implied {implied*100:.1f}%",
        annotation_position="top right",
    )
    if not np.isnan(hist_mean):
        fig.add_hline(
            y=hist_mean * 100, line_dash="dot", line_color=GREEN,
            annotation_text=f"Hist. μ {hist_mean*100:.1f}%",
            annotation_position="bottom right",
        )
        if hist_std and not np.isnan(hist_std):
            fig.add_hrect(
                y0=(hist_mean - 2 * hist_std) * 100,
                y1=(hist_mean + 2 * hist_std) * 100,
                fillcolor=GREEN, opacity=0.10, line_width=0,
                annotation_text="±2σ rational band",
                annotation_position="bottom left",
            )
    _layout(fig, "Rationality Gap — Implied vs. Historical Growth")
    fig.update_yaxes(title_text="Growth (%)")
    return fig


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
    grid = np.full((len(wacc_range), len(term_g_range)), np.nan)
    for i, w in enumerate(wacc_range):
        for j, tg in enumerate(term_g_range):
            res = solver(market_cap, base_rev, fcf_margin, w, tg, years, net_debt)
            if res.converged and res.implied_growth is not None:
                grid[i, j] = res.implied_growth * 100
    fig = go.Figure(go.Heatmap(
        z=grid,
        x=[f"{g*100:.1f}%" for g in term_g_range],
        y=[f"{w*100:.1f}%" for w in wacc_range],
        colorscale="RdYlGn_r",
        colorbar=dict(title="Implied g (%)"),
        text=np.round(grid, 1),
        texttemplate="%{text}%",
    ))
    _layout(fig, "Implied Growth Sensitivity (WACC × Terminal g)")
    fig.update_xaxes(title_text="Terminal growth")
    fig.update_yaxes(title_text="WACC")
    return fig


def price_vs_intrinsic(price: float, intrinsic: float | None) -> go.Figure:
    bars: list[tuple[str, float, str]] = [("Market Price", price, ACCENT)]
    if intrinsic is not None:
        c = GREEN if intrinsic > price else RED
        bars.append(("Intrinsic Value", intrinsic, c))
    fig = go.Figure([go.Bar(x=[n], y=[v], marker_color=c, name=n) for n, v, c in bars])
    if intrinsic is not None:
        gap_pct = (intrinsic - price) / price * 100
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=1.06, showarrow=False,
            text=f"Margin of Safety: {gap_pct:+.1f}%",
            font=dict(size=14, color=GREEN if gap_pct > 0 else RED),
        )
    _layout(fig, "Market Price vs. Intrinsic Value")
    fig.update_yaxes(title_text="$ / share")
    return fig


def historical_growth_distribution(growths: list[float], implied: float | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=[g * 100 for g in growths], nbinsx=12,
        marker_color=ACCENT, name="Annual YoY %",
    ))
    if implied is not None:
        fig.add_vline(
            x=implied * 100, line_color=RED, line_width=3,
            annotation_text=f"Implied {implied*100:.1f}%", annotation_position="top",
        )
    _layout(fig, "Historical YoY Growth Distribution")
    fig.update_xaxes(title_text="YoY (%)")
    fig.update_yaxes(title_text="Frequency")
    return fig


__all__ = [
    "rationality_gap_chart",
    "implied_growth_sensitivity",
    "price_vs_intrinsic",
    "historical_growth_distribution",
]
