"""Rationality vs. Market (RvM) — Forecasting Tool.

Auto-Predictive Mode  — solves the growth rate the market is pricing in.
Manual Sensitivity    — user-defined assumptions → intrinsic value.
Rationality Check     — flags Speculative / Rational / Pessimistic via z-score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from modules import edgar, market_data, visuals
from modules.reverse_dcf import solve_implied_growth
from modules.forward_dcf import intrinsic_value
from modules.rationality import assess, historical_growth_stats


st.set_page_config(page_title="RvM — Rationality vs. Market", layout="wide", page_icon="📊")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
  .stApp { background-color: #0b0f19; }
  div[data-testid="stMetric"] { background:#141a2a; border:1px solid #1f2937; border-radius:6px; padding:10px 14px; }
  .verdict-badge { display:inline-block; padding:8px 18px; border-radius:4px; font-weight:700;
                   letter-spacing:1.5px; font-size:14px; }
  .small-mono { font-family: ui-monospace, monospace; color:#9ca3af; font-size:12px; }
  .terminal-card { background:#141a2a; border:1px solid #1f2937; border-radius:6px;
                   padding:14px 18px; margin-top:8px; }
  h1, h2, h3 { color:#e6edf3; }
</style>
""", unsafe_allow_html=True)

st.title("📊 RvM — Rationality vs. Market")
st.markdown(
    "<span class='small-mono'>"
    "Reverse-DCF · Implied Growth · Rationality Gap · Manual Sensitivity"
    "</span>", unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Input")
    ticker = st.text_input("US-listed ticker", value="AAPL").upper().strip()
    forecast_years = st.slider("Forecast horizon (yrs)", 5, 15, 10)
    st.markdown("---")
    st.subheader("Macro")
    rf = st.number_input("Risk-free rate (%)", value=4.25, step=0.05) / 100
    erp = st.number_input("Equity risk premium (%)", value=5.0, step=0.1) / 100
    kd_pre = st.number_input("Pre-tax cost of debt (%)", value=4.5, step=0.1) / 100
    tax = st.number_input("Tax rate (%)", value=21.0, step=0.5) / 100
    st.markdown("---")
    run = st.button("⚡ ANALYSE", type="primary", use_container_width=True)

if not run and "loaded" not in st.session_state:
    st.info("Enter a ticker and press **ANALYSE** to reverse-engineer the market-implied growth rate.")
    st.stop()

ticker_info = edgar.resolve_ticker(ticker)
if not ticker_info:
    st.error(f"Ticker '{ticker}' not found in SEC EDGAR.")
    st.stop()

with st.spinner(f"Pulling fundamentals for {ticker}…"):
    facts = edgar.fetch_company_facts(ticker_info["cik"])
    fin = facts.build_financials(years=10)
    snap = market_data.fetch_market_snapshot(ticker)

st.session_state["loaded"] = True


def _row(key: str) -> pd.Series:
    if key not in fin.index:
        return pd.Series(dtype="float64")
    return fin.loc[key].astype(float).dropna()


revenue = _row("Revenue")
cfo = _row("CashFromOps")
capex = _row("CapEx").abs()
ebit = _row("OperatingIncome")
lt_debt = _row("LongTermDebt")
cash = _row("CashAndEquivalents")

if revenue.empty:
    st.error("No revenue data available — cannot run Reverse DCF.")
    st.stop()

base_rev = float(revenue.iloc[-1])
fcf = (cfo - capex.reindex(cfo.index).fillna(0)) if not cfo.empty else pd.Series(dtype="float64")
fcf_margin_hist = float((fcf / revenue.reindex(fcf.index)).dropna().tail(3).mean()) if not fcf.empty else 0.15
op_margin_hist = float((ebit / revenue.reindex(ebit.index)).dropna().tail(3).mean()) if not ebit.empty else 0.20
capex_pct_hist = float((capex / revenue.reindex(capex.index)).dropna().tail(3).mean()) if not capex.empty else 0.05

market_cap = float(snap.market_cap) if snap.market_cap else None
price = float(snap.price) if snap.price else None
shares = float(snap.shares_outstanding) if snap.shares_outstanding else None
beta = float(snap.beta) if snap.beta else 1.0
last_debt = float(lt_debt.iloc[-1]) if not lt_debt.empty else 0.0
last_cash = float(cash.iloc[-1]) if not cash.empty else 0.0
net_debt = last_debt - last_cash

ke = rf + beta * erp
debt_w = (last_debt / (last_debt + market_cap)) if market_cap else 0.0
wacc_est = (1 - debt_w) * ke + debt_w * kd_pre * (1 - tax)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Ticker", ticker)
c2.metric("Price", f"${price:,.2f}" if price else "—")
c3.metric("Market Cap", f"${market_cap/1e9:,.1f}B" if market_cap else "—")
c4.metric("Beta", f"{beta:.2f}")
c5.metric("Cost of Equity", f"{ke*100:.2f}%")
c6.metric("WACC (est.)", f"{wacc_est*100:.2f}%")

st.caption(snap.long_name or ticker_info["title"])
st.markdown("---")

tab_auto, tab_manual, tab_compare, tab_data = st.tabs([
    "🤖 Auto · Reverse-DCF",
    "🎛️  Manual Sensitivity",
    "⚖️  Rationality Check",
    "📂 Fundamentals",
])

with tab_auto:
    st.subheader("Market-Implied Growth Rate")
    st.caption("Solving: what revenue CAGR must this firm sustain for today's price to be 'fair'?")

    a1, a2, a3 = st.columns(3)
    wacc_in = a1.number_input("WACC (%)", value=round(wacc_est * 100, 2), step=0.1) / 100
    term_g = a2.number_input("Terminal growth (%)", value=2.5, step=0.1) / 100
    fcf_m = a3.number_input("FCF margin (%)", value=round(fcf_margin_hist * 100, 2), step=0.5) / 100

    if not market_cap:
        st.warning("No market cap available — cannot reverse-engineer.")
    else:
        result = solve_implied_growth(
            market_cap=market_cap, base_revenue=base_rev, fcf_margin=fcf_m,
            wacc=wacc_in, terminal_growth=term_g, years=forecast_years, net_debt=net_debt,
        )

        if result.converged and result.implied_growth is not None:
            verdict = assess(result.implied_growth, revenue)
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Implied Revenue CAGR", f"{result.implied_growth*100:.2f}%")
            r2.metric("Historical μ", f"{verdict.hist_mean*100:.2f}%" if not np.isnan(verdict.hist_mean) else "—")
            r3.metric("Historical σ", f"{verdict.hist_std*100:.2f} pp" if not np.isnan(verdict.hist_std) else "—")
            r4.metric("Z-score", f"{verdict.z_score:+.2f}σ" if not np.isnan(verdict.z_score) else "—")

            st.markdown(
                f"<div class='verdict-badge' style='background:{verdict.color}; color:#0b0f19;'>"
                f"VERDICT: {verdict.verdict.upper()}</div>",
                unsafe_allow_html=True,
            )

            _, _, growths = historical_growth_stats(revenue)
            st.plotly_chart(
                visuals.rationality_gap_chart(
                    result.implied_growth, growths, verdict.hist_mean, verdict.hist_std,
                ),
                use_container_width=True,
            )

            with st.expander("Implied-growth sensitivity (WACC × Terminal g)"):
                wacc_range = np.linspace(max(0.04, wacc_in - 0.02), wacc_in + 0.02, 5)
                term_range = np.linspace(0.015, 0.04, 6)
                st.plotly_chart(
                    visuals.implied_growth_sensitivity(
                        wacc_range, term_range, market_cap, base_rev, fcf_m,
                        forecast_years, net_debt, solve_implied_growth,
                    ),
                    use_container_width=True,
                )
        else:
            st.error(
                "Solver did not converge — implied growth lies outside the [-30%, +50%] bracket. "
                "Try adjusting WACC, FCF margin, or terminal growth."
            )

with tab_manual:
    st.subheader("Forward DCF — Your Assumptions")
    st.caption("Build the intrinsic value bottom-up. Compare against the market-implied verdict.")

    default_g = 0.05
    if len(revenue) > 1 and revenue.iloc[0] > 0:
        default_g = (revenue.iloc[-1] / revenue.iloc[0]) ** (1.0 / (len(revenue) - 1)) - 1.0
    default_g = max(-0.05, min(0.20, default_g))

    m1, m2, m3 = st.columns(3)
    g_user = m1.slider("Revenue growth (%)", -10.0, 30.0, round(default_g * 100, 1), 0.1) / 100
    om_user = m2.slider("Operating margin (%)", 0.0, 60.0, round(op_margin_hist * 100, 1), 0.5) / 100
    capex_user = m3.slider("CapEx / Revenue (%)", 0.0, 25.0, round(capex_pct_hist * 100, 1), 0.5) / 100

    n1, n2, n3 = st.columns(3)
    tax_user = n1.slider("Tax rate (%)", 10.0, 35.0, round(tax * 100, 1), 0.5) / 100
    wacc_user = n2.slider("WACC (%)", 4.0, 15.0, round(wacc_est * 100, 1), 0.1) / 100
    term_user = n3.slider("Terminal growth (%)", 0.5, 4.0, 2.5, 0.1) / 100

    iv = intrinsic_value(
        base_revenue=base_rev, revenue_growth=g_user, operating_margin=om_user,
        tax_rate=tax_user, capex_pct=capex_user, wacc=wacc_user,
        terminal_growth=term_user, years=forecast_years,
        shares_out=shares, current_price=price, net_debt=net_debt,
    )

    s1, s2, s3 = st.columns(3)
    s1.metric("Intrinsic Equity", f"${iv.intrinsic_equity/1e9:,.1f}B")
    s2.metric("Per-share value", f"${iv.intrinsic_per_share:,.2f}" if iv.intrinsic_per_share else "—")
    s3.metric(
        "Margin of Safety",
        f"{iv.margin_of_safety*100:+.1f}%" if iv.margin_of_safety is not None else "—",
        delta_color="normal" if (iv.margin_of_safety or 0) >= 0 else "inverse",
    )

    if iv.intrinsic_per_share and price:
        st.plotly_chart(visuals.price_vs_intrinsic(price, iv.intrinsic_per_share), use_container_width=True)

    st.markdown("**Projected cash flows**")
    st.dataframe(
        iv.projections.style.format({"Revenue": "${:,.0f}", "FCFF": "${:,.0f}", "PV_FCFF": "${:,.0f}"}),
        use_container_width=True,
    )

with tab_compare:
    st.subheader("Implied vs. Realised — Is the Market Rational?")

    if not market_cap or revenue.empty:
        st.warning("Insufficient data.")
    else:
        result = solve_implied_growth(
            market_cap=market_cap, base_revenue=base_rev, fcf_margin=fcf_margin_hist,
            wacc=wacc_est, terminal_growth=0.025, years=forecast_years, net_debt=net_debt,
        )
        _, _, growths = historical_growth_stats(revenue)

        if growths:
            st.plotly_chart(
                visuals.historical_growth_distribution(
                    growths, result.implied_growth if result.converged else None,
                ),
                use_container_width=True,
            )

        if result.converged:
            v = assess(result.implied_growth, revenue)
            gap_pp = (v.rationality_gap * 100) if v.rationality_gap is not None else 0.0
            st.markdown(
                f"""
                <div class='terminal-card'>
                  <h3 style='margin:0 0 8px 0;'>Rationality Gap: {gap_pp:+.2f} pp</h3>
                  <p class='small-mono' style='line-height:1.6;'>
                    Today's market cap implies a <b>{v.implied_growth*100:.2f}%</b> revenue CAGR
                    over the next {forecast_years} years. The company's realised average over the
                    last {len(growths)} years was <b>{v.hist_mean*100:.2f}%</b>
                    (σ = {v.hist_std*100:.2f} pp).<br/>
                    Z = {v.z_score:+.2f}σ → <b style='color:{v.color};'>{v.verdict.upper()}</b>.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab_data:
    st.subheader("Annual Fundamentals — SEC EDGAR (10-K)")
    if fin.empty:
        st.warning("No fundamentals available.")
    else:
        st.dataframe(fin.style.format("{:,.0f}", na_rep="—"), use_container_width=True)
