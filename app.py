"""Rationality vs. Market (RvM) — Reverse-DCF Forecasting Tool.

Core question: What revenue growth rate is the market currently pricing in?

Auto mode  — given market price, solve for the implied growth rate (Reverse-DCF).
Manual mode — given your own growth assumptions, compute intrinsic value.
Rationality — flag whether the implied rate is Speculative / Rational / Pessimistic
              vs. the company's own historical realised growth distribution.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from modules import edgar, market_data, visuals, earnings as earnings_mod, montecarlo as mc_mod
from modules.reverse_dcf import solve_implied_growth
from modules.forward_dcf import intrinsic_value
from modules.rationality import assess, historical_growth_stats


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RvM — Rationality vs. Market",
    layout="wide",
    page_icon="📡",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"], [class*="st-"] {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace !important;
}
.main .block-container { padding-top: 1rem; }

/* ── Custom metric card ── */
.rvm-card {
  background: #141a2a;
  border: 1px solid #1f2937;
  border-left: 3px solid #4f8cff;
  border-radius: 6px;
  padding: 10px 14px;
  flex: 1 1 165px;
  min-width: 155px;
  max-width: 100%;
  overflow: hidden;
}
.rvm-card-label {
  color: #6b7280;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1.1px;
  margin-bottom: 4px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.rvm-card-value {
  color: #e6edf3;
  font-size: 17px;
  font-weight: 700;
  line-height: 1.15;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  letter-spacing: -0.3px;
}
.rvm-card-sub {
  color: #9ca3af;
  font-size: 10px;
  margin-top: 3px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.rvm-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin: 8px 0 16px 0;
}

/* ── Data freshness badge ── */
.rvm-stamp {
  display: inline-block;
  background: #1f2937;
  color: #9ca3af;
  font-size: 10px;
  padding: 3px 9px;
  border-radius: 10px;
  letter-spacing: 0.6px;
  margin-left: 8px;
  vertical-align: middle;
}
.rvm-stamp.stale { background: #2d1f07; color: #fbbf24; }

/* ── Verdict badge ── */
.verdict-badge {
  display: inline-block;
  padding: 8px 20px;
  border-radius: 4px;
  font-weight: 800;
  letter-spacing: 2px;
  font-size: 13px;
  margin: 10px 0;
}

/* ── Callout boxes ── */
.rvm-callout {
  background: #141a2a;
  border: 1px solid #1f2937;
  border-radius: 6px;
  padding: 14px 18px;
  margin: 10px 0;
  line-height: 1.7;
}
.rvm-callout h4 { margin: 0 0 6px 0; color: #e6edf3; font-size: 14px; }
.rvm-callout p  { margin: 0; color: #9ca3af; font-size: 12px; }
.rvm-callout b  { color: #e6edf3; }

/* ── Warning/info ── */
.rvm-warn {
  background: #2d1f07;
  border-left: 3px solid #f59e0b;
  border-radius: 4px;
  padding: 8px 14px;
  color: #fbbf24;
  font-size: 12px;
  margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Colour palette (mirrors visuals.py so tabs can colour-code inline HTML) ──
GREEN  = "#22c55e"
RED    = "#ef4444"
AMBER  = "#f59e0b"
PURPLE = "#a78bfa"
ACCENT = "#4f8cff"
MUTED  = "#6b7280"


# ── HTML helper: metric card row ──────────────────────────────────────────────
def _card(label: str, value: str, sub: str = "", accent: str = "#4f8cff") -> str:
    sub_html = f'<div class="rvm-card-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="rvm-card" style="border-left-color:{accent};">'
        f'<div class="rvm-card-label">{label}</div>'
        f'<div class="rvm-card-value">{value}</div>'
        f'{sub_html}</div>'
    )


def _cards(*items: str) -> None:
    st.markdown(
        '<div class="rvm-row">' + "".join(items) + "</div>",
        unsafe_allow_html=True,
    )


def _fmt_b(v: float) -> str:
    """Compact dollar value: T / B / M with sign preserved."""
    if v is None or not np.isfinite(v):
        return "—"
    sign = "-" if v < 0 else ""
    av = abs(v)
    if av >= 1e12:
        return f"{sign}${av/1e12:.2f}T"
    if av >= 1e9:
        return f"{sign}${av/1e9:.2f}B"
    if av >= 1e6:
        return f"{sign}${av/1e6:.0f}M"
    return f"{sign}${av:,.0f}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📡 RvM Inputs")
    ticker = st.text_input("US-listed ticker", value="AAPL", help="Enter any SEC-listed ticker").upper().strip()
    forecast_years = st.slider("Forecast horizon (years)", 5, 15, 10)
    st.markdown("---")
    st.markdown("**Macro assumptions**")
    rf      = st.number_input("Risk-free rate (%)",      value=4.25, step=0.05, format="%.2f") / 100
    erp     = st.number_input("Equity risk premium (%)", value=5.00, step=0.10, format="%.2f") / 100
    kd_pre  = st.number_input("Pre-tax cost of debt (%)", value=4.50, step=0.10, format="%.2f") / 100
    st.markdown("---")
    col_run, col_refresh = st.columns([3, 1])
    run     = col_run.button("⚡  ANALYSE", type="primary", use_container_width=True)
    refresh = col_refresh.button("🔄", help="Force-refresh live market data (bypass cache)", use_container_width=True)

# ── Gate: only run after first click ─────────────────────────────────────────
if not run and "rvm_ticker" not in st.session_state:
    st.title("📡 Rationality vs. Market")
    st.markdown("""
    <div class="rvm-callout">
      <h4>How this tool works</h4>
      <p>
        Enter a ticker and press <b>ANALYSE</b>. The engine reverse-engineers the
        revenue growth rate the market is currently implying via the current share price —
        without any forward analyst estimates. It then compares that implied rate against
        the company's own historical growth distribution and flags whether the market is
        being <b style="color:#ef4444">Speculative</b>,
        <b style="color:#22c55e">Rational</b>, or
        <b style="color:#f59e0b">Pessimistic</b>.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Resolve ticker, fetch data ────────────────────────────────────────────────
if run:
    st.session_state["rvm_ticker"] = ticker

active_ticker = st.session_state.get("rvm_ticker", ticker)

# Force-bust the in-process caches so live price/market-cap re-pull from yfinance.
if refresh or run:
    for _fn in (
        getattr(market_data, "fetch_market_snapshot", None),
        getattr(market_data, "_estimate_beta", None),
        getattr(market_data, "fetch_price_history", None),
        getattr(market_data, "fetch_risk_free_rate", None),
    ):
        if _fn is not None and hasattr(_fn, "cache_clear"):
            _fn.cache_clear()
    st.session_state["rvm_fetched_at"] = datetime.now(timezone.utc)

ticker_info = edgar.resolve_ticker(active_ticker)
if not ticker_info:
    st.error(f"Ticker **{active_ticker}** not found in SEC EDGAR. Check the symbol and try again.")
    st.stop()

with st.spinner(f"Pulling SEC EDGAR filings & market data for {active_ticker}…"):
    facts = edgar.fetch_company_facts(ticker_info["cik"])
    fin   = facts.build_financials(years=10)
    snap  = market_data.fetch_market_snapshot(active_ticker)

fetched_at: datetime | None = st.session_state.get("rvm_fetched_at")


# ── Helper: pull a row from the financials DataFrame ─────────────────────────
def _row(key: str) -> pd.Series:
    if key not in fin.index:
        return pd.Series(dtype="float64")
    return fin.loc[key].astype(float).dropna()


# ── Derive financial line items ───────────────────────────────────────────────
revenue  = _row("Revenue")
cfo      = _row("CashFromOps")
capex    = _row("CapEx").abs()
ebit     = _row("OperatingIncome")
ni       = _row("NetIncome")
lt_debt  = _row("LongTermDebt")
st_debt  = _row("ShortTermDebt").abs()
cash     = _row("CashAndEquivalents")
tax_exp  = _row("IncomeTaxExpense")
pretax   = _row("PreTaxIncome")

if revenue.empty:
    st.error("No revenue data available — SEC EDGAR returned no annual filings for this ticker.")
    st.stop()

base_rev = float(revenue.iloc[-1])

# ── Effective tax rate (3-yr median, capped 10-45%) ──────────────────────────
_tax_raw = pd.Series(dtype="float64")
if not tax_exp.empty and not pretax.empty:
    _common = tax_exp.index.intersection(pretax.index)
    if not _common.empty:
        _tax_raw = (tax_exp.reindex(_common) / pretax.reindex(_common)).dropna()
        _tax_raw = _tax_raw[(_tax_raw > 0.10) & (_tax_raw < 0.45)]
effective_tax = float(_tax_raw.tail(3).median()) if not _tax_raw.empty else 0.21

# ── FCF margin — try two methods; pick the more reliable ─────────────────────
#  Method A: (CFO − CapEx) / Revenue  [3-yr median]
fcf_margin_cfo: float | None = None
if not cfo.empty and not capex.empty:
    common_idx = cfo.index.intersection(capex.index).intersection(revenue.index)
    if not common_idx.empty:
        fcf_a = (cfo.reindex(common_idx) - capex.reindex(common_idx).fillna(0))
        fcf_a_margin = (fcf_a / revenue.reindex(common_idx)).dropna()
        if not fcf_a_margin.empty:
            fcf_margin_cfo = float(fcf_a_margin.tail(3).median())

#  Method B: NOPAT / Revenue × (1 − reinvestment rate)  [3-yr median]
fcf_margin_nopat: float | None = None
if not ebit.empty:
    common_idx = ebit.index.intersection(revenue.index)
    if not common_idx.empty:
        nopat = ebit.reindex(common_idx) * (1.0 - effective_tax)
        nopat_margin = (nopat / revenue.reindex(common_idx)).dropna()
        if not nopat_margin.empty:
            # reinvestment ≈ capex / nopat (capped 0–80%)
            if not capex.empty:
                cap_common = capex.index.intersection(nopat.index)
                if not cap_common.empty:
                    reinvest = (capex.reindex(cap_common) / nopat.reindex(cap_common)).dropna()
                    reinvest = reinvest.clip(0, 0.80)
                    r_rate = float(reinvest.tail(3).median()) if not reinvest.empty else 0.30
                else:
                    r_rate = 0.30
            else:
                r_rate = 0.30
            fcf_margin_nopat = float(nopat_margin.tail(3).median()) * (1.0 - r_rate)

# Pick best margin:
#   prefer CFO-based if > 1%; fall back to NOPAT; last resort = 10% generic
if fcf_margin_cfo is not None and fcf_margin_cfo > 0.01:
    fcf_margin_hist  = fcf_margin_cfo
    fcf_margin_label = "CFO-based (3-yr median)"
elif fcf_margin_nopat is not None and fcf_margin_nopat > 0.005:
    fcf_margin_hist  = fcf_margin_nopat
    fcf_margin_label = "NOPAT-based (3-yr median)"
else:
    fcf_margin_hist  = 0.10
    fcf_margin_label = "Default estimate (10%)"

# ── Operating margin (3-yr median) ───────────────────────────────────────────
op_margin_hist = 0.20
if not ebit.empty:
    common = ebit.index.intersection(revenue.index)
    if not common.empty:
        om = (ebit.reindex(common) / revenue.reindex(common)).dropna()
        if not om.empty:
            op_margin_hist = float(om.tail(3).median())

# ── CapEx % of revenue (3-yr median) ─────────────────────────────────────────
capex_pct_hist = 0.05
if not capex.empty:
    common = capex.index.intersection(revenue.index)
    if not common.empty:
        cx = (capex.reindex(common) / revenue.reindex(common)).dropna()
        if not cx.empty:
            capex_pct_hist = float(cx.tail(3).median())

# ── Market data ───────────────────────────────────────────────────────────────
market_cap = float(snap.market_cap) if snap.market_cap else None
price      = float(snap.price)      if snap.price      else None
shares     = float(snap.shares_outstanding) if snap.shares_outstanding else None
beta       = float(snap.beta)       if snap.beta       else 1.0

# ── Debt / cash ───────────────────────────────────────────────────────────────
last_lt_debt = float(lt_debt.iloc[-1]) if not lt_debt.empty else 0.0
last_st_debt = float(st_debt.iloc[-1]) if not st_debt.empty else 0.0
total_debt   = last_lt_debt + last_st_debt
last_cash    = float(cash.iloc[-1])    if not cash.empty    else 0.0
net_debt     = total_debt - last_cash

# ── WACC (book-value weights — more stable than market-cap weights) ───────────
ke = rf + beta * erp
if market_cap and (total_debt + market_cap) > 0:
    debt_w = total_debt / (total_debt + market_cap)
else:
    debt_w = 0.0
kd_at    = kd_pre * (1.0 - effective_tax)
wacc_est = (1.0 - debt_w) * ke + debt_w * kd_at

# Enterprise Value (used in Reverse-DCF)
ev = (market_cap + net_debt) if market_cap is not None else None

# ── Historical CAGR (used in both Manual and Monte Carlo tabs) ────────────────
_hist_cagr = 0.05
if len(revenue) > 1 and float(revenue.iloc[0]) > 0:
    _r = float(revenue.iloc[-1]) / float(revenue.iloc[0])
    _hist_cagr = _r ** (1.0 / (len(revenue) - 1)) - 1.0
_hist_cagr = float(np.clip(_hist_cagr, -0.10, 0.40))

# Historical growth std (for MC default σ)
_, _hist_std_global, _ = historical_growth_stats(revenue)
_hist_std_global = _hist_std_global or 0.08

# ── Page header ───────────────────────────────────────────────────────────────
st.title(f"📡 {active_ticker}  —  Rationality vs. Market")

# Caption with data-freshness badge
_company = snap.long_name or ticker_info.get("title", "")
if fetched_at:
    _age_min = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 60.0
    _stamp_cls = "rvm-stamp stale" if _age_min > 15 else "rvm-stamp"
    _stamp = (
        f'<span class="{_stamp_cls}">Live data fetched '
        f'{fetched_at.strftime("%Y-%m-%d %H:%M UTC")}</span>'
    )
else:
    _stamp = ""
st.markdown(
    f'<div style="color:#9ca3af;font-size:13px;margin-top:-4px;">'
    f'{_company}{_stamp}</div>',
    unsafe_allow_html=True,
)
if not price or not market_cap:
    st.markdown(
        '<div class="rvm-warn">⚠️  Live market data unavailable from yfinance. '
        'Price / market cap may be missing or stale. Click 🔄 in the sidebar to retry.</div>',
        unsafe_allow_html=True,
    )

# Header metric cards — EV decomposition shown explicitly
_cards(
    _card("Price", f"${price:,.2f}"         if price      else "—"),
    _card("Market Cap", _fmt_b(market_cap)  if market_cap else "—", sub="Shares × Price"),
    _card("Net Debt",
          _fmt_b(net_debt),
          sub="Debt − Cash",
          accent="#ef4444" if net_debt > 0 else "#22c55e"),
    _card("Enterprise Value",
          _fmt_b(ev)                         if ev is not None else "—",
          sub="MCap + Net Debt",
          accent="#f59e0b"),
    _card("Beta (β)", f"{beta:.2f}"),
    _card("WACC (est.)", f"{wacc_est*100:.2f}%",
          sub=f"ke={ke*100:.1f}% | kd={kd_at*100:.1f}%"),
)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_auto, tab_manual, tab_compare, tab_mc, tab_earn, tab_data, tab_help = st.tabs([
    "🤖  Reverse-DCF (Auto)",
    "🎛️  Manual Sensitivity",
    "⚖️  Rationality Check",
    "🎲  Monte Carlo",
    "📞  Earnings Intel",
    "📂  Fundamentals",
    "❓  Help & Methodology",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — AUTO REVERSE-DCF
# ═══════════════════════════════════════════════════════════════════════════════
with tab_auto:
    st.subheader("Market-Implied Revenue CAGR")
    st.markdown(
        "<p style='color:#6b7280;font-size:12px;margin-top:-8px;'>"
        "Solving: at what annual revenue growth rate does the DCF of future cash flows equal"
        " today's enterprise value?</p>",
        unsafe_allow_html=True,
    )

    # ── Solver input overrides ────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        wacc_in = st.number_input(
            "WACC (%)", value=round(wacc_est * 100, 2), min_value=1.0, max_value=30.0, step=0.1,
            help="Weighted Average Cost of Capital. Auto-estimated from CAPM + debt weights.",
        ) / 100
    with col_b:
        term_g = st.number_input(
            "Terminal growth (%)", value=2.5, min_value=0.0, max_value=wacc_in * 100 - 0.5, step=0.1,
            help="Long-run perpetual growth rate after the explicit forecast period. Typically GDP-linked (2–3%).",
        ) / 100
    with col_c:
        fcf_m = st.number_input(
            "FCF margin (%)",
            value=max(1.0, round(fcf_margin_hist * 100, 1)),
            min_value=0.5, max_value=70.0, step=0.5,
            help=f"Auto-derived: {fcf_margin_label}. FCFF as % of revenue.",
        ) / 100

    # ── Warn about FCF margin quality ────────────────────────────────────────
    if fcf_margin_hist < 0.03:
        st.markdown(
            f'<div class="rvm-warn">⚠️  Derived FCF margin is very low '
            f'({fcf_margin_hist*100:.1f}%). Override with a normalised estimate above.</div>',
            unsafe_allow_html=True,
        )

    if market_cap is None or ev is None:
        st.warning("No market price / market cap available — cannot run Reverse-DCF.")
    else:
        result = solve_implied_growth(
            market_cap=market_cap, base_revenue=base_rev, fcf_margin=fcf_m,
            wacc=wacc_in, terminal_growth=term_g, years=forecast_years, net_debt=net_debt,
        )

        if result.converged and result.implied_growth is not None:
            verdict = assess(result.implied_growth, revenue)

            # ── Key result metrics ────────────────────────────────────────────
            _cards(
                _card("Implied Revenue CAGR",
                      f"{result.implied_growth*100:.2f}%",
                      sub=f"{forecast_years}-yr horizon",
                      accent=verdict.color),
                _card("Historical Avg (YoY)",
                      f"{verdict.hist_mean*100:.2f}%"
                      if np.isfinite(verdict.hist_mean) else "—",
                      sub="Mean of annual growth rates"),
                _card("Historical σ",
                      f"{verdict.hist_std*100:.2f} pp"
                      if np.isfinite(verdict.hist_std) else "—",
                      sub="Std-dev of YoY rates"),
                _card("Z-score",
                      f"{verdict.z_score:+.2f}σ"
                      if np.isfinite(verdict.z_score) else "—",
                      sub="(Implied − Hist.avg) / σ",
                      accent=verdict.color),
                _card("Terminal Value %",
                      f"{result.tv_pct*100:.0f}%",
                      sub="% of EV from terminal period",
                      accent="#f59e0b" if result.tv_pct > 0.70 else "#4f8cff"),
            )

            # ── Verdict badge ─────────────────────────────────────────────────
            st.markdown(
                f'<div class="verdict-badge" '
                f'style="background:{verdict.color};color:#0b0f19;">'
                f'MARKET VERDICT: {verdict.verdict.upper()}</div>',
                unsafe_allow_html=True,
            )

            # ── TV% warning ───────────────────────────────────────────────────
            if result.tv_pct > 0.75:
                st.markdown(
                    f'<div class="rvm-warn">⚠️  Terminal value = {result.tv_pct*100:.0f}% of EV. '
                    f'This DCF is highly sensitive to the terminal growth assumption. '
                    f'Small changes in g_T materially shift the implied CAGR.</div>',
                    unsafe_allow_html=True,
                )

            # ── Narrative ─────────────────────────────────────────────────────
            gap_pp = verdict.rationality_gap * 100 if verdict.rationality_gap is not None else 0.0
            ref_cagr = verdict.hist_cagr_5yr or verdict.hist_cagr_3yr
            cagr_line = (
                f"The realised 5-yr CAGR was <b>{ref_cagr*100:.1f}%</b>."
                if ref_cagr is not None else ""
            )
            st.markdown(
                f"""
                <div class="rvm-callout">
                  <h4>What the market is saying</h4>
                  <p>
                    At today's enterprise value of <b>{_fmt_b(ev)}</b>, the market is pricing in
                    a revenue CAGR of <b style="color:{verdict.color};">{result.implied_growth*100:.2f}%</b>
                    per year for the next <b>{forecast_years} years</b>,
                    reaching <b>{_fmt_b(result.implied_revenue_final)}</b> in revenue by year {forecast_years}.
                    The company's historical annual growth averaged <b>{verdict.hist_mean*100:.1f}%</b>
                    (σ = {verdict.hist_std*100:.1f} pp). {cagr_line}
                    The implied rate sits <b>{gap_pp:+.1f} pp</b> above the historical average
                    (z = {verdict.z_score:+.2f}σ) →
                    <b style="color:{verdict.color};">{verdict.verdict}</b>.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Charts ────────────────────────────────────────────────────────
            col_l, col_r = st.columns([3, 2])
            with col_l:
                st.plotly_chart(
                    visuals.rationality_gap_chart(
                        result.implied_growth, revenue, verdict.hist_mean, verdict.hist_std,
                    ),
                    use_container_width=True,
                )
            with col_r:
                st.plotly_chart(
                    visuals.revenue_projection_chart(revenue, result.implied_growth, forecast_years),
                    use_container_width=True,
                )

            with st.expander("🔬  Sensitivity: Implied CAGR across WACC × Terminal growth"):
                wacc_lo = max(0.04, wacc_in - 0.025)
                wacc_hi = min(0.20, wacc_in + 0.025)
                wacc_range = np.linspace(wacc_lo, wacc_hi, 6)
                term_range = np.linspace(0.010, min(0.050, wacc_in - 0.01), 7)
                st.plotly_chart(
                    visuals.implied_growth_sensitivity(
                        wacc_range, term_range, market_cap, base_rev, fcf_m,
                        forecast_years, net_debt, solve_implied_growth,
                    ),
                    use_container_width=True,
                )

        else:
            st.error(
                f"Solver could not converge. {result.failure_reason} "
                "Try adjusting WACC, FCF margin, or terminal growth rate."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MANUAL SENSITIVITY (Forward DCF)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.subheader("Forward DCF — Your Own Assumptions")
    st.markdown(
        "<p style='color:#6b7280;font-size:12px;margin-top:-8px;'>"
        "Enter your view on growth and profitability. The tool computes intrinsic value "
        "and shows how far it diverges from the market price.</p>",
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3 = st.columns(3)
    g_user  = mc1.slider("Revenue growth (%)", -10.0, 35.0, round(_hist_cagr * 100, 1), 0.1) / 100
    om_user = mc2.slider("Operating margin (%)", 0.0, 65.0, round(op_margin_hist * 100, 1), 0.5) / 100
    cx_user = mc3.slider("CapEx / Revenue (%)", 0.0, 30.0, round(capex_pct_hist * 100, 1), 0.5) / 100

    mn1, mn2, mn3 = st.columns(3)
    tx_user   = mn1.slider("Tax rate (%)", 5.0, 45.0, round(effective_tax * 100, 1), 0.5) / 100
    wacc_user = mn2.slider("WACC (%)", 4.0, 20.0, round(wacc_est * 100, 1), 0.1) / 100
    term_user = mn3.slider("Terminal growth (%)", 0.5, 5.0, 2.5, 0.1) / 100

    iv = intrinsic_value(
        base_revenue=base_rev, revenue_growth=g_user, operating_margin=om_user,
        tax_rate=tx_user, capex_pct=cx_user, wacc=wacc_user,
        terminal_growth=term_user, years=forecast_years,
        shares_out=shares, current_price=price, net_debt=net_debt,
    )

    # Result cards
    mos_color = "#22c55e" if (iv.margin_of_safety or 0) >= 0 else "#ef4444"
    _cards(
        _card("Intrinsic Equity", _fmt_b(iv.intrinsic_equity)),
        _card("Per-share Value",
              f"${iv.intrinsic_per_share:,.2f}" if iv.intrinsic_per_share else "—"),
        _card("Margin of Safety",
              f"{iv.margin_of_safety*100:+.1f}%" if iv.margin_of_safety is not None else "—",
              sub="(Intrinsic − Market) / Market",
              accent=mos_color),
        _card("Your Growth Assumption", f"{g_user*100:.1f}% p.a."),
    )

    # Show implied CAGR from Auto tab for comparison
    if market_cap:
        _auto = solve_implied_growth(
            market_cap=market_cap, base_revenue=base_rev, fcf_margin=fcf_margin_hist,
            wacc=wacc_user, terminal_growth=term_user, years=forecast_years, net_debt=net_debt,
        )
        implied_for_comparison = _auto.implied_growth if _auto.converged else None
    else:
        implied_for_comparison = None

    if iv.intrinsic_per_share and price:
        st.plotly_chart(
            visuals.price_vs_intrinsic(
                price, iv.intrinsic_per_share,
                implied_g=implied_for_comparison,
                user_g=g_user,
            ),
            use_container_width=True,
        )

    with st.expander("📋  Projected cash flows"):
        fmt = {"Revenue": "${:,.0f}", "FCFF": "${:,.0f}", "PV_FCFF": "${:,.0f}"}
        st.dataframe(
            iv.projections.style.format(fmt, na_rep="—"),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RATIONALITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Implied vs. Realised — Statistical Rationality Check")

    if not market_cap or revenue.empty:
        st.warning("Insufficient market or revenue data.")
    else:
        rc_result = solve_implied_growth(
            market_cap=market_cap, base_revenue=base_rev, fcf_margin=fcf_margin_hist,
            wacc=wacc_est, terminal_growth=0.025, years=forecast_years, net_debt=net_debt,
        )
        _, _, growths = historical_growth_stats(revenue)
        v = assess(rc_result.implied_growth if rc_result.converged else None, revenue)

        # Summary metrics
        _cards(
            _card("Implied CAGR",
                  f"{rc_result.implied_growth*100:.2f}%"
                  if rc_result.converged else "—",
                  accent=v.color),
            _card("Hist. YoY Mean",
                  f"{v.hist_mean*100:.2f}%"
                  if np.isfinite(v.hist_mean) else "—"),
            _card("3-yr CAGR",
                  f"{v.hist_cagr_3yr*100:.2f}%"
                  if v.hist_cagr_3yr is not None else "—"),
            _card("5-yr CAGR",
                  f"{v.hist_cagr_5yr*100:.2f}%"
                  if v.hist_cagr_5yr is not None else "—"),
            _card("Z-score",
                  f"{v.z_score:+.2f}σ"
                  if np.isfinite(v.z_score) else "—",
                  accent=v.color),
            _card("Verdict", v.verdict, accent=v.color),
        )

        if growths:
            st.plotly_chart(
                visuals.historical_growth_distribution(growths, rc_result.implied_growth if rc_result.converged else None),
                use_container_width=True,
            )

        if rc_result.converged:
            gap_pp = (v.rationality_gap * 100) if v.rationality_gap is not None else 0.0
            ref = v.hist_cagr_5yr or v.hist_cagr_3yr
            bullets = []
            if np.isfinite(v.hist_mean):
                bullets.append(f"Mean YoY growth: <b>{v.hist_mean*100:.1f}%</b> (σ = {v.hist_std*100:.1f} pp)")
            if ref:
                bullets.append(f"5-yr realised CAGR: <b>{ref*100:.1f}%</b>")
            bullets.append(f"Market-implied CAGR: <b style='color:{v.color};'>{rc_result.implied_growth*100:.1f}%</b>")
            bullets.append(f"Rationality gap: <b>{gap_pp:+.1f} pp</b>  (z = {v.z_score:+.2f}σ)")
            bullets.append(f"Verdict: <b style='color:{v.color};'>{v.verdict}</b>")

            st.markdown(
                '<div class="rvm-callout"><h4>Analysis</h4>'
                + "<br>".join(f"· {b}" for b in bullets)
                + "</div>",
                unsafe_allow_html=True,
            )

            if v.verdict == "Speculative":
                st.markdown(
                    '<div class="rvm-warn">'
                    'The market is demanding growth well above the company\'s historical track record. '
                    'This does not necessarily mean the stock will fall — it means the current price '
                    'leaves <b>very little room for operational disappointment</b>.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            elif v.verdict == "Pessimistic":
                st.markdown(
                    '<div class="rvm-warn">'
                    'The market is pricing in growth below what the company has historically delivered. '
                    'This may indicate a <b>structural decline concern</b> or an '
                    '<b>asymmetric opportunity</b> if the pessimism is overdone.'
                    '</div>',
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mc:
    st.subheader("Monte Carlo Valuation — Parametric Uncertainty")
    st.markdown(
        "<p style='color:#6b7280;font-size:12px;margin-top:-8px;'>"
        "Sample thousands of (growth, margin, WACC, terminal-g) draws from their "
        "uncertainty distributions. Output: full distribution of intrinsic values "
        "and implied CAGRs, percentile table, and OAT tornado showing which input "
        "dominates valuation uncertainty.</p>",
        unsafe_allow_html=True,
    )

    if not market_cap or not price:
        st.warning("Live market data required. Click ⚡ ANALYSE first.")
    else:
        st.markdown("#### Distribution Inputs &nbsp;*(centre = auto-derived; σ = your uncertainty)*")
        mc_c1, mc_c2, mc_c3, mc_c4 = st.columns(4)
        mc_g_mu   = mc_c1.number_input("Rev. growth mean (%)",  value=round(float(np.clip(_hist_cagr * 100, -10, 30)), 1), step=0.5, format="%.1f") / 100
        mc_g_sig  = mc_c1.number_input("Rev. growth σ (%)",     value=round(float(np.clip(_hist_std_global * 100, 1.0, 25.0)), 1), step=0.5, format="%.1f", help="1-std spread on revenue growth") / 100
        mc_w_mu   = mc_c2.number_input("WACC mean (%)",         value=round(wacc_est*100, 2), step=0.1, format="%.2f") / 100
        mc_w_sig  = mc_c2.number_input("WACC σ (%)",            value=1.0, step=0.1, format="%.1f", help="±1σ uncertainty in WACC") / 100
        mc_op_mu  = mc_c3.number_input("Op. margin mean (%)",   value=round(op_margin_hist*100, 1), step=0.5, format="%.1f") / 100
        mc_op_sig = mc_c3.number_input("Op. margin σ (%)",      value=round(max(op_margin_hist*0.20*100, 1.0), 1), step=0.5, format="%.1f", help="Uncertainty in operating margin") / 100
        mc_tg_mu  = mc_c4.number_input("Terminal growth mean (%)", value=2.5, step=0.1, format="%.1f") / 100
        mc_tg_sig = mc_c4.number_input("Terminal growth σ (%)",    value=0.5, step=0.1, format="%.1f") / 100

        mc_col_n, mc_col_run = st.columns([2, 1])
        n_sims_choice = mc_col_n.select_slider(
            "Simulations", options=[500, 1000, 2000, 5000], value=2000,
        )
        run_mc = mc_col_run.button("▶  Run Simulation", type="primary", use_container_width=True)

        if run_mc or "mc_result" in st.session_state:
            if run_mc:
                with st.spinner(f"Running {n_sims_choice:,} simulations…"):
                    mc_res = mc_mod.run_monte_carlo(
                        base_rev=base_rev,
                        net_debt=net_debt,
                        market_cap=market_cap,
                        shares=shares,
                        current_price=price,
                        tax_rate=effective_tax,
                        capex_pct=capex_pct_hist,
                        years=forecast_years,
                        wacc_mu=mc_w_mu,   wacc_sig=mc_w_sig,
                        g_mu=mc_g_mu,      g_sig=mc_g_sig,
                        op_mu=mc_op_mu,    op_sig=mc_op_sig,
                        tg_mu=mc_tg_mu,    tg_sig=mc_tg_sig,
                        fcf_mu=fcf_margin_hist, fcf_sig=max(fcf_margin_hist*0.30, 0.01),
                        n_sims=n_sims_choice,
                    )
                st.session_state["mc_result"] = mc_res
            else:
                mc_res = st.session_state["mc_result"]

            # ── Summary cards ──────────────────────────────────────────────────
            pu = mc_res.prob_undervalued
            er = mc_res.expected_return
            _cards(
                _card("Median Fair Value",
                      f"${mc_res.p50:,.2f}" if mc_res.p50 else "—",
                      sub="50th pct of simulated IVs",
                      accent=GREEN if (mc_res.p50 or 0) > price else RED),
                _card("P5 / P95",
                      f"${mc_res.p5:,.0f} – ${mc_res.p95:,.0f}" if mc_res.p5 else "—",
                      sub="90% confidence interval"),
                _card("Prob. Undervalued",
                      f"{pu*100:.1f}%" if pu is not None else "—",
                      sub="P(IV > market price)",
                      accent=GREEN if (pu or 0) > 0.60 else AMBER if (pu or 0) > 0.40 else RED)
                      if pu is not None else _card("Prob. Undervalued", "—"),
                _card("Expected Return",
                      f"{er*100:+.1f}%" if er is not None else "—",
                      sub="E[(IV/Price) − 1]",
                      accent=GREEN if (er or 0) > 0 else RED),
                _card("Implied CAGR P50",
                      f"{mc_res.cagr_p50*100:.1f}%" if mc_res.cagr_p50 else "—",
                      sub=f"Range: {mc_res.cagr_p5*100:.1f}% – {mc_res.cagr_p95*100:.1f}%"
                      if mc_res.cagr_p5 else "",
                      accent=PURPLE),
                _card("Valid Sims",
                      f"{mc_res.n_valid_fwd:,} / {mc_res.n_sims:,}",
                      sub="Forward DCF convergence"),
            )

            if mc_res.scenarios:
                st.markdown("#### Bull / Base / Bear Scenarios")
                scenario_df = pd.DataFrame(mc_res.scenarios).T
                st.dataframe(scenario_df, use_container_width=True)

            # ── Charts ────────────────────────────────────────────────────────
            ch1, ch2 = st.columns(2)
            with ch1:
                if len(mc_res.iv_per_share) > 10:
                    st.plotly_chart(
                        visuals.mc_intrinsic_histogram(mc_res.iv_per_share, price),
                        use_container_width=True,
                    )
            with ch2:
                if len(mc_res.implied_cagrs) > 10:
                    # Get base implied from Auto tab calculation
                    _base_res = solve_implied_growth(
                        market_cap=market_cap, base_revenue=base_rev,
                        fcf_margin=fcf_margin_hist, wacc=wacc_est,
                        terminal_growth=0.025, years=forecast_years, net_debt=net_debt,
                    )
                    base_cagr = _base_res.implied_growth if _base_res.converged else None
                    st.plotly_chart(
                        visuals.mc_cagr_histogram(mc_res.implied_cagrs, base_cagr),
                        use_container_width=True,
                    )

            if mc_res.sensitivity:
                st.plotly_chart(
                    visuals.mc_tornado(mc_res.sensitivity, shares),
                    use_container_width=True,
                )

            st.markdown(
                f"""<div class="rvm-callout">
                <h4>Interpreting these results</h4>
                <p>
                The histogram shows the distribution of fair values generated by sampling
                each model input from its uncertainty distribution.
                A <b>probability of undervaluation of {f"{pu*100:.0f}%" if pu else "—"}</b>
                means that in that fraction of simulated worlds, your intrinsic value
                estimate exceeds the current price of <b>${price:,.2f}</b>.
                The tornado chart ranks which parameter drives the <i>most</i> spread in
                outcomes — focus your research there. If WACC dominates, narrow your
                discount-rate view. If revenue growth dominates, that's where thesis
                certainty matters most.
                </p></div>""",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EARNINGS INTEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_earn:
    st.subheader("Earnings Call Quantitative Analysis")
    st.markdown(
        "<p style='color:#6b7280;font-size:12px;margin-top:-8px;'>"
        "Loughran-McDonald (2011) financial NLP — the academic standard for "
        "earnings-text analysis. Decomposes management language into positive, "
        "negative, uncertainty, and litigious signals. Source: SEC EDGAR 8-K filings "
        "or paste any transcript text below.</p>",
        unsafe_allow_html=True,
    )

    earn_c1, earn_c2 = st.columns([3, 1])
    with earn_c2:
        fetch_btn = st.button("📥  Auto-fetch SEC 8-K", use_container_width=True,
                              help="Searches EDGAR for the most recent earnings press release")
        clear_btn = st.button("🗑️  Clear", use_container_width=True)

    if clear_btn:
        for k in ("earn_text", "earn_url", "earn_date", "earn_analysis"):
            st.session_state.pop(k, None)

    if fetch_btn and ticker_info:
        with st.spinner("Searching SEC EDGAR for earnings filing…"):
            earnings_mod.fetch_edgar_transcript.cache_clear()
            etxt, eurl, edate = earnings_mod.fetch_edgar_transcript(
                active_ticker, str(ticker_info["cik"])
            )
        if etxt:
            st.session_state["earn_text"] = etxt
            st.session_state["earn_url"]  = eurl
            st.session_state["earn_date"] = edate
            st.session_state.pop("earn_analysis", None)
            st.success(f"Loaded {len(etxt.split()):,} words from SEC 8-K ({edate})")
        else:
            st.warning(
                "Could not auto-fetch an earnings document from SEC EDGAR. "
                "Paste the transcript text manually below."
            )

    with earn_c1:
        manual_text = st.text_area(
            "Paste earnings call transcript / press release text here",
            value=st.session_state.get("earn_text", ""),
            height=180,
            placeholder="Paste the full transcript from Seeking Alpha, Motley Fool, "
                        "the company's IR page, or SEC EDGAR. Any length ≥ 500 words.",
        )

    if manual_text:
        st.session_state["earn_text"] = manual_text

    run_earn = st.button("🔬  Analyse Text", type="primary",
                         disabled=not st.session_state.get("earn_text"))

    if run_earn or "earn_analysis" in st.session_state:
        if run_earn:
            text_to_analyse = st.session_state.get("earn_text", "")
            if len(text_to_analyse.split()) < 100:
                st.error("Text is too short (< 100 words). Paste a full transcript.")
                st.stop()
            with st.spinner("Running Loughran-McDonald analysis…"):
                ea = earnings_mod.analyze_transcript(
                    text_to_analyse,
                    source_url=st.session_state.get("earn_url", "manual"),
                    filing_date=st.session_state.get("earn_date"),
                )
            st.session_state["earn_analysis"] = ea
        else:
            ea = st.session_state["earn_analysis"]

        # ── Verdict badge ──────────────────────────────────────────────────
        st.markdown(
            f'<div class="verdict-badge" '
            f'style="background:{ea.sentiment_color};color:#0b0f19;">'
            f'EARNINGS TONE: {ea.net_sentiment.upper()}</div>',
            unsafe_allow_html=True,
        )
        if ea.filing_date:
            st.caption(f"Filing date: {ea.filing_date} · Source: {ea.source_url[:80]}")

        # ── Top-line metrics ───────────────────────────────────────────────
        _cards(
            _card("Tone Score",
                  f"{ea.tone_score:+.3f}",
                  sub="(Positive − Negative) / (Pos + Neg)",
                  accent=ea.sentiment_color),
            _card("Positive words",
                  f"{ea.lm_positive}",
                  sub=f"{ea.pos_per_1k:.1f} per 1k words",
                  accent=GREEN),
            _card("Negative words",
                  f"{ea.lm_negative}",
                  sub=f"{ea.neg_per_1k:.1f} per 1k words",
                  accent=RED),
            _card("Uncertainty",
                  f"{ea.lm_uncertainty}",
                  sub=f"{ea.unc_per_1k:.1f} per 1k words",
                  accent=AMBER),
            _card("Litigious",
                  f"{ea.lm_litigious}",
                  sub="Legal-risk language count"),
            _card("Words analysed",
                  f"{ea.word_count:,}",
                  sub=f"{ea.sentence_count} sentences"),
        )

        # Forward/backward language cards
        fwd_col = GREEN if ea.fwd_bwd_ratio > 1.2 else MUTED
        _cards(
            _card("Forward-looking words", str(ea.forward_looking),
                  sub="Will / expect / plan / forecast…", accent=fwd_col),
            _card("Backward-looking words", str(ea.backward_looking),
                  sub="Was / had / reported / grew…"),
            _card("Fwd / Bwd ratio",
                  f"{ea.fwd_bwd_ratio:.2f}×",
                  sub="> 1.0 = more future-focused",
                  accent=fwd_col),
            _card("Mgmt optimism signals", str(ea.management_optimism),
                  sub="Confident / thrilled / record…",
                  accent=GREEN if ea.management_optimism > 5 else MUTED),
        )

        # ── Charts ────────────────────────────────────────────────────────
        earn_ch1, earn_ch2 = st.columns(2)
        with earn_ch1:
            st.plotly_chart(
                visuals.earnings_sentiment_chart(ea),
                use_container_width=True,
            )
        with earn_ch2:
            st.plotly_chart(
                visuals.earnings_topic_chart(ea.topic_hits),
                use_container_width=True,
            )

        if len(ea.paragraph_tones) >= 4:
            st.plotly_chart(
                visuals.earnings_tone_timeline(ea.paragraph_tones),
                use_container_width=True,
            )

        # ── Top words ─────────────────────────────────────────────────────
        with st.expander("🔤  Most frequent content words"):
            tw_df = pd.DataFrame(ea.top_words, columns=["Word", "Count"])
            st.dataframe(tw_df, use_container_width=True, height=300)

        # ── Interpretation ────────────────────────────────────────────────
        tone_interp = {
            "Bullish":          "Management is clearly optimistic. Heavy positive language signals confidence in outlook. Verify whether substance (guidance, margins) supports the rhetoric.",
            "Neutral-Positive": "Slight positive lean. Management is measured but constructive. Look for specific quantitative commitments that back up the tone.",
            "Neutral":          "Balanced language. Management is not over-selling or under-selling. Focus on specific guidance and margin trajectory rather than sentiment.",
            "Cautious":         "Elevated negative and uncertainty language. Management is flagging risks. Identify whether these are transient (supply chain, macro) or structural (competition, pricing).",
            "Bearish":          "Strongly negative tone. Management may be guiding to disappointment. Cross-check against the historical baseline — is this a one-off event or trend?",
        }.get(ea.net_sentiment, "")

        st.markdown(
            f'<div class="rvm-callout"><h4>Quantitative Interpretation</h4>'
            f'<p><b>Overall tone:</b> {ea.net_sentiment} (score = {ea.tone_score:+.3f}). {tone_interp}</p>'
            f'<p style="margin-top:8px;"><b>Uncertainty ratio:</b> {ea.unc_per_1k:.1f} uncertainty-words per 1k. '
            f'Academic benchmark for US large-caps ≈ 30–60. '
            f'{"Above benchmark — management is hedging heavily." if ea.unc_per_1k > 60 else "Within normal range." if ea.unc_per_1k > 30 else "Below benchmark — unusually decisive language."}'
            f'</p>'
            f'<p style="margin-top:8px;"><b>Litigious language:</b> {ea.lm_litigious} hits. '
            f'{"Elevated — review 10-K risk factors for active litigation." if ea.lm_litigious > 20 else "Unremarkable."}'
            f'</p></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader(f"Annual Fundamentals — {active_ticker} (SEC EDGAR 10-K)")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Fiscal years", str(fin.shape[1]) if not fin.empty else "0")
    col_info2.metric("Line items", str(fin.shape[0]) if not fin.empty else "0")
    col_info3.metric(
        "FCF margin used",
        f"{fcf_margin_hist*100:.1f}%",
        help=fcf_margin_label,
    )

    if fin.empty:
        st.warning("No fundamental data available from SEC EDGAR for this ticker.")
    else:
        def _fmt_fin(v: float) -> str:
            if pd.isna(v):
                return "—"
            if abs(v) >= 1e9:
                return f"${v/1e9:.2f}B"
            if abs(v) >= 1e6:
                return f"${v/1e6:.1f}M"
            return f"{v:,.0f}"

        display_df = fin.copy()
        st.dataframe(
            display_df.style.format(_fmt_fin, na_rep="—"),
            use_container_width=True,
            height=500,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — HELP & METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_help:
    st.subheader("How to use this tool")

    st.markdown("""
    <div class="rvm-callout">
      <h4>Quick Start (60 seconds)</h4>
      <p>
        <b>1.</b> Type a US-listed ticker (e.g. <code>AAPL</code>, <code>NVDA</code>,
        <code>META</code>) in the left sidebar.<br>
        <b>2.</b> Set your macro assumptions (defaults are sensible).<br>
        <b>3.</b> Click <b>⚡ ANALYSE</b>. Use <b>🔄</b> next to it to force-pull
        a fresh price from yfinance (bypasses the in-process cache).<br>
        <b>4.</b> Read the four tabs left-to-right.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### What each tab shows")
    st.markdown("""
- **🤖 Reverse-DCF (Auto)** — the core feature. Given today's enterprise value,
  the engine **solves backward** for the revenue CAGR the market is pricing in.
  This is not a forecast; it's an *implicit assumption* baked into the share price.
- **🎛️ Manual Sensitivity** — you supply your own assumptions (growth, margin,
  WACC, terminal g). The engine computes an intrinsic per-share value and shows
  margin of safety vs. the live market price.
- **⚖️ Rationality Check** — statistical test. We compare the implied CAGR
  against the firm's own historical YoY growth distribution and flag the gap
  as *Speculative*, *Elevated*, *Rational*, *Cautious*, or *Pessimistic*.
- **📂 Fundamentals** — raw 10-K data pulled from SEC EDGAR XBRL. Trust the
  source: no third-party scraping.
""")

    st.markdown("### Header cards — what the six top-row numbers mean")
    st.markdown("""
| Card | Meaning |
|------|---------|
| **Price** | Latest trade price from yfinance |
| **Market Cap** | Shares outstanding × Price |
| **Net Debt** | Long-term + Short-term debt − Cash & equivalents (latest 10-K) |
| **Enterprise Value** | Market Cap + Net Debt — this is what we discount cash flows to |
| **Beta (β)** | From yfinance; falls back to 5-yr monthly regression vs. S&P 500 |
| **WACC (est.)** | Auto-derived from CAPM equity cost + after-tax debt cost, book-value weighted |
""")

    st.markdown("### Where the *intrinsic value* in Tab 2 comes from")
    st.markdown(r"""
The Manual Sensitivity tab runs a two-stage **Forward DCF** on the assumptions
you set:

1. **Project annual Free Cash Flow to Firm (FCFF)** for each year of the horizon:
   $$
   FCFF_t = Rev_t \times OpMargin \times (1 - Tax) - Rev_t \times CapExPct
   $$
   where revenue grows at your chosen rate: $Rev_t = Rev_{t-1} \times (1+g)$.

2. **Discount** each FCFF to present value at WACC:
   $$
   PV(FCFF_t) = \frac{FCFF_t}{(1 + WACC)^t}
   $$

3. **Add a Gordon Growth terminal value** at the end of the horizon:
   $$
   TV = \frac{FCFF_N \times (1 + g_T)}{WACC - g_T}, \qquad PV(TV) = \frac{TV}{(1+WACC)^N}
   $$

4. **Enterprise Value** = sum of PVs.
   **Equity Value** = EV − Net Debt.
   **Per-share Intrinsic Value** = Equity / Shares Outstanding.

5. **Margin of Safety** = (Intrinsic − Market Price) / Market Price.
   Positive → undervalued on your assumptions, negative → overvalued.

Code: `modules/forward_dcf.py :: intrinsic_value()`
""")

    st.markdown("### Where the *implied growth* in Tab 1 comes from")
    st.markdown(r"""
The Reverse-DCF inverts the same DCF. We hold WACC, FCF-margin, terminal g,
and net debt **fixed**, and ask:

> *What value of $g$ (revenue CAGR) makes the DCF equal today's Enterprise Value?*

Numerically:
$$
EV \;=\; \sum_{t=1}^{N} \frac{Rev_0 (1+g)^t \cdot fcfMargin}{(1+WACC)^t}
       + \frac{Rev_0 (1+g)^N \cdot fcfMargin \cdot (1+g_T)}{(WACC - g_T)(1+WACC)^N}
$$

We solve for $g$ using **Brent's method** (`scipy.optimize.brentq`) over
progressively wider brackets ([-30%, +50%] → [-40%, +70%] → [-50%, +90%]).

Code: `modules/reverse_dcf.py :: solve_implied_growth()`
""")

    st.markdown("### The Rationality Check (Tab 3)")
    st.markdown(r"""
We compute the **z-score** of the implied CAGR against the firm's historical
annual YoY revenue growth distribution:
$$
z \;=\; \frac{g_{\text{implied}} - \mu_{\text{YoY}}}{\sigma_{\text{YoY}}}
$$

| z-score | Verdict | Meaning |
|---------|---------|---------|
| z > +2.0 | **Speculative** | Market demands growth far beyond historical track record |
| +1.0 < z ≤ +2.0 | **Elevated** | Above average, but within plausible range |
| -1.0 ≤ z ≤ +1.0 | **Rational** | Implied growth sits inside the historical band |
| -2.0 ≤ z < -1.0 | **Cautious** | Market is more conservative than history |
| z < -2.0 | **Pessimistic** | Market pricing in a steep deceleration |

Code: `modules/rationality.py :: assess()`
""")

    st.markdown("---")
    # ── Bottom: Logic section (per user request) ─────────────────────────────
    st.markdown("## Logic — How the whole pipeline works")

    st.markdown("""
The app does **four** things every time you click ANALYSE. Here is the full
pipeline, in order:
""")

    st.markdown("""
**Step 1 — Resolve the ticker on SEC EDGAR.**
`modules/edgar.py :: resolve_ticker()` hits the SEC EDGAR ticker→CIK map. If
the symbol isn't a US-listed filer, we stop with an error.

**Step 2 — Pull 10 years of annual fundamentals from SEC XBRL.**
`edgar.fetch_company_facts(cik)` pulls every reported concept (Revenues,
NetIncome, CashFromOps, CapitalExpenditures, LongTermDebt, ShortTermDebt,
CashAndEquivalents, IncomeTaxExpense, IncomeLoss, etc.) directly from the
EDGAR XBRL endpoint. No third-party scraping; the data is straight from
the company's own filings.

**Step 3 — Pull live market data from yfinance.**
`modules/market_data.py :: fetch_market_snapshot(ticker)` returns: price,
market cap, shares outstanding, beta, sector tags. The 🔄 button in the
sidebar clears the in-process cache so the next call re-hits yfinance.
If a value comes back stale, that's because the underlying yfinance
endpoint is rate-limited or temporarily unavailable — the freshness badge
at the top will turn amber.

**Step 4 — Derive every input the DCF needs, then run both engines.**

*Cost of capital (WACC):*
We estimate the after-tax weighted cost of capital from book-value weights.
""")

    st.latex(r"""
    \begin{aligned}
      k_e \,&=\, R_f + \beta \cdot ERP \\
      k_d^{\,at} \,&=\, k_d^{\,pre} \cdot (1 - \tau) \\
      w_d \,&=\, \frac{TotalDebt}{TotalDebt + MarketCap} \\
      WACC \,&=\, (1 - w_d)\,k_e + w_d\,k_d^{\,at}
    \end{aligned}
    """)

    st.markdown("""
*Effective tax rate:* 3-year median of `IncomeTaxExpense / PreTaxIncome`,
clamped to the [10%, 45%] band so one bad year doesn't poison the result.

*FCF margin* (the steady-state ratio of free cash flow to revenue):
We try **two derivations** and pick the more reliable one:

1. **CFO-based** — 3-year median of `(CFO − CapEx) / Revenue`. Preferred
   when the firm has clean cash-flow statements.
2. **NOPAT-based fallback** — `EBIT × (1 − tax) × (1 − reinvest_rate) / Rev`,
   where reinvest_rate ≈ CapEx / NOPAT, capped at 80%. Used when CFO data
   is sparse or negative.

If both fall below 1%, we use a 10% default and **flag a warning** on the
Auto tab — because a sub-1% derived FCF margin almost always means the
extraction missed something, and a Reverse-DCF on a near-zero margin will
return nonsense.

*Net debt:* `LongTermDebt + ShortTermDebt − Cash` from the latest 10-K.

*Enterprise Value:* `MarketCap + NetDebt`. This is what we solve to in
the Reverse-DCF, not the equity market cap directly — that's the academically
correct way to handle leverage in a DCF.

*Terminal Value % diagnostic:* When the Reverse-DCF converges, we
report what fraction of EV comes from the terminal period vs. the
explicit-period cash flows. If TV% > 75%, we flag a warning — the
valuation is essentially entirely a bet on terminal growth, and small
changes in `g_T` materially shift the implied CAGR.

**Step 5 — Two parallel solvers run.**

| Engine | File | Question it answers |
|--------|------|---------------------|
| Reverse-DCF | `modules/reverse_dcf.py` | What CAGR does the current price imply? |
| Forward DCF | `modules/forward_dcf.py` | What is the per-share value at *my* assumptions? |
| Rationality | `modules/rationality.py` | Is the implied CAGR realistic vs. history? |

The Reverse-DCF uses Brent's root-finder with three progressively wider
brackets so it converges even for extreme valuations. The Rationality
engine uses both the YoY growth distribution (z-score) and the 3-yr / 5-yr
realised CAGRs as cross-checks.
""")

    st.markdown("### Advanced usage — getting the most out of every tab")
    st.markdown("""
**Workflow A — "Is this stock priced rationally?"**
1. Run **Reverse-DCF (Auto)**. Note the implied CAGR and z-score.
2. Open **Rationality Check**. Read the verdict and the histogram — is the implied
   rate in the fat part of the historical distribution or in the tail?
3. If Speculative (z > +2σ): run **Monte Carlo** to see the *probability* the stock
   is fairly priced even in the optimistic scenario. If P(undervalued) < 20%,
   the setup is asymmetric against you.
4. Run **Earnings Intel** on the latest call. Is management language consistent with
   the growth rate the market is pricing in? Optimistic tone + speculative implied
   CAGR = risk of narrative collapse.

**Workflow B — "What is the stock worth to me?"**
1. Open **Manual Sensitivity**. Set your own growth, margin, WACC.
2. Check the Margin of Safety card. Is your intrinsic > market price?
3. Cross-check your assumed CAGR against the **Reverse-DCF** implied CAGR.
   If they're far apart, articulate *why* your view differs from the market.
4. Run **Monte Carlo** with your assumptions as the distribution centre —
   see what range of values your thesis implies.

**Workflow C — "Screen before deep-diving"**
- Use the implied CAGR as a quick filter. A company priced at +60% CAGR implied
  needs exceptional research to justify. One at −5% implied may be a turn-around
  play with limited downside in the DCF.

**Interpreting edge cases:**
- **Solver fails to converge** → Usually means FCF margin is near-zero (the company
  generates little cash), or WACC ≤ terminal growth (invalid math). Fix by
  raising the FCF margin override or lowering terminal growth below WACC − 1%.
- **TV% = 90%+** → Almost all value is in the terminal period. The implied CAGR
  becomes extremely sensitive to terminal-g. Use the sensitivity heatmap to stress-test.
- **Negative implied CAGR** → Market is pricing in revenue contraction. This often
  signals secular decline concerns. Check the Rationality Check — if Pessimistic
  (z < −2σ), the market may be overdoing the pessimism.
- **Beta = 0 or very low** → yfinance returned unreliable beta; WACC is understated.
  Override with sector median beta (e.g. ~1.2 for large-cap tech, ~0.7 for utilities).

**FCF margin override guide (when auto-derivation fails):**
| Sector              | Typical FCF/Rev  |
|---------------------|-----------------|
| Software / SaaS     | 20 – 35%         |
| Large-cap tech      | 18 – 28%         |
| Consumer staples    | 8 – 14%          |
| Industrials         | 5 – 10%          |
| Retail              | 2 – 6%           |
| Capital-intensive   | 2 – 8%           |
| Early-stage / SaaS loss-making | Use NOPAT method |

**Monte Carlo interpretation guide:**
- **P(undervalued) > 70%** → Strong bull case across the parameter space.
- **P(undervalued) 40–70%** → Thesis-dependent; conviction in WACC or growth matters.
- **P(undervalued) < 30%** → Bear case dominates. Stock needs to justify premium.
- The **tornado chart** tells you where to focus: if Revenue Growth drives the widest
  bar, your key research question is whether the company can sustain its growth rate.
  If WACC drives it, interest-rate sensitivity is the main risk.

**Earnings Intel interpretation guide:**
- **Tone score > 0.3** (Bullish) is common for beats; below 0.0 often precedes
  guidance cuts. The *change* in tone quarter-over-quarter matters more than the
  level alone.
- **Uncertainty ratio > 60 per 1k words** signals management has low visibility —
  expect wider guidance ranges and possible mis-estimation.
- **Sentiment timeline** reveals if the Q&A session was more or less optimistic than
  prepared remarks — management typically scripts positive language but gives real
  signal during analyst questions.
- The **academic benchmark** (Loughran & McDonald, 2011) found that negative-word
  frequency in 10-Ks is significantly correlated with contemporaneous excess returns
  and future volatility.
""")

    st.markdown("### Caveats & limitations")
    st.markdown("""
- **Price data depends on yfinance**, which depends on Yahoo's public endpoint
  being current. Use the 🔄 button for a fresh pull. If a price looks stale,
  the freshness badge turns amber.
- **FCF margin is the single most influential lever** in any DCF. A 5 pp error in
  FCF margin can shift the implied CAGR by 2–4 pp. Always verify the auto-derived
  margin against the company's own reported free cash flow disclosures.
- **Reverse-DCF is descriptive, not prescriptive.** It tells you what growth the
  market is currently pricing in — not what growth *should* be priced in. The
  Rationality Check is a starting hypothesis, not an oracle.
- **No analyst consensus is used anywhere.** The implied CAGR is derived purely
  from price + fundamentals + your WACC/terminal-g choices. This is a deliberate
  design choice: analyst estimates often embed herding bias.
- **Monte Carlo assumes independence of inputs.** In practice, high-growth
  scenarios often coincide with high-multiple (lower WACC) environments. The
  simulation does not model this co-movement, so the P(undervalued) metric
  should be interpreted as an *order-of-magnitude* guide, not a precise probability.
- **Earnings transcript availability varies.** Companies that file earnings call
  transcripts as 8-K exhibits are covered automatically. Others require manual paste.
  The NLP analysis is only as good as the text quality.
""")

    st.markdown("---")
    st.markdown("## Logic — Complete Mathematical Framework")

    st.markdown("### Step 1 — Ticker resolution & fundamentals (SEC EDGAR XBRL)")
    st.markdown("""
`edgar.resolve_ticker(ticker)` maps the symbol to a CIK via the SEC's public
company index. `fetch_company_facts(cik)` then pulls the XBRL concept database
(`https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`) — every number the
company has ever reported to the SEC, tagged by concept (Revenues, NetIncome,
CashFromOps, CapEx, LongTermDebt, etc.). We extract 10 years of annual figures
and build a clean pivot DataFrame indexed by fiscal year.
""")

    st.markdown("### Step 2 — Live market data (yfinance)")
    st.markdown("""
`market_data.fetch_market_snapshot(ticker)` calls Yahoo Finance for: current
price, market cap, shares outstanding, beta. Fallback hierarchy:
1. `info["currentPrice"]` → `info["regularMarketPrice"]` → `fast_info.last_price`
2. 5-day price history if all fields missing
3. Beta: `info["beta"]` → 5-year monthly regression against S&P 500 (^GSPC)
4. Market cap: `info["marketCap"]` → price × shares_outstanding
The 🔄 sidebar button calls `.cache_clear()` on all yfinance LRU caches.
""")

    st.markdown("### Step 3 — WACC derivation")
    st.latex(r"""
    \begin{aligned}
      k_e \;&=\; R_f \;+\; \beta \cdot ERP \\[4pt]
      k_d^{\,at} \;&=\; k_d^{\,pre} \cdot (1 - \tau_{\text{eff}}) \\[4pt]
      w_d \;&=\; \frac{\text{TotalDebt}}{\text{TotalDebt} + \text{MarketCap}} \\[4pt]
      WACC \;&=\; (1 - w_d)\,k_e \;+\; w_d\,k_d^{\,at}
    \end{aligned}
    """)
    st.markdown("""
Book-value weights are used (total debt from the balance sheet) rather than
market-value weights, which avoids the circularity problem in iterative WACC
estimation. Effective tax rate τ is the 3-year median of
IncomeTaxExpense / PreTaxIncome, capped to the [10%, 45%] band.
""")

    st.markdown("### Step 4 — FCF Margin derivation (dual method)")
    st.markdown(r"""
**Method A — Cash-flow statement basis (preferred):**
$$
\text{FCF Margin} = \text{median}_{3\text{yr}}\!\left(\frac{CFO - CapEx}{Revenue}\right)
$$
Used when the 3-year median exceeds 1%.

**Method B — NOPAT basis (fallback):**
$$
\text{FCF Margin} = \text{median}_{3\text{yr}}\!\left(\frac{EBIT \cdot (1-\tau)}{Revenue}\right)
\times (1 - \text{ReinvestRate})
$$
where $\text{ReinvestRate} = \min\!\left(\frac{CapEx}{NOPAT},\; 80\%\right)$.

If both methods yield < 1%, the 10% generic default is applied with a warning banner.
The FCF margin is the steady-state ratio of free cash to revenue that the DCF
rolls forward at the implied/assumed growth rate.
""")

    st.markdown("### Step 5 — Reverse-DCF: solving for implied revenue CAGR")
    st.latex(r"""
    EV = \sum_{t=1}^{N} \frac{Rev_0 (1+g)^t \cdot m}{(1+WACC)^t}
         \;+\; \frac{Rev_0 (1+g)^N \cdot m \cdot (1+g_T)}{(WACC - g_T)(1+WACC)^N}
    """)
    st.markdown(r"""
where $m$ = FCF margin, $g_T$ = terminal growth, $N$ = forecast horizon, and
$EV = \text{MarketCap} + \text{NetDebt}$.

We solve $DCF(g) - \text{MarketCap} = 0$ using **Brent's method** with three
progressively wider brackets: $[-30\%, +50\%]$, $[-40\%, +70\%]$, $[-50\%, +90\%]$.
Tolerance $\varepsilon = 10^{-6}$, max 400 iterations. The bracket list ensures
convergence even for extreme valuations (hypergrowth tech or deep-value restructuring).

`modules/reverse_dcf.py :: solve_implied_growth()`
""")

    st.markdown("### Step 6 — Rationality Check")
    st.latex(r"""
    z \;=\; \frac{g_{\text{implied}} - \mu_{\text{YoY}}}{\sigma_{\text{YoY}}}
    """)
    st.markdown("""
| z | Verdict | Colour |
|---|---------|--------|
| z > +2.0 | **Speculative** — far above historical track record | Red |
| +1.0 < z ≤ +2.0 | **Elevated** | Amber |
| −1.0 ≤ z ≤ +1.0 | **Rational** | Green |
| −2.0 ≤ z < −1.0 | **Cautious** | Blue |
| z < −2.0 | **Pessimistic** — market pricing severe deceleration | Amber |

Secondary reference points: 3-yr and 5-yr realised CAGRs (more stable than
YoY mean for volatile revenue series). `modules/rationality.py :: assess()`
""")

    st.markdown("### Step 7 — Forward DCF (Manual Sensitivity tab)")
    st.latex(r"""
    FCFF_t \;=\; Rev_t \cdot OpMargin \cdot (1-\tau) - Rev_t \cdot \text{CapExPct}
    \qquad Rev_t = Rev_0 (1+g)^t
    """)
    st.latex(r"""
    IV_{\text{equity}} = \sum_{t=1}^{N}\frac{FCFF_t}{(1+WACC)^t}
      + \frac{FCFF_N (1+g_T)}{(WACC-g_T)(1+WACC)^N} - \text{NetDebt}
    """)
    st.markdown("""
Per-share intrinsic value = IV_equity / Shares Outstanding.
Margin of Safety = (Intrinsic − Price) / Price.
`modules/forward_dcf.py :: intrinsic_value()`
""")

    st.markdown("### Step 8 — Monte Carlo simulation")
    st.markdown(r"""
Each of the $N$ simulations draws independent samples:
$$
g \sim \mathcal{N}(\mu_g, \sigma_g), \quad
WACC \sim \mathcal{N}(\mu_w, \sigma_w), \quad
m_{op} \sim \mathcal{N}(\mu_{op}, \sigma_{op}), \quad
g_T \sim \mathcal{N}(\mu_T, \sigma_T)
$$
clipped to prevent economically impossible values.

**Forward MC** uses a vectorised NumPy implementation (entire $N \times T$ revenue
and FCFF matrix computed in one call) — fast even for 5 000 draws.

**Reverse MC** loops Brent's solver for each (WACC, FCF margin, terminal-g) draw
at the fixed market price — reveals the distribution of implied CAGRs consistent
with current market pricing.

**OAT Tornado** varies each input by ±1σ while holding others at their mean.
The width of each bar = magnitude of IV swing = driver ranking of uncertainty.

`modules/montecarlo.py :: run_monte_carlo()`
""")

    st.markdown("### Step 9 — Earnings call NLP (Loughran-McDonald lexicon)")
    st.markdown("""
Text is tokenized (lowercase, punctuation stripped) and matched against four
curated word lists from the Loughran-McDonald Master Dictionary (2011):

| Category | What it measures |
|----------|-----------------|
| **Positive** | Confident, constructive language (achieve, growth, record…) |
| **Negative** | Risk, deterioration, failure language (decline, lawsuit, loss…) |
| **Uncertainty** | Hedging, imprecision (could, may, approximately, expect…) |
| **Litigious** | Legal-risk signals (lawsuit, violation, judgment…) |

**Tone score** = (Positive − Negative) / (Positive + Negative), ∈ [−1, +1].

**Sentiment timeline** splits the text into sequential 200-word chunks and
scores each — revealing whether optimism peaks in prepared remarks or Q&A.

**Topic detection** counts keyword hits for revenue, margin, guidance, AI/cloud,
macro, competition, cost, CapEx, cash, and workforce.

**Academic validation:** Loughran & McDonald (2011) showed that the negative-word
frequency in 10-Ks is significantly negatively correlated with contemporaneous
excess stock returns and positively correlated with future return volatility.
The same lexicon applied to earnings calls is widely used in empirical asset
pricing and accounting research (e.g., Brochet et al., 2015; Huang et al., 2014).

`modules/earnings.py :: analyze_transcript(), fetch_edgar_transcript()`
""")
