"""Rationality vs. Market (RvM) — Reverse-DCF Forecasting Tool.

Core question: What revenue growth rate is the market currently pricing in?

Auto mode  — given market price, solve for the implied growth rate (Reverse-DCF).
Manual mode — given your own growth assumptions, compute intrinsic value.
Rationality — flag whether the implied rate is Speculative / Rational / Pessimistic
              vs. the company's own historical realised growth distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from modules import edgar, market_data, visuals
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
  padding: 11px 15px;
  flex: 1;
  min-width: 120px;
}
.rvm-card-label {
  color: #6b7280;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  margin-bottom: 4px;
}
.rvm-card-value {
  color: #e6edf3;
  font-size: 19px;
  font-weight: 700;
  line-height: 1.2;
}
.rvm-card-sub {
  color: #9ca3af;
  font-size: 10px;
  margin-top: 3px;
}
.rvm-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin: 8px 0 16px 0;
}

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
    """Format a dollar value in billions."""
    if abs(v) >= 1e12:
        return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:.1f}B"
    return f"${v/1e6:.0f}M"


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
    run = st.button("⚡  ANALYSE", type="primary", use_container_width=True)

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

ticker_info = edgar.resolve_ticker(active_ticker)
if not ticker_info:
    st.error(f"Ticker **{active_ticker}** not found in SEC EDGAR. Check the symbol and try again.")
    st.stop()

with st.spinner(f"Pulling SEC EDGAR filings & market data for {active_ticker}…"):
    facts = edgar.fetch_company_facts(ticker_info["cik"])
    fin   = facts.build_financials(years=10)
    snap  = market_data.fetch_market_snapshot(active_ticker)


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

# ── Page header ───────────────────────────────────────────────────────────────
st.title(f"📡 {active_ticker}  —  Rationality vs. Market")
st.caption(snap.long_name or ticker_info.get("title", ""))

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
tab_auto, tab_manual, tab_compare, tab_data = st.tabs([
    "🤖  Reverse-DCF (Auto)",
    "🎛️  Manual Sensitivity",
    "⚖️  Rationality Check",
    "📂  Fundamentals",
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

    # Default revenue CAGR from history
    _hist_cagr = 0.05
    if len(revenue) > 1 and float(revenue.iloc[0]) > 0:
        _r = float(revenue.iloc[-1]) / float(revenue.iloc[0])
        _hist_cagr = _r ** (1.0 / (len(revenue) - 1)) - 1.0
    _hist_cagr = float(np.clip(_hist_cagr, -0.05, 0.25))

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
# TAB 4 — FUNDAMENTALS
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
