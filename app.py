"""RvM — Rationality vs. Market Forecast Terminal.

A Streamlit app that asks one question of every US-listed company:
is today's price rationally justified by forward fundamentals,
or is it pricing in a speculative growth premium?

Data backbone: SEC EDGAR XBRL (free, no key) for fundamentals,
yfinance for live price + beta + sector + analyst consensus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from modules import edgar, market_data
from modules import visualizations as viz
from modules.dcf import (
    WaccInputs, ReverseDcfInputs, IntrinsicInputs,
    reverse_dcf, intrinsic_dcf, sensitivity_grid, fair_value_curve,
    monte_carlo_implied_growth,
)
from modules.rationality import rationality_check

# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title="RvM Forecast Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CSS — Bloomberg dark aesthetic
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', 'IBM Plex Mono', ui-monospace, monospace !important;
}
* { -webkit-font-smoothing: antialiased; }

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at top, #0e1118 0%, #0b0d10 80%) !important;
}
.block-container { padding-top: 1.4rem !important; padding-bottom: 4rem !important; max-width: 1480px; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(180deg, #161b24 0%, #111620 100%);
    border: 1px solid #1f2b3e; border-radius: 12px; padding: 14px 16px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 6px 18px rgba(0,0,0,0.25);
    transition: border-color .2s ease;
}
[data-testid="metric-container"]:hover { border-color: #ffb00055; }
[data-testid="metric-container"] label {
    color: #7d8693 !important; font-size: 0.7rem !important;
    text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #d8dde3 !important; font-size: 1.5rem !important;
    font-weight: 700 !important; letter-spacing: -0.01em;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.76rem !important; font-weight: 600 !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
    border-bottom: 1px solid #1f2b3e !important; gap: 2px;
    background: transparent !important;
}
[data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important; padding: 10px 16px !important;
    color: #7d8693 !important; font-weight: 500 !important;
    font-size: 0.86rem !important; transition: all .15s ease;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #ffb000 !important; border-bottom: 2px solid #ffb000 !important;
    background: linear-gradient(180deg, transparent, #ffb00010) !important;
    font-weight: 700 !important;
}
[data-baseweb="tab"]:hover { color: #d8dde3 !important; background: #161b24 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090b0f 0%, #0b0d10 100%) !important;
    border-right: 1px solid #1f2b3e;
}

/* Buttons */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #ffb000 0%, #e09800 100%) !important;
    color: #0b0d10 !important; border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: 0.85rem !important;
    box-shadow: 0 4px 14px rgba(255,176,0,0.35) !important;
}
[data-testid="baseButton-primary"]:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(255,176,0,0.5) !important; }
[data-testid="baseButton-secondary"] {
    background: #161b24 !important; border: 1px solid #1f2b3e !important;
    border-radius: 8px !important; color: #d8dde3 !important;
}

/* Inputs */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #090b0f !important; border: 1px solid #1f2b3e !important;
    border-radius: 6px !important; color: #d8dde3 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus { border-color: #ffb000 !important; }
[data-testid="stSlider"] [role="slider"] { background: #ffb000 !important; }

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #1f2b3e; border-radius: 10px; overflow: hidden;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #1f2b3e !important; border-radius: 10px !important;
    background: #111620 !important;
}
[data-testid="stExpander"] summary {
    padding: 12px 16px !important; font-weight: 600 !important;
    color: #d8dde3 !important;
}

/* Alerts */
[data-testid="stAlert"] { border-radius: 8px !important; border-width: 1px !important; }
[data-testid="stAlert"][kind="error"]   { background: #2e0c10 !important; border-color: #ef444455 !important; }
[data-testid="stAlert"][kind="warning"] { background: #2e1f0c !important; border-color: #f59e0b55 !important; }
[data-testid="stAlert"][kind="success"] { background: #0c2e15 !important; border-color: #22c55e55 !important; }
[data-testid="stAlert"][kind="info"]    { background: #0c1e2e !important; border-color: #ffb00055 !important; }

.section-title {
    font-size: 0.78rem; font-weight: 700; color: #7d8693;
    border-left: 3px solid #ffb000; padding-left: 10px;
    margin: 22px 0 12px 0; text-transform: uppercase; letter-spacing: 0.08em;
}

.company-header {
    display: flex; align-items: flex-start; justify-content: space-between;
    background: linear-gradient(135deg, #0e1420 0%, #0b1018 50%, #0d1220 100%);
    border: 1px solid #1f2b3e; border-radius: 14px;
    padding: 18px 24px; margin-bottom: 14px; flex-wrap: wrap; gap: 18px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 1px 0 rgba(255,255,255,0.03) inset;
}
.company-name { font-size: 1.65rem; font-weight: 800; color: #f1f5f9; letter-spacing: -0.02em; line-height: 1.1; }
.ticker-badge {
    color: #ffb000; font-size: 1.05rem; font-weight: 700;
    background: #ffb00015; padding: 2px 10px; border-radius: 6px;
    border: 1px solid #ffb00033; margin-left: 8px;
}
.company-sub { font-size: 0.7rem; color: #5c6680; margin-top: 4px; }
.badge {
    display: inline-block; background: #1a2535; color: #7eb0ff;
    border: 1px solid #2a3f5e; border-radius: 20px;
    padding: 2px 10px; font-size: 0.68rem; font-weight: 600;
    margin-right: 4px; margin-top: 6px; letter-spacing: 0.02em;
}
.kv-block { display: flex; gap: 10px; flex-wrap: wrap; }
.kv-card {
    background: linear-gradient(180deg, #0c1220 0%, #090e18 100%);
    border: 1px solid #1f2b3e; border-radius: 10px;
    padding: 8px 14px; min-width: 120px;
}
.kv-label { color: #7d8693; font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
.kv-value { color: #f1f5f9; font-size: 1.25rem; font-weight: 700; line-height: 1.2; margin-top: 3px; }
.kv-sub { color: #7d8693; font-size: 0.68rem; margin-top: 1px; }

.verdict-card {
    border-radius: 12px; padding: 16px 20px; text-align: center;
    border: 1px solid; min-width: 180px;
}

hr { border-color: #1f2b3e !important; margin: 12px 0 !important; }
.stMarkdown p { color: #d8dde3 !important; }
.stMarkdown code { background: #161b24; color: #27e0c5; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }
.stMarkdown table th { background: #161b24; color: #d8dde3; border: 1px solid #1f2b3e; padding: 8px 12px; }
.stMarkdown table td { border: 1px solid #1f2b3e; padding: 8px 12px; color: #d8dde3; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helpers
# ============================================================================

def _fmt_money(v, na="—"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return na
    v = float(v)
    abs_v = abs(v)
    if abs_v >= 1e12: return f"${v/1e12:,.2f}T"
    if abs_v >= 1e9:  return f"${v/1e9:,.2f}B"
    if abs_v >= 1e6:  return f"${v/1e6:,.2f}M"
    if abs_v >= 1e3:  return f"${v/1e3:,.2f}K"
    return f"${v:,.2f}"

def _fmt_pct(v, dp=2, signed=False, na="—"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return na
    return f"{float(v)*100:{'+' if signed else ''}.{dp}f}%"

def _fmt_num(v, dp=2, na="—"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return na
    return f"{float(v):,.{dp}f}"

def _fmt_int(v, na="—"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return na
    return f"{int(round(float(v))):,}"

def pct_slider(label, val, lo, hi, step=0.25, key=None, help=None):
    v = st.slider(
        label, float(lo * 100), float(hi * 100), float(val * 100),
        step=float(step), format="%.2f%%", key=key, help=help,
    )
    return v / 100.0

def pct_input(label, val, lo=0.0, hi=1.0, step=0.0025, key=None, help=None):
    v = st.number_input(
        label, float(lo * 100), float(hi * 100), float(val * 100),
        step=float(step * 100), format="%.2f", key=key, help=help,
    )
    return v / 100.0

def _format_statement(df: pd.DataFrame) -> pd.DataFrame:
    """Format an as-reported statement with $B values and 2dp."""
    if df.empty: return df
    out = pd.DataFrame(index=df.index, columns=[str(c) for c in df.columns], dtype=object)
    for idx in df.index:
        for col in df.columns:
            v = df.loc[idx, col]
            if pd.isna(v): out.loc[idx, str(col)] = "—"
            elif idx == "EPS": out.loc[idx, str(col)] = f"${v:,.2f}"
            elif abs(v) >= 1e6: out.loc[idx, str(col)] = f"${v/1e9:,.2f}B"
            else: out.loc[idx, str(col)] = f"{v:,.2f}"
    return out


# ============================================================================
# Sidebar
# ============================================================================
st.sidebar.markdown("## 📈 RvM Terminal")
st.sidebar.caption("Rationality vs. Market · SEC EDGAR + yfinance · No API key")
st.sidebar.markdown("---")

ticker = st.sidebar.text_input(
    "🔍 Ticker (US-listed)", value="AAPL",
).strip().upper()
years_hist = st.sidebar.slider(
    "Years of historical data", 5, 10, 7, 1,
    help="Annual 10-K periods to pull from SEC EDGAR.",
)
fetch_btn = st.sidebar.button("⚡ Fetch & Analyse", type="primary", use_container_width=True)

if st.sidebar.button("🗑️ Clear cache"):
    st.cache_data.clear()
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ DCF defaults**")
sb_wacc = pct_slider("WACC", 0.090, 0.04, 0.20, 0.25, key="sb_wacc")
sb_tg   = pct_slider("Terminal g", 0.025, 0.00, 0.05, 0.10, key="sb_tg")
sb_hgy  = st.sidebar.slider("High-growth years", 3, 10, 5, 1, key="sb_hgy")
sb_fy   = st.sidebar.slider("Fade years", 2, 8, 5, 1, key="sb_fy")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How to use**
1. Enter a US ticker
2. Click **Fetch & Analyse**
3. Review the Rationality verdict
4. Adjust assumptions in any tab
5. Browse the full Financial Statements
""")
st.sidebar.markdown("<small>Data: SEC EDGAR XBRL · yfinance · 100% free</small>", unsafe_allow_html=True)


# ============================================================================
# Data loading (cached)
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(sym: str, n_years: int) -> dict:
    info = edgar.resolve_ticker(sym)
    if not info:
        raise ValueError(f"'{sym}' not found in SEC ticker file. Check the symbol — "
                         "only US-listed entities with SEC filings are supported.")
    facts = edgar.fetch_company_facts(info["cik"])
    fin = facts.build_financials(n_years)
    income_stmt = edgar.extract_statement(facts, edgar.INCOME_STATEMENT_ROWS, n_years)
    balance_sheet = edgar.extract_statement(facts, edgar.BALANCE_SHEET_ROWS, n_years)
    cash_flow = edgar.extract_statement(facts, edgar.CASH_FLOW_ROWS, n_years)
    filings = edgar.fetch_recent_filings(info["cik"])
    snap = market_data.fetch_market_snapshot(sym)
    hist = market_data.fetch_price_history(sym, period=f"{min(n_years, 10)}y")
    rf   = market_data.fetch_risk_free_rate()
    return dict(
        cik=info["cik"],
        name=facts.name or info["title"],
        financials=fin,
        income_statement=income_stmt,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        filings=filings,
        snap=snap,
        price_history=hist,
        risk_free_rate=rf,
    )

if fetch_btn:
    for k in ("data", "_loaded_ticker"):
        st.session_state.pop(k, None)

if "data" not in st.session_state or st.session_state.get("_loaded_ticker") != ticker:
    try:
        with st.spinner(f"Fetching **{ticker}** from SEC EDGAR + Yahoo Finance…"):
            st.session_state["data"] = load_data(ticker, years_hist)
            st.session_state["_loaded_ticker"] = ticker
    except Exception as exc:
        st.error(f"**Could not load {ticker}:** {exc}")
        st.stop()

data = st.session_state["data"]
fin: pd.DataFrame = data["financials"]
snap = data["snap"]
hist = data["price_history"]
rf = data["risk_free_rate"]

if fin.empty:
    st.warning("⚠️ No annual US-GAAP fundamentals returned from SEC EDGAR. "
               "This usually means the issuer files in a non-XBRL form (small / foreign filer).")


# ============================================================================
# Compute key inputs
# ============================================================================

# Net debt + FCF history from EDGAR
net_debt = edgar.compute_net_debt(fin) or 0.0
fcf_history = edgar.compute_fcf_history(fin)
latest_fcf = float(fcf_history.iloc[-1]) if not fcf_history.empty else None
fcf_3y = float(fcf_history.tail(3).mean()) if len(fcf_history) >= 3 else latest_fcf
fcf_5y = float(fcf_history.tail(5).mean()) if len(fcf_history) >= 5 else fcf_3y

latest_revenue = edgar.latest_value(fin, "Revenue")
latest_ebit    = edgar.latest_value(fin, "OperatingIncome")
latest_ni      = edgar.latest_value(fin, "NetIncome")
shares = float(snap.shares_outstanding or edgar.latest_value(fin, "SharesOutstanding") or 0)

rev_growth_history = edgar.compute_growth_history(
    fin.loc["Revenue"] if "Revenue" in fin.index else pd.Series(dtype=float)
)
eps_growth_history = edgar.compute_growth_history(
    fin.loc["NetIncome"] if "NetIncome" in fin.index else pd.Series(dtype=float)
)
fcf_growth_history = edgar.compute_growth_history(fcf_history)

# WACC (CAPM)
_wi = WaccInputs(
    beta=float(snap.beta or 1.1),
    risk_free_rate=rf,
    equity_risk_premium=0.055,
    cost_of_debt_pretax=0.05,
    tax_rate=0.21,
    debt_weight=0.20,
)
wacc_default = _wi.wacc
wacc_used = sb_wacc if abs(sb_wacc - 0.09) > 1e-4 else max(wacc_default, 0.06)

# Reverse-DCF
rd_result = None
rat_result = None
mc_result = None

if snap.market_cap and fcf_3y and fcf_3y > 0 and shares > 0:
    rd_inp = ReverseDcfInputs(
        fcf_base=fcf_3y, net_debt=net_debt, shares=shares,
        wacc=wacc_used, terminal_g=sb_tg,
        high_growth_years=sb_hgy, fade_years=sb_fy,
    )
    rd_result = reverse_dcf(snap.market_cap, rd_inp)
    rat_result = rationality_check(
        rd_result.implied_growth,
        rev_growth_history.tolist(),
        eps_growth_history.tolist(),
        fcf_growth_history.tolist(),
        snap.analyst_growth_5y,
    )
    fcf_scenarios = {
        "3y avg (primary)": fcf_3y,
        "5y avg": fcf_5y if fcf_5y else fcf_3y,
        "TTM": latest_fcf,
    }
    mc_result = monte_carlo_implied_growth(snap.market_cap, rd_inp, fcf_scenarios)


# ============================================================================
# Header banner (persistent across tabs)
# ============================================================================
badge_html = "".join(
    f'<span class="badge">{b}</span>'
    for b in filter(None, [snap.sector, snap.industry])
)

verdict_card = ""
if rat_result:
    c = rat_result.verdict_color
    verdict_card = f"""
    <div class="kv-card" style="border-color:{c}55;background:linear-gradient(180deg,{c}12,{c}06);min-width:200px;">
      <div class="kv-label" style="color:{c};">Rationality</div>
      <div class="kv-value" style="color:{c};font-size:1.05rem;">{rat_result.emoji} {rat_result.verdict}</div>
      <div class="kv-sub" style="color:{c};">z = {rat_result.z_score:+.2f}σ · implied {_fmt_pct(rat_result.implied_growth)}</div>
    </div>"""

name = data['name']
if snap.long_name and len(snap.long_name) > len(name):
    name = snap.long_name

st.markdown(f"""
<div class="company-header">
  <div style="flex:1;min-width:280px;">
    <div class="company-name">{name}
      <span class="ticker-badge">{ticker}</span>
    </div>
    <div class="company-sub">CIK {data['cik']} · {snap.currency} · SEC EDGAR + yfinance</div>
    <div>{badge_html}</div>
  </div>
  <div class="kv-block">
    <div class="kv-card">
      <div class="kv-label">Last Price</div>
      <div class="kv-value" style="color:#ffb000;">{('$'+format(snap.price, ',.2f')) if snap.price else '—'}</div>
    </div>
    <div class="kv-card">
      <div class="kv-label">Market Cap</div>
      <div class="kv-value">{_fmt_money(snap.market_cap)}</div>
    </div>
    <div class="kv-card">
      <div class="kv-label">FCF (3y avg)</div>
      <div class="kv-value">{_fmt_money(fcf_3y)}</div>
    </div>
    {verdict_card}
  </div>
</div>
""", unsafe_allow_html=True)

# KPI strip
ev = (snap.market_cap or 0) + net_debt
ev_ebit = (ev / latest_ebit) if (latest_ebit and latest_ebit > 0) else None
ev_sales = (ev / latest_revenue) if (latest_revenue and latest_revenue > 0) else None
price_to_fcf = (snap.price / (fcf_3y / shares)) if (snap.price and fcf_3y and shares) else None

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Trailing P/E",  _fmt_num(snap.trailing_pe, 1))
c2.metric("Forward P/E",   _fmt_num(snap.forward_pe, 1))
c3.metric("P / FCF",        _fmt_num(price_to_fcf, 1))
c4.metric("EV / EBIT",     _fmt_num(ev_ebit, 1))
c5.metric("EV / Sales",    _fmt_num(ev_sales, 2))
c6.metric("WACC (CAPM)",   _fmt_pct(wacc_used))

st.markdown("---")


# ============================================================================
# Tabs
# ============================================================================
tab_ov, tab_rev, tab_man, tab_stmt, tab_viz, tab_guide = st.tabs([
    "📊 Overview",
    "🔄 Reverse DCF",
    "🎯 Manual DCF",
    "📑 Financial Statements",
    "📈 Visualisations",
    "📖 Guide",
])


# ---------------------------------------------------------------------------
# TAB 1 — Overview
# ---------------------------------------------------------------------------
with tab_ov:
    st.markdown('<div class="section-title">Snapshot</div>', unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Revenue (latest FY)",  _fmt_money(latest_revenue))
    g2.metric("EBIT (latest FY)",     _fmt_money(latest_ebit))
    g3.metric("Net Income",            _fmt_money(latest_ni))
    g4.metric("Net Debt",              _fmt_money(net_debt),
              delta="net cash" if net_debt < 0 else None,
              delta_color="inverse" if net_debt < 0 else "normal")

    g5, g6, g7, g8 = st.columns(4)
    g5.metric("FCF — latest FY",       _fmt_money(latest_fcf))
    g6.metric("FCF — 3y avg",          _fmt_money(fcf_3y))
    g7.metric("FCF — 5y avg",          _fmt_money(fcf_5y))
    g8.metric("Beta",                  _fmt_num(snap.beta, 2))

    if rat_result:
        st.markdown('<div class="section-title">Rationality verdict</div>', unsafe_allow_html=True)

        cc1, cc2 = st.columns([1, 2])
        with cc1:
            c = rat_result.verdict_color
            st.markdown(f"""
            <div class="verdict-card" style="background:{c}14;border-color:{c}55;">
                <div style="font-size:2.6rem;">{rat_result.emoji}</div>
                <div style="font-size:1.15rem;font-weight:800;color:{c};margin-top:4px;">{rat_result.verdict}</div>
                <div style="font-size:0.72rem;color:#7d8693;margin-top:6px;">z = {rat_result.z_score:+.2f}σ</div>
                <div style="font-size:0.7rem;color:#7d8693;margin-top:2px;">{rat_result.historical_n} historical observations</div>
            </div>""", unsafe_allow_html=True)
        with cc2:
            st.markdown(f"**{rat_result.rationale}**")
            if rat_result.analyst_delta is not None:
                direction = "more" if rat_result.analyst_delta > 0 else "less"
                st.markdown(
                    f"· Analyst consensus is **{_fmt_pct(rat_result.analyst_growth)}** "
                    f"— the market is pricing **{_fmt_pct(abs(rat_result.analyst_delta))}** {direction} growth than analysts."
                )
            st.plotly_chart(viz.growth_comparison_bar(rat_result), use_container_width=True)

        if mc_result and mc_result.scenario_growths:
            st.markdown('<div class="section-title">FCF-base sensitivity (Monte-Carlo)</div>', unsafe_allow_html=True)
            st.caption("Re-running the reverse-DCF over multiple FCF-base scenarios shows "
                       "how sensitive the implied growth rate is to your FCF normalisation choice.")
            mc_rows = [
                {"Scenario": k, "FCF base": _fmt_money(
                    {"3y avg (primary)": fcf_3y, "5y avg": fcf_5y, "TTM": latest_fcf}.get(k)
                ), "Implied growth": _fmt_pct(v)}
                for k, v in mc_result.scenario_growths.items()
            ]
            mc_rows.append({"Scenario": "— Mean ± 1σ —",
                            "FCF base": "",
                            "Implied growth":
                                f"{_fmt_pct(mc_result.mean)}  ±  {_fmt_pct(mc_result.std)}"})
            st.dataframe(pd.DataFrame(mc_rows), use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ Reverse-DCF needs positive 3-year-average FCF and shares outstanding. "
                "This ticker may have negative or non-disclosed FCF — use **Manual DCF** for a revenue-driven valuation.")

    # Recent filings
    st.markdown('<div class="section-title">Recent SEC filings</div>', unsafe_allow_html=True)
    filings = data["filings"]
    if isinstance(filings, pd.DataFrame) and not filings.empty:
        shown = filings.head(15).rename(columns={
            "form": "Form", "filingDate": "Filed", "accessionNumber": "Accession",
        })
        st.dataframe(
            shown[["Form", "Filed", "Accession", "url"]],
            use_container_width=True, hide_index=True,
            column_config={
                "url": st.column_config.LinkColumn("Document", display_text="Open ↗"),
                "Form": st.column_config.TextColumn(width="small"),
                "Filed": st.column_config.TextColumn(width="small"),
            },
        )
    else:
        st.caption("No recent filings found.")


# ---------------------------------------------------------------------------
# TAB 2 — Reverse DCF
# ---------------------------------------------------------------------------
with tab_rev:
    st.markdown('<div class="section-title">Auto-predictive — reverse-DCF engine</div>', unsafe_allow_html=True)
    st.caption("Solves for the high-growth rate gʰ that equates the two-stage DCF enterprise value "
               "with today's observed enterprise value (market cap + net debt). "
               "Bisection over [-30%, +200%] to 1e-7 precision.")

    if not (snap.market_cap and shares):
        st.error("❌ Need a market cap and shares outstanding to run reverse-DCF.")
    elif not (fcf_3y and fcf_3y > 0):
        st.error("❌ FCF is zero or negative. Reverse-DCF requires positive free cash flow. "
                 "Use the **Manual DCF** tab with a revenue-driven model instead.")
    else:
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            r_wacc = pct_slider("WACC", wacc_used, 0.04, 0.20, 0.25, key="r_wacc")
        with rc2:
            r_tg = pct_slider("Terminal g", sb_tg, 0.0, 0.05, 0.10, key="r_tg")
        with rc3:
            r_hgy = st.slider("High-growth years", 3, 10, sb_hgy, 1, key="r_hgy")
        with rc4:
            r_fy = st.slider("Fade years", 2, 8, sb_fy, 1, key="r_fy")

        rc5, rc6 = st.columns([1, 2])
        with rc5:
            r_fcf_mode = st.selectbox(
                "FCF base",
                ["3y average (primary)", "5y average", "TTM only", "Manual"],
                key="r_fcf_mode",
            )
        if r_fcf_mode == "3y average (primary)":
            fcf_use = fcf_3y
        elif r_fcf_mode == "5y average":
            fcf_use = fcf_5y if fcf_5y else fcf_3y
        elif r_fcf_mode == "TTM only":
            fcf_use = latest_fcf
        else:
            with rc6:
                fcf_use = st.number_input(
                    "Manual FCF override ($)",
                    value=float(fcf_3y or 1e9), step=1e8, format="%.0f", key="r_fcf_man",
                )

        if not fcf_use or fcf_use <= 0:
            st.error("❌ Selected FCF base is non-positive.")
        else:
            rd_inp2 = ReverseDcfInputs(
                fcf_base=fcf_use, net_debt=net_debt, shares=shares,
                wacc=r_wacc, terminal_g=r_tg,
                high_growth_years=r_hgy, fade_years=r_fy,
            )
            rd2 = reverse_dcf(snap.market_cap, rd_inp2)
            rat2 = rationality_check(
                rd2.implied_growth,
                rev_growth_history.tolist(),
                eps_growth_history.tolist(),
                fcf_growth_history.tolist(),
                snap.analyst_growth_5y,
            )

            st.markdown("---")
            ra1, ra2, ra3, ra4 = st.columns(4)
            ra1.metric("Implied growth", _fmt_pct(rd2.implied_growth))
            ra2.metric("Analyst consensus", _fmt_pct(rat2.analyst_growth))
            ra3.metric("Historical mean", _fmt_pct(rat2.historical_mean))
            ra4.metric("Historical σ", _fmt_pct(rat2.historical_std))

            rb1, rb2, rb3, rb4 = st.columns(4)
            rb1.metric("z-score", f"{rat2.z_score:+.2f}σ")
            rb2.metric("Verdict", f"{rat2.emoji} {rat2.verdict}")
            rb3.metric("EV target", _fmt_money(rd2.ev_target))
            rb4.metric("Solver converged", "✅" if rd2.converged else "❌")

            st.info(rat2.rationale)

            with st.expander("📌 Reverse-DCF assumptions used"):
                arows = [
                    ("FCF base",          _fmt_money(fcf_use)),
                    ("Net debt",           _fmt_money(net_debt)),
                    ("Shares outstanding", _fmt_int(shares)),
                    ("WACC",               _fmt_pct(r_wacc)),
                    ("Terminal g",         _fmt_pct(r_tg)),
                    ("High-growth years",  str(r_hgy)),
                    ("Fade years",         str(r_fy)),
                    ("Market cap",         _fmt_money(snap.market_cap)),
                    ("Enterprise value",   _fmt_money(rd2.ev_target)),
                    ("Solver iterations",  str(rd2.iterations)),
                ]
                st.dataframe(
                    pd.DataFrame(arows, columns=["Input", "Value"]),
                    use_container_width=True, hide_index=True,
                )

            vc, gc = st.columns(2)
            with vc:
                st.plotly_chart(viz.rationality_gauge(rat2), use_container_width=True)
            with gc:
                st.plotly_chart(viz.growth_distribution(rat2), use_container_width=True)

            curve_df = fair_value_curve(rd_inp2, shares, snap.price)
            st.plotly_chart(
                viz.fair_value_curve_chart(curve_df, rd2.implied_growth, snap.price),
                use_container_width=True,
            )
            st.caption("The dashed market line intersects the cyan curve **at** the implied growth rate. "
                       "This visualises what growth assumption the market is paying for today.")


# ---------------------------------------------------------------------------
# TAB 3 — Manual DCF
# ---------------------------------------------------------------------------
with tab_man:
    st.markdown('<div class="section-title">Manual sensitivity — intrinsic DCF</div>', unsafe_allow_html=True)
    st.caption("Set forward assumptions; intrinsic value updates live. Margin of safety is computed against the live market price.")

    if not (latest_revenue and shares):
        st.error("❌ Need revenue and shares outstanding to run intrinsic DCF.")
    else:
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown("**📈 Growth & margins**")
            init_rev_g = (rev_growth_history.tail(3).mean()
                          if not rev_growth_history.empty else 0.08)
            m_rev_g = pct_input("Revenue growth", float(init_rev_g),
                                -0.20, 0.60, 0.005, key="m_rev_g")
            init_op_m = ((latest_ebit / latest_revenue)
                         if (latest_ebit and latest_revenue and latest_revenue > 0)
                         else 0.15)
            m_op_m = pct_input("Operating margin", float(init_op_m),
                               -0.10, 0.60, 0.005, key="m_op_m")
            m_tax  = pct_input("Effective tax rate", 0.21, 0.0, 0.40, 0.005, key="m_tax")
        with mc2:
            st.markdown("**🏗️ Reinvestment**")
            da_init = 0.04
            if "DepreciationAmortization" in fin.index and latest_revenue:
                da_v = edgar.latest_value(fin, "DepreciationAmortization")
                if da_v and latest_revenue > 0:
                    da_init = da_v / latest_revenue
            capex_init = 0.05
            if "CapEx" in fin.index and latest_revenue:
                cx = edgar.latest_value(fin, "CapEx")
                if cx and latest_revenue > 0:
                    capex_init = abs(cx) / latest_revenue
            m_da    = pct_input("D&A / Revenue", float(da_init), 0.0, 0.20, 0.0025, key="m_da")
            m_capex = pct_input("CapEx / Revenue", float(capex_init), 0.0, 0.30, 0.0025, key="m_capex")
            m_nwc   = pct_input("ΔNWC / ΔRevenue", 0.02, 0.0, 0.20, 0.0025, key="m_nwc")
            m_years = int(st.slider("Projection years", 3, 15, 10, 1, key="m_years"))
        with mc3:
            st.markdown("**💰 Cost of capital**")
            m_wacc = pct_input("WACC", float(wacc_used), 0.04, 0.25, 0.0025, key="m_wacc")
            m_tg   = pct_input("Terminal g", float(sb_tg), 0.0, 0.05, 0.001, key="m_tg")
            m_rev0 = st.number_input(
                "Base revenue ($)", value=float(latest_revenue),
                step=1e8, format="%.0f", key="m_rev0",
            )

        if m_wacc <= m_tg:
            st.error(f"⚠️ WACC ({_fmt_pct(m_wacc)}) ≤ terminal g ({_fmt_pct(m_tg)}). Lower terminal g or raise WACC.")
        else:
            m_inp = IntrinsicInputs(
                revenue_base=m_rev0, net_debt=net_debt, shares=shares,
                revenue_growth=m_rev_g, operating_margin=m_op_m, tax_rate=m_tax,
                da_pct_revenue=m_da, capex_pct_revenue=m_capex, nwc_pct_revenue=m_nwc,
                wacc=m_wacc, terminal_g=m_tg, projection_years=m_years,
            )
            m_res = intrinsic_dcf(m_inp, snap.price)

            st.markdown("---")
            va1, va2, va3, va4 = st.columns(4)
            fv = m_res.fair_value_per_share
            mos = ((fv - (snap.price or 0)) / fv
                   if (snap.price and np.isfinite(fv) and fv > 0) else None)
            va1.metric("Intrinsic value / share",
                       f"${fv:,.2f}" if np.isfinite(fv) else "—")
            va2.metric("Market price",
                       f"${snap.price:,.2f}" if snap.price else "—")
            up = m_res.upside_pct
            va3.metric("Upside / downside",
                       _fmt_pct(up, signed=True) if up is not None else "—",
                       delta=(f"${(fv - snap.price):+,.2f}"
                              if (snap.price and np.isfinite(fv)) else None),
                       delta_color="normal")
            mos_label = (
                "✅ Adequate" if (mos and mos > 0.15) else
                "⚠️ Marginal" if (mos and mos > 0) else
                "🚨 Overvalued"
            )
            va4.metric(
                "Margin of safety",
                f"{_fmt_pct(mos)}" if mos is not None else "—",
                delta=mos_label,
                delta_color="off",
            )

            vb1, vb2, vb3 = st.columns(3)
            vb1.metric("Enterprise value", _fmt_money(m_res.enterprise_value))
            vb2.metric("Terminal value (PV)", _fmt_money(m_res.pv_terminal))
            vb3.metric("TV % of EV", _fmt_pct(m_res.tv_pct_of_ev))

            # Projection table
            st.markdown('<div class="section-title">Year-by-year projections</div>', unsafe_allow_html=True)
            proj = m_res.projections.copy()
            disp = proj[["Revenue", "EBIT", "NOPAT", "FCFF", "PV FCFF"]].copy()
            for c in disp.columns:
                disp[c] = disp[c].apply(lambda v: f"${v/1e9:,.2f}B")
            disp.index.name = "Year"
            st.dataframe(disp, use_container_width=True)

            # Valuation waterfall
            st.plotly_chart(
                viz.valuation_waterfall(
                    pv_explicit=m_res.enterprise_value - m_res.pv_terminal,
                    pv_terminal=m_res.pv_terminal,
                    net_debt=net_debt,
                    market_cap=snap.market_cap,
                ),
                use_container_width=True,
            )

            # Sensitivity heatmap
            with st.spinner("Building sensitivity grid…"):
                grid = sensitivity_grid(m_inp, snap.price)
            st.plotly_chart(
                viz.sensitivity_heatmap(grid, snap.price),
                use_container_width=True,
            )

            # Projection chart
            st.plotly_chart(viz.dcf_projection_chart(proj), use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 4 — Financial Statements
# ---------------------------------------------------------------------------
with tab_stmt:
    st.markdown('<div class="section-title">As-reported — SEC EDGAR XBRL</div>', unsafe_allow_html=True)
    st.caption("Annual fundamentals reconstructed from SEC us-gaap tags. Missing rows mean the issuer didn't tag that concept — the data is real, the absence is real too. Values ≥ $1M shown in billions.")

    inc = data["income_statement"]
    bal = data["balance_sheet"]
    cfs = data["cash_flow"]

    st.markdown("### 📈 Income Statement")
    if inc.empty:
        st.info("No income-statement data tagged for this filer.")
    else:
        st.dataframe(_format_statement(inc), use_container_width=True,
                     height=min(640, 50 + 36 * len(inc.index)))
        st.download_button(
            "⬇️ Income Statement CSV",
            inc.to_csv().encode(),
            f"{ticker}_income_statement.csv", "text/csv", key="dl_inc",
        )

    st.markdown("### 🏦 Balance Sheet")
    if bal.empty:
        st.info("No balance-sheet data tagged.")
    else:
        st.dataframe(_format_statement(bal), use_container_width=True,
                     height=min(640, 50 + 36 * len(bal.index)))
        st.download_button(
            "⬇️ Balance Sheet CSV",
            bal.to_csv().encode(),
            f"{ticker}_balance_sheet.csv", "text/csv", key="dl_bal",
        )

    st.markdown("### 💵 Cash Flow Statement")
    if cfs.empty:
        st.info("No cash-flow data tagged.")
    else:
        st.dataframe(_format_statement(cfs), use_container_width=True,
                     height=min(640, 50 + 36 * len(cfs.index)))
        st.download_button(
            "⬇️ Cash Flow CSV",
            cfs.to_csv().encode(),
            f"{ticker}_cash_flow.csv", "text/csv", key="dl_cfs",
        )


# ---------------------------------------------------------------------------
# TAB 5 — Visualisations
# ---------------------------------------------------------------------------
with tab_viz:
    st.markdown('<div class="section-title">Interactive chart suite</div>', unsafe_allow_html=True)

    vc1, vc2 = st.columns(2)
    with vc1:
        # Price history; if Manual DCF was run, m_res lives in tab_man scope so
        # we just use intrinsic_dcf with sane defaults to get a fair-value band.
        fv_chart = None
        if (latest_revenue and shares and snap.price
                and wacc_used > 0.025):
            try:
                quick_inp = IntrinsicInputs(
                    revenue_base=latest_revenue, net_debt=net_debt, shares=shares,
                    revenue_growth=float(rev_growth_history.tail(3).mean())
                                  if not rev_growth_history.empty else 0.08,
                    operating_margin=float(latest_ebit / latest_revenue)
                                     if (latest_ebit and latest_revenue) else 0.15,
                    wacc=wacc_used, terminal_g=sb_tg, projection_years=10,
                )
                fv_chart = intrinsic_dcf(quick_inp, snap.price).fair_value_per_share
            except Exception:
                fv_chart = None
        mos_low = fv_chart * 0.85 if (fv_chart and np.isfinite(fv_chart)) else None
        st.plotly_chart(
            viz.price_history_chart(hist, fv_chart, mos_low),
            use_container_width=True,
        )
        st.plotly_chart(viz.revenue_and_margins(fin), use_container_width=True)
        st.plotly_chart(
            viz.fcf_quality_chart(
                fcf_history,
                fin.loc["Revenue"] if "Revenue" in fin.index else pd.Series(dtype=float),
            ),
            use_container_width=True,
        )

    with vc2:
        st.plotly_chart(
            viz.historical_growth_bars(
                rev_growth_history, eps_growth_history,
                rd_result.implied_growth if rd_result else None,
                snap.analyst_growth_5y,
            ),
            use_container_width=True,
        )
        if rat_result:
            st.plotly_chart(viz.rationality_gauge(rat_result), use_container_width=True)
            st.plotly_chart(viz.growth_distribution(rat_result), use_container_width=True)
        st.plotly_chart(viz.roic_vs_wacc(fin, wacc_used), use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 6 — Guide
# ---------------------------------------------------------------------------
with tab_guide:
    st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
    st.markdown("""
### What this tool does

The **RvM (Rationality vs. Market)** terminal asks one question of every US-listed company:
*is today's price rationally justified by forward fundamentals, or is it pricing in a speculative growth premium?*

It answers two ways at once:

1. **Auto-Predictive (Reverse-DCF)** — ticker in, the tool back-solves the high-growth rate the market is *implicitly paying for*.
2. **Manual Sensitivity (Intrinsic-DCF)** — you supply forward assumptions, the tool returns intrinsic value per share.

The **Rationality Gate** then compares the implied rate against the company's 5-year empirical growth distribution.

---

### Data backbone

| Source | Used for | Why |
|---|---|---|
| **SEC EDGAR XBRL** | Revenue, EBIT, NetIncome, CapEx, CFO, debt, cash, shares, EPS — 10 years annual | Filed by issuers themselves; most reliable free fundamentals data on the web |
| **yfinance** | Live price, market cap, beta, sector tags, forward P/E, analyst 5y growth consensus | Real-time and forward-looking data SEC doesn't serve |
| **^TNX** | 10y US Treasury yield (risk-free rate) | Public Yahoo symbol |

---

### Reverse-DCF math

```
EV(gʰ) = Σ [FCF₀ × (1+gʰ)^t / (1+WACC)^t]   (Stage 1: t = 1 … H)
         + Σ [FCF_{H+i} / (1+WACC)^{H+i}]      (Stage 2: linear fade, i = 1 … F)
         + TV / (1+WACC)^{H+F}                   (Terminal value)

where TV = FCF_{H+F} × (1+g_t) / (WACC − g_t)
      g_i = gʰ + (g_t − gʰ) × (i/F)
```

A **bisection solver** iterates over `gʰ ∈ [−30%, +200%]` to <1e-7 precision. 
The tool also runs the solver over a basket of FCF-base scenarios (TTM, 3y avg, 5y avg) and reports a Monte-Carlo mean ± 1σ.

---

### Rationality Gate

| Metric | Definition |
|---|---|
| Pool | Pooled YoY series across Revenue, NetIncome, FCF |
| Mean / Std | Winsorised at ±3σ to reduce outlier distortion |
| z-score | `(gʰ − μ) / σ` |
| **Rational** | z ≤ 1.0 |
| **Stretched** | 1.0 < z ≤ 2.0 |
| **Speculative** | 2.0 < z ≤ 3.0 |
| **Bubble-like** | z > 3.0 |

---

### Intrinsic DCF (Manual Mode)

Standard revenue-driven FCFF bridge:

```
Revenue_t  = Revenue_{t-1} × (1+g_rev)
EBIT_t     = Revenue_t × op_margin
NOPAT_t    = EBIT_t × (1 − tax)
FCFF_t     = NOPAT_t + D&A_t − CapEx_t − ΔNWC_t
```

`Terminal Value = FCFF_N × (1+g_t) / (WACC − g_t)`

`Equity Value = PV(FCFF₁…ₙ) + PV(TV) − Net Debt`

---

### WACC (CAPM)

```
K_e   = Rf + β × ERP
WACC  = W_e × K_e + W_d × K_d × (1 − t)
```

β from yfinance; if missing, computed via 5-year monthly regression vs. ^GSPC.

---

### Margin of Safety

`MoS = (Intrinsic − Market) / Intrinsic`

- **MoS ≥ 15%** — ✅ adequate cushion 
- **0–15%** — ⚠️ marginal 
- **< 0** — 🚨 market above intrinsic

---

### Deploying to Streamlit Community Cloud (free)

1. Fork / push this repo to a GitHub account
2. Sign in at [streamlit.io/cloud](https://streamlit.io/cloud) with GitHub
3. **New app** → select repo → main file = `app.py` → **Deploy**
4. Free tier is sufficient — no API keys required

> **Disclaimer:** Educational and research tool. Not investment advice. 
> Backtest your assumptions before acting on any output.
""")
