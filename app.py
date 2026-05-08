"""RvM — Rationality vs. Market Forecast Terminal

Streamlit single-page app: enter a US ticker, press Fetch, and the tool
instantly shows:
  1. Whether the market price is fundamentally justified (Reverse-DCF)
  2. Your own custom intrinsic value estimate (Manual DCF)
  3. A full visual suite: gauges, distribution curves, sensitivity heatmaps

Deployable to Streamlit Community Cloud — no API keys required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from modules.market_data import fetch_snapshot, fetch_price_history, fetch_risk_free_rate
from modules.dcf import (
    WaccInputs, ReverseDcfInputs, IntrinsicInputs,
    reverse_dcf, intrinsic_dcf, sensitivity_grid, fair_value_curve,
)
from modules.rationality import rationality_check
import modules.visualizations as viz

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
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', 'IBM Plex Mono', ui-monospace, monospace !important;
}
* { -webkit-font-smoothing: antialiased; }

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at top, #0e1118 0%, #0b0d10 80%) !important;
}
.block-container { padding-top: 1.5rem !important; padding-bottom: 4rem !important; max-width: 1440px; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(180deg, #161b24 0%, #111620 100%);
    border: 1px solid #1f2b3e;
    border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 6px 20px rgba(0,0,0,0.3);
    transition: border-color .2s ease;
}
[data-testid="metric-container"]:hover { border-color: #ffb00055; }
[data-testid="metric-container"] label {
    color: #7d8693 !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #d8dde3 !important; font-size: 1.55rem !important;
    font-weight: 700 !important; letter-spacing: -0.01em;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important; font-weight: 600 !important;
}

/* Tabs */
[data-baseweb="tab-list"] { border-bottom: 1px solid #1f2b3e !important; gap: 2px; background: transparent !important; }
[data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important; padding: 10px 16px !important;
    color: #7d8693 !important; font-weight: 500 !important; font-size: 0.88rem !important;
    transition: all .15s ease;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #ffb000 !important; border-bottom: 2px solid #ffb000 !important;
    background: linear-gradient(180deg, transparent, #ffb00010) !important; font-weight: 700 !important;
}
[data-baseweb="tab"]:hover { color: #d8dde3 !important; background: #161b24 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #090b0f 0%, #0b0d10 100%) !important; border-right: 1px solid #1f2b3e; }

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
[data-testid="stDataFrame"] { border: 1px solid #1f2b3e; border-radius: 10px; overflow: hidden; }

/* Expander */
[data-testid="stExpander"] { border: 1px solid #1f2b3e !important; border-radius: 10px !important; background: #111620 !important; }

/* Alerts */
[data-testid="stAlert"][kind="error"]   { background: #2e0c10 !important; border: 1px solid #ef444455 !important; border-radius: 8px !important; }
[data-testid="stAlert"][kind="warning"] { background: #2e1f0c !important; border: 1px solid #f59e0b55 !important; border-radius: 8px !important; }
[data-testid="stAlert"][kind="success"] { background: #0c2e15 !important; border: 1px solid #22c55e55 !important; border-radius: 8px !important; }
[data-testid="stAlert"][kind="info"]    { background: #0c1e2e !important; border: 1px solid #ffb00055 !important; border-radius: 8px !important; }

.section-title {
    font-size: 0.78rem; font-weight: 700; color: #7d8693;
    border-left: 3px solid #ffb000; padding-left: 10px;
    margin: 20px 0 10px 0; text-transform: uppercase; letter-spacing: 0.08em;
}

.company-header {
    display: flex; align-items: flex-start; justify-content: space-between;
    background: linear-gradient(135deg, #0e1420 0%, #0b1018 50%, #0d1220 100%);
    border: 1px solid #1f2b3e; border-radius: 14px;
    padding: 20px 26px; margin-bottom: 14px; flex-wrap: wrap; gap: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 1px 0 rgba(255,255,255,0.03) inset;
}
.company-name { font-size: 1.7rem; font-weight: 800; color: #f1f5f9; letter-spacing: -0.02em; }
.ticker-badge {
    color: #ffb000; font-size: 1.1rem; font-weight: 700;
    background: #ffb00015; padding: 2px 10px; border-radius: 6px;
    border: 1px solid #ffb00033; margin-left: 8px;
}
.company-sub { font-size: 0.72rem; color: #4a5568; margin-top: 3px; }
.badge {
    display: inline-block; background: #1a2535; color: #7eb0ff;
    border: 1px solid #2a3f5e; border-radius: 20px;
    padding: 2px 10px; font-size: 0.69rem; font-weight: 600;
    margin-right: 4px; margin-top: 6px; letter-spacing: 0.02em;
}

.kv-block {
    display: flex; gap: 12px; flex-wrap: wrap;
}
.kv-card {
    background: linear-gradient(180deg, #0c1220 0%, #090e18 100%);
    border: 1px solid #1f2b3e; border-radius: 10px;
    padding: 10px 16px; min-width: 130px;
}
.kv-label { color: #7d8693; font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
.kv-value { color: #f1f5f9; font-size: 1.35rem; font-weight: 700; line-height: 1.2; margin-top: 3px; }
.kv-delta-pos { color: #22c55e; font-size: 0.76rem; font-weight: 700; margin-top: 2px; }
.kv-delta-neg { color: #ef4444; font-size: 0.76rem; font-weight: 700; margin-top: 2px; }
.kv-delta-neu { color: #7d8693; font-size: 0.76rem; margin-top: 2px; }

.verdict-card {
    border-radius: 12px; padding: 16px 20px; text-align: center;
    border: 1px solid; min-width: 180px;
}

hr { border-color: #1f2b3e !important; margin: 12px 0 !important; }
.stMarkdown p { color: #d8dde3 !important; }
.stMarkdown code { background: #161b24; color: #27e0c5; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# helpers
# ============================================================================

def _fmt(v, prefix="$", suffix="", div=1, dp=2, na="—"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return na
    return f"{prefix}{float(v)/div:,.{dp}f}{suffix}"

def _pct(v, dp=2, na="—", signed=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return na
    fmt = f"{float(v)*100:{'+' if signed else ''}.{dp}f}%"
    return fmt

def _big(v, na="—"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return na
    v = float(v)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"${v/div:,.2f}{suf}"
    return f"${v:,.2f}"

def pct_slider(label, val, lo, hi, step=0.25, key=None, help=None):
    """Slider that works in percent but returns decimal."""
    v = st.slider(
        label, float(lo * 100), float(hi * 100), float(val * 100),
        step=float(step), format="%.2f%%", key=key, help=help,
    )
    return v / 100.0

def pct_input(label, val, lo=0.0, hi=1.0, step=0.01, key=None, help=None):
    v = st.number_input(
        label, float(lo * 100), float(hi * 100), float(val * 100),
        step=float(step * 100), format="%.2f", key=key, help=help,
    )
    return v / 100.0


# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.markdown("## 📈 RvM Terminal")
st.sidebar.caption("Rationality vs. Market — Free · No API key")
st.sidebar.markdown("---")

ticker = st.sidebar.text_input(
    "🔍 Ticker (US-listed)", value="AAPL",
    help="Enter any US-listed ticker: AAPL, NVDA, MSFT, META …",
).strip().upper()

fetch_btn = st.sidebar.button("⚡ Fetch & Analyse", type="primary", use_container_width=True)

if st.sidebar.button("🗑️ Clear cache", help="Force a fresh data pull."):
    st.cache_data.clear()
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ DCF defaults**")
sb_wacc = pct_slider("WACC", 0.09, 0.04, 0.20, 0.25, key="sb_wacc")
sb_tg   = pct_slider("Terminal g", 0.025, 0.00, 0.05, 0.10, key="sb_tg")
sb_hgy  = st.sidebar.slider("High-growth years", 3, 10, 5, 1, key="sb_hgy")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How to use**
1. Enter a US ticker above
2. Click **Fetch & Analyse**
3. Review the Rationality verdict
4. Adjust assumptions in the tabs
5. Export via the **Manual DCF** tab
""")
st.sidebar.markdown("<small>Data: Yahoo Finance (yfinance) · 100% free · no paid APIs</small>", unsafe_allow_html=True)


# ============================================================================
# Data loading (cached)
# ============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def load_data(sym: str):
    snap = fetch_snapshot(sym)
    hist = fetch_price_history(sym, "5y")
    rf   = fetch_risk_free_rate()
    return snap, hist, rf


if fetch_btn:
    for k in ("snap", "hist", "rf", "_loaded_ticker"):
        st.session_state.pop(k, None)

if "snap" not in st.session_state or st.session_state.get("_loaded_ticker") != ticker:
    try:
        with st.spinner(f"Fetching **{ticker}** from Yahoo Finance…"):
            snap, hist, rf = load_data(ticker)
            st.session_state["snap"] = snap
            st.session_state["hist"] = hist
            st.session_state["rf"]   = rf
            st.session_state["_loaded_ticker"] = ticker
    except Exception as exc:
        st.error(f"**Could not load {ticker}:** {exc}")
        st.stop()

snap  = st.session_state["snap"]
hist  = st.session_state["hist"]
rf    = st.session_state["rf"]

if snap.data_quality == "ERROR":
    st.error(f"❌ No data found for **{ticker}**. Check the symbol is US-listed and try again.")
    st.stop()

if snap.data_quality == "PARTIAL":
    st.warning("⚠️ Partial data — some fields could not be pulled. DCF results may be less accurate.")


# ============================================================================
# WACC
# ============================================================================

_wacc_inp = WaccInputs(
    beta=float(snap.beta or 1.1),
    risk_free_rate=rf,
    equity_risk_premium=0.055,
    cost_of_debt_pretax=float(snap.cost_of_debt or 0.05),
    tax_rate=0.21,
    debt_weight=0.20,
)
_ke   = _wacc_inp.cost_of_equity
_kd   = _wacc_inp.after_tax_kd
_wacc = _wacc_inp.wacc

# Use sidebar override if user changed it
wacc_used = sb_wacc if abs(sb_wacc - 0.09) > 0.0001 else max(_wacc, 0.06)


# ============================================================================
# Reverse-DCF (auto-computed whenever snap is loaded)
# ============================================================================

rd_result = None
rat_result = None

if (snap.market_cap and snap.fcf_base and snap.fcf_base > 0
        and snap.shares and snap.shares > 0):
    rd_inp = ReverseDcfInputs(
        fcf_base=snap.fcf_base,
        net_debt=snap.net_debt,
        shares=snap.shares,
        wacc=wacc_used,
        terminal_g=sb_tg,
        high_growth_years=sb_hgy,
        fade_years=5,
    )
    rd_result = reverse_dcf(snap.market_cap, rd_inp)
    rat_result = rationality_check(
        implied_growth=rd_result.implied_growth,
        rev_growth_history=snap.rev_growth_history,
        eps_growth_history=snap.eps_growth_history,
        analyst_growth_5y=snap.analyst_growth_5y,
    )


# ============================================================================
# Header banner
# ============================================================================

badge_html = ""
for b in filter(None, [snap.sector, snap.industry]):
    badge_html += f'<span class="badge">{b}</span>'

delta_html = ""
if rd_result and rat_result:
    v = rat_result.verdict
    c = rat_result.verdict_color
    z = rat_result.z_score
    delta_html = f'''
    <div class="kv-card" style="border-color:{c}44;background:linear-gradient(180deg,{c}10 0%,{c}06 100%);">
        <div class="kv-label">Rationality</div>
        <div class="kv-value" style="color:{c};font-size:1.1rem;">{rat_result.verdict_emoji} {v}</div>
        <div style="color:{c};font-size:0.72rem;margin-top:2px;">z = {z:+.2f}σ — Implied {_pct(rd_result.implied_growth)}</div>
    </div>'''

price_str = _fmt(snap.price, prefix="$", dp=2)
mcap_str  = _big(snap.market_cap)

st.markdown(f"""
<div class="company-header">
  <div style="flex:1;min-width:260px;">
    <div class="company-name">{snap.name}
      <span class="ticker-badge">{snap.ticker}</span>
    </div>
    <div class="company-sub">{snap.currency} · Yahoo Finance · data quality: {snap.data_quality}</div>
    <div>{badge_html}</div>
  </div>
  <div class="kv-block">
    <div class="kv-card">
      <div class="kv-label">Last Price</div>
      <div class="kv-value" style="color:#ffb000;">{price_str}</div>
    </div>
    <div class="kv-card">
      <div class="kv-label">Market Cap</div>
      <div class="kv-value">{mcap_str}</div>
    </div>
    <div class="kv-card">
      <div class="kv-label">FCF (base)</div>
      <div class="kv-value">{_big(snap.fcf_base)}</div>
      <div class="kv-delta-neu" style="font-size:0.65rem;">
        {'3y avg' if snap.fcf_3y_avg else 'TTM'}</div>
    </div>
    {delta_html}
  </div>
</div>
""", unsafe_allow_html=True)

# KPI strip
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Trailing P/E",   _fmt(snap.pe_ttm,     prefix="", dp=1))
c2.metric("Forward P/E",    _fmt(snap.forward_pe,  prefix="", dp=1))
c3.metric("PEG",            _fmt(snap.peg,         prefix="", dp=2))
c4.metric("Beta",           _fmt(snap.beta,        prefix="", dp=2))
c5.metric("WACC (est.)",    _pct(wacc_used))
c6.metric("Net Debt",       _big(snap.net_debt),
          delta="net cash" if snap.net_debt < 0 else None)

st.markdown("---")


# ============================================================================
# Tabs
# ============================================================================

tab_ov, tab_rev, tab_man, tab_viz, tab_guide = st.tabs([
    "📊 Overview",
    "🔄 Reverse DCF",
    "🎯 Manual DCF",
    "📈 Visualisations",
    "📖 Guide",
])


# ============================================================================
# TAB 1 — Overview
# ============================================================================
with tab_ov:
    st.markdown('<div class="section-title">Fundamentals snapshot</div>', unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Operating Margin",  _pct(snap.operating_margin))
    g2.metric("Gross Margin",      _pct(snap.gross_margin))
    g3.metric("ROE",               _pct(snap.roe))
    g4.metric("ROIC",              _pct(snap.roic))

    g5, g6, g7, g8 = st.columns(4)
    g5.metric("Revenue TTM",       _big(snap.revenue_ttm))
    g6.metric("EBIT TTM",          _big(snap.ebit_ttm))
    g7.metric("FCF TTM",           _big(snap.fcf_ttm))
    g8.metric("FCF 3y Avg",        _big(snap.fcf_3y_avg))

    st.markdown('<div class="section-title">Growth intelligence</div>', unsafe_allow_html=True)

    if rat_result:
        vc, dc, ec = st.columns([1, 1.5, 1.5])
        with vc:
            v = rat_result.verdict
            c = rat_result.verdict_color
            st.markdown(f"""
            <div class="verdict-card" style="background:{c}10;border-color:{c}44;">
                <div style="font-size:2.4rem;">{rat_result.verdict_emoji}</div>
                <div style="font-size:1.05rem;font-weight:800;color:{c};margin-top:4px;">{v}</div>
                <div style="font-size:0.72rem;color:#7d8693;margin-top:4px;">Rationality verdict</div>
            </div>""", unsafe_allow_html=True)

        with dc:
            rows = [
                ("Market-implied growth",    _pct(rat_result.implied_growth)),
                ("Analyst consensus (5y)",   _pct(rat_result.analyst_growth)),
                ("Historical growth mean",   _pct(rat_result.historical_mean)),
                ("Historical growth stdev",  _pct(rat_result.historical_std)),
                ("z-score vs. history",      f"{rat_result.z_score:+.2f}σ"),
                ("Implied vs analyst delta", _pct(rat_result.analyst_delta, signed=True)),
            ]
            df_rat = pd.DataFrame(rows, columns=["Metric", "Value"])
            st.dataframe(df_rat, use_container_width=True, hide_index=True, height=254)

        with ec:
            growth_diff = (
                rat_result.implied_growth - rat_result.historical_mean
                if np.isfinite(rat_result.historical_mean) else None
            )
            st.markdown('<div class="section-title">Context</div>', unsafe_allow_html=True)
            if growth_diff is not None:
                dir_word = "above" if growth_diff >= 0 else "below"
                st.markdown(f"""
The market-implied growth rate of **{_pct(rat_result.implied_growth)}** is
**{_pct(abs(growth_diff))} {dir_word}** the company\'s own 5-year historical mean.
\n{'The market is pricing a meaningful acceleration in growth.' if growth_diff > 0.05 else
 'The market is pricing in-line with historical performance.' if abs(growth_diff) <= 0.05 else
 'The market is pricing a significant deceleration or decline.'}
""")
            if rat_result.analyst_delta is not None:
                st.markdown(f"Analyst consensus estimates **{_pct(rat_result.analyst_growth)}**. "
                            f"The market is pricing **{_pct(abs(rat_result.analyst_delta))}** "
                            + ("more" if rat_result.analyst_delta > 0 else "less") + " than analysts.")
    else:
        st.info("ℹ️ Reverse-DCF requires positive TTM / 3y-average FCF. "
                "This ticker may be pre-earnings or reporting negative free cash flow.")

    # Multiples table
    st.markdown('<div class="section-title">Market multiples</div>', unsafe_allow_html=True)
    mult_rows = [
        ("Trailing P/E",   _fmt(snap.pe_ttm,    prefix="", dp=1)),
        ("Forward P/E",    _fmt(snap.forward_pe, prefix="", dp=1)),
        ("PEG Ratio",      _fmt(snap.peg,        prefix="", dp=2)),
        ("EV / Sales",     _fmt((snap.enterprise_value or 0) / snap.revenue_ttm
                                if snap.revenue_ttm and snap.revenue_ttm > 0 else None, prefix="", dp=2)),
        ("EV / EBIT",      _fmt((snap.enterprise_value or 0) / snap.ebit_ttm
                                if snap.ebit_ttm and snap.ebit_ttm > 0 else None, prefix="", dp=2)),
        ("Price / FCF",    _fmt(snap.price / (snap.fcf_base / snap.shares)
                                if snap.fcf_base and snap.shares and snap.fcf_base > 0
                                else None, prefix="", dp=1)),
    ]
    st.dataframe(
        pd.DataFrame(mult_rows, columns=["Multiple", "Value"]),
        use_container_width=True, hide_index=True, height=260,
    )


# ============================================================================
# TAB 2 — Reverse DCF
# ============================================================================
with tab_rev:
    st.markdown('<div class="section-title">Auto-predictive — reverse-DCF engine</div>', unsafe_allow_html=True)
    st.caption("Solves for the high-growth rate gʰ that equates the two-stage DCF enterprise value "
               "with today\'s observed enterprise value (market cap + net debt).")

    r1, r2, r3 = st.columns(3)
    with r1:
        r_wacc = pct_slider("WACC", wacc_used, 0.04, 0.20, 0.25, key="r_wacc",
                            help="Weighted average cost of capital. Pre-filled from CAPM.")
    with r2:
        r_tg = pct_slider("Terminal growth", sb_tg, 0.0, 0.05, 0.10, key="r_tg",
                          help="Perpetuity growth rate after the fade period. Should ≈ long-run nominal GDP.")
    with r3:
        r_hgy = st.slider("High-growth years", 3, 10, sb_hgy, 1, key="r_hgy",
                          help="Number of years of explicit high-growth before the fade period.")

    r4, r5 = st.columns(2)
    with r4:
        r_fy = st.slider("Fade years", 2, 8, 5, 1, key="r_fy",
                         help="Years over which growth linearly declines from gʰ to terminal g.")
    with r5:
        r_fcf_mode = st.selectbox(
            "FCF base",
            ["Auto (3y avg preferred)", "TTM only", "Manual override"],
            key="r_fcf_mode",
        )

    fcf_base = snap.fcf_base
    if r_fcf_mode == "TTM only":
        fcf_base = snap.fcf_ttm
    elif r_fcf_mode == "Manual override":
        fcf_base = st.number_input(
            "FCF override ($)", value=float(snap.fcf_base or 1e9),
            step=1e8, format="%.0f", key="r_fcf_ov",
        )

    if not fcf_base or fcf_base <= 0:
        st.error("❌ FCF base is zero or negative. Reverse-DCF requires positive free cash flow. "
                 "Try 'Manual override' with a normalised estimate.")
    else:
        rd_inp2 = ReverseDcfInputs(
            fcf_base=fcf_base,
            net_debt=snap.net_debt,
            shares=float(snap.shares or 1),
            wacc=r_wacc, terminal_g=r_tg,
            high_growth_years=r_hgy, fade_years=r_fy,
        )
        rd2 = reverse_dcf(float(snap.market_cap or 0), rd_inp2)
        rat2 = rationality_check(
            rd2.implied_growth,
            snap.rev_growth_history,
            snap.eps_growth_history,
            snap.analyst_growth_5y,
        )

        # Result grid
        st.markdown("---")
        ra1, ra2, ra3, ra4 = st.columns(4)
        color = rat2.verdict_color
        ra1.metric(
            label="Market-implied growth",
            value=_pct(rd2.implied_growth),
            delta="⚠️ check FCF quality" if not rd2.converged else None,
        )
        ra2.metric("Analyst consensus 5y", _pct(rat2.analyst_growth))
        ra3.metric("Historical growth mean", _pct(rat2.historical_mean))
        ra4.metric("Historical stdev", _pct(rat2.historical_std))

        rb1, rb2, rb3, rb4 = st.columns(4)
        rb1.metric("z-score", f"{rat2.z_score:+.2f}σ")
        rb2.metric("Verdict", f"{rat2.verdict_emoji} {rat2.verdict}")
        rb3.metric("EV target", _big(rd2.ev_target))
        rb4.metric("Solver converged", "✅ Yes" if rd2.converged else "❌ No")

        # Assumptions table
        with st.expander("📌 Reverse-DCF assumptions used"):
            arows = [
                ("FCF base",          _big(fcf_base)),
                ("Net debt",           _big(snap.net_debt)),
                ("Shares outstanding", f"{snap.shares:,.0f}" if snap.shares else "—"),
                ("WACC",               _pct(r_wacc)),
                ("Terminal g",         _pct(r_tg)),
                ("High-growth years",  str(r_hgy)),
                ("Fade years",         str(r_fy)),
                ("Market cap",         _big(snap.market_cap)),
                ("Enterprise value",   _big(rd2.ev_target)),
            ]
            st.dataframe(pd.DataFrame(arows, columns=["Input", "Value"]),
                         use_container_width=True, hide_index=True)

        # Charts
        vc, gc = st.columns(2)
        with vc:
            st.plotly_chart(viz.rationality_gauge(rat2), use_container_width=True)
        with gc:
            st.plotly_chart(viz.growth_distribution(rat2), use_container_width=True)

        # Fair value curve (intersection = implied growth)
        curve_df = fair_value_curve(rd_inp2, float(snap.shares or 1), snap.price)
        st.plotly_chart(
            viz.fair_value_curve_chart(curve_df, rd2.implied_growth, snap.price),
            use_container_width=True,
        )

        st.caption(
            "The curve shows the DCF fair value at each assumed high-growth rate. "
            "The intersection with the market price dashed line **is** the market-implied growth rate."
        )


# ============================================================================
# TAB 3 — Manual DCF
# ============================================================================
with tab_man:
    st.markdown('<div class="section-title">Manual sensitivity — intrinsic DCF</div>', unsafe_allow_html=True)
    st.caption("Set your own forward-looking assumptions. Intrinsic value updates live.")

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("**📈 Growth & Revenue**")
        m_rev_growth = pct_input("Revenue growth rate", snap.rev_growth_history[-1]
                                 if snap.rev_growth_history else 0.08,
                                 -0.20, 0.60, 0.5, key="m_rev_g",
                                 help="Expected annual revenue CAGR over the projection period.")
        m_op_margin  = pct_input("Operating margin", snap.operating_margin or 0.15,
                                 -0.10, 0.60, 0.5, key="m_op_m",
                                 help="EBIT / Revenue. Blends in from historical if available.")
        m_tax        = pct_input("Effective tax rate", 0.21, 0.0, 0.40, 0.5, key="m_tax")

    with mc2:
        st.markdown("**🏗️ Reinvestment**")
        m_da     = pct_input("D&A / Revenue",  (
                             snap.da_ttm / snap.revenue_ttm
                             if snap.da_ttm and snap.revenue_ttm and snap.revenue_ttm > 0
                             else 0.04), 0.0, 0.20, 0.25, key="m_da")
        m_capex  = pct_input("CapEx / Revenue", (
                             snap.capex_ttm / snap.revenue_ttm
                             if snap.capex_ttm and snap.revenue_ttm and snap.revenue_ttm > 0
                             else 0.05), 0.0, 0.30, 0.25, key="m_capex")
        m_nwc    = pct_input("ΔNWC / ΔRevenue", 0.02, 0.0, 0.20, 0.25, key="m_nwc")
        m_years  = int(st.slider("Projection years", 3, 15, 10, 1, key="m_years"))

    with mc3:
        st.markdown("**💰 Cost of Capital**")
        m_wacc   = pct_input("WACC", wacc_used, 0.04, 0.25, 0.25, key="m_wacc")
        m_tg     = pct_input("Terminal growth", sb_tg, 0.00, 0.05, 0.10, key="m_tg")
        m_rev0   = st.number_input(
            "Base revenue ($)", value=float(snap.revenue_ttm or 1e9),
            step=1e8, format="%.0f", key="m_rev0",
            help="Most recent annual revenue. Pre-filled from Yahoo Finance.",
        )

    # compute
    m_inp = IntrinsicInputs(
        revenue_base=m_rev0,
        net_debt=snap.net_debt,
        shares=float(snap.shares or 1),
        revenue_growth=m_rev_growth,
        operating_margin=m_op_margin,
        tax_rate=m_tax,
        da_pct_revenue=m_da,
        capex_pct_revenue=m_capex,
        nwc_pct_revenue=m_nwc,
        wacc=m_wacc,
        terminal_g=m_tg,
        projection_years=m_years,
    )

    if m_wacc <= m_tg:
        st.error(f"⚠️ WACC ({_pct(m_wacc)}) ≤ terminal g ({_pct(m_tg)}). Terminal value is undefined. Lower terminal g or raise WACC.")
    else:
        m_res = intrinsic_dcf(m_inp, snap.price)

        st.markdown("---")
        va1, va2, va3, va4 = st.columns(4)
        fv  = m_res.fair_value_per_share
        mos = (fv - (snap.price or 0)) / fv if (snap.price and np.isfinite(fv) and fv > 0) else None
        va1.metric("Intrinsic value / share", f"${fv:,.2f}" if np.isfinite(fv) else "—")
        va2.metric("Market price",            _fmt(snap.price, prefix="$"))
        up  = m_res.upside_pct
        va3.metric("Upside / downside",       _pct(up, signed=True) if up else "—",
                   delta=f"${(fv - snap.price):+,.2f}" if (snap.price and np.isfinite(fv)) else None,
                   delta_color="normal")
        mos_color_label = (
            "✅ Positive" if mos and mos > 0.15 else
            "⚠️ Marginal" if mos and mos > 0 else
            "🚨 Overvalued"
        )
        va4.metric("Margin of safety", f"{_pct(mos)} {mos_color_label}" if mos else "—")

        vb1, vb2, vb3 = st.columns(3)
        vb1.metric("Enterprise Value",   _big(m_res.enterprise_value))
        vb2.metric("Terminal Value",      _big(m_res.terminal_value))
        vb3.metric("TV % of EV",         _pct(m_res.tv_pct_of_ev))

        # Projection table
        st.markdown('<div class="section-title">Year-by-year projections</div>', unsafe_allow_html=True)
        proj = m_res.projections.copy()
        proj_display = proj[["Revenue", "EBIT", "NOPAT", "FCFF", "PV FCFF"]].copy()
        for col in proj_display.columns:
            proj_display[col] = proj_display[col].apply(lambda v: f"${v/1e9:,.2f}B")
        proj_display.index.name = "Year"
        st.dataframe(proj_display, use_container_width=True)

        # TV breakdown
        st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1628,#0b1020);border:1px solid #1f2b3e;
border-radius:10px;padding:14px 18px;margin-top:10px;display:flex;gap:28px;flex-wrap:wrap;">
  <div><div class="kv-label">Explicit-period PV</div>
  <div class="kv-value" style="font-size:1.15rem;">{_big(m_res.enterprise_value - m_res.pv_terminal)}</div>
  <div class="kv-delta-neu">{_pct(1 - m_res.tv_pct_of_ev)} of EV</div></div>
  <div><div class="kv-label">PV of Terminal Value</div>
  <div class="kv-value" style="font-size:1.15rem;">{_big(m_res.pv_terminal)}</div>
  <div class="kv-delta-neu">{_pct(m_res.tv_pct_of_ev)} of EV</div></div>
  <div><div class="kv-label">Less: Net Debt</div>
  <div class="kv-value" style="font-size:1.15rem;">{_big(snap.net_debt)}</div></div>
  <div><div class="kv-label">= Equity Value</div>
  <div class="kv-value" style="font-size:1.15rem;color:#22c55e;">{_big(m_res.equity_value)}</div></div>
</div>
""", unsafe_allow_html=True)

        # Sensitivity heatmap
        st.markdown('<div class="section-title">Sensitivity — Fair Value / Share (× WACC × Terminal g)</div>', unsafe_allow_html=True)
        with st.spinner("Building sensitivity grid…"):
            grid = sensitivity_grid(m_inp, snap.price)
        st.plotly_chart(viz.sensitivity_heatmap(grid, snap.price), use_container_width=True)

        # DCF projection chart
        st.plotly_chart(viz.dcf_projection_chart(m_res.projections, m_wacc), use_container_width=True)


# ============================================================================
# TAB 4 — Visualisations
# ============================================================================
with tab_viz:
    st.markdown('<div class="section-title">Interactive chart suite</div>', unsafe_allow_html=True)

    vc1, vc2 = st.columns(2)

    with vc1:
        # Price history
        fv_for_chart = None
        if "m_res" in dir() and np.isfinite(getattr(m_res, "fair_value_per_share", float("nan"))):
            fv_for_chart = m_res.fair_value_per_share
        mos_low = fv_for_chart * 0.85 if fv_for_chart else None
        st.plotly_chart(
            viz.price_history_chart(hist, fv_for_chart, mos_low),
            use_container_width=True,
        )

        # Revenue & margins
        st.plotly_chart(
            viz.revenue_and_margins(snap.rev_history),
            use_container_width=True,
        )

        # FCF quality
        if snap.fcf_history and snap.rev_history:
            st.plotly_chart(
                viz.fcf_quality_chart(snap.fcf_history, snap.rev_history),
                use_container_width=True,
            )

    with vc2:
        # Historical growth with implied reference
        implied_for_chart = rd_result.implied_growth if rd_result else None
        st.plotly_chart(
            viz.historical_growth_bars(
                snap.rev_growth_history,
                snap.eps_growth_history,
                implied_for_chart,
                snap.analyst_growth_5y,
            ),
            use_container_width=True,
        )

        # Growth distribution (if rationality result available)
        if rat_result:
            st.plotly_chart(viz.growth_distribution(rat_result), use_container_width=True)
            st.plotly_chart(viz.rationality_gauge(rat_result), use_container_width=True)


# ============================================================================
# TAB 5 — Guide
# ============================================================================
with tab_guide:
    st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
    st.markdown("""
### Reverse-DCF (Auto-Predictive Mode)

The tool solves for the single **high-growth rate gʰ** that makes the two-stage
DCF enterprise value equal to today’s **observed enterprise value** (market cap + net debt):

```
EV(gʰ) = Σ [FCF₀ × (1+gʰ)^t / (1+WACC)^t]   (Stage 1: t = 1 … H)
         + Σ [FCF_{H+i} / (1+WACC)^{H+i}]        (Stage 2: fade, i = 1 … F)
         + TV / (1+WACC)^{H+F}                     (Terminal value)

where TV = FCF_{H+F} × (1+g_t) / (WACC − g_t)
      g_i = gʰ + (g_t − gʰ) × (i/F)            (linear fade)
```

A **bisection solver** iterates over `gʰ ∈ [−30%, +150%]` to <1e-7 precision in <120 iterations.

---

### Rationality Gate

| Metric | Formula |
|--------|---------|
| Pool | `[rev_growth_yoy₁…ₙ] + [eps_growth_yoy₁…ₙ]` |
| Mean / Std | Winsorised at ±3σ to reduce outlier distortion |
| z-score | `(gʰ − μ) / σ` |
| **Rational** | z ≤ 1.0 |
| **Stretched** | 1.0 < z ≤ 2.0 |
| **Speculative** | z > 2.0 |

---

### Manual DCF (Intrinsic Mode)

Standard revenue-driven FCFF bridge:

```
Revenue_t  = Revenue_{t-1} × (1+g_rev)
EBIT_t     = Revenue_t × op_margin
NOPAT_t    = EBIT_t × (1 − tax_rate)
FCFF_t     = NOPAT_t + D&A_t − CapEx_t − ΔNWC_t
```

`Terminal Value = FCFF_N × (1+g_t) / (WACC − g_t)`

`Equity Value = PV(FCFF₁…ₙ) + PV(TV) − Net Debt`

---

### WACC Estimation

```
K_e   = Rf + β × ERP
WACC  = W_e × K_e + W_d × K_d × (1 − t)
```

β is pulled from Yahoo Finance; if missing, computed via 5-year monthly
regression against ^GSPC. Risk-free rate from `^TNX` (10y UST yield).

---

### Margin of Safety

`MoS = (Intrinsic − Market Price) / Intrinsic`

- MoS ≥ 15% → ✅ Adequate safety cushion
- MoS 0–15% → ⚠️ Marginal
- MoS < 0 → 🚨 Market above intrinsic

---

### Deploying to Streamlit Community Cloud

1. Fork this repo to your GitHub account
2. Sign in at [streamlit.io/cloud](https://streamlit.io/cloud)
3. **New app** → select repo → main file = `app.py` → **Deploy**
4. Free tier supports unlimited public apps — no API keys required

---

### Data sources & limitations

| Field | Source | Notes |
|-------|--------|-------|
| Price, market cap, beta | Yahoo Finance `info` | ±15 min delay |
| FCF, revenue, EBIT | Yahoo Finance `financials`, `cashflow` | Annual; 4 years max |
| Net debt | Yahoo Finance `balance_sheet` | Most recent annual |
| Analyst 5y growth | Yahoo Finance `growth_estimates` | Consensus; may be absent |
| Risk-free rate | `^TNX` (10y UST) | Falls back to 4.35% |

> **Disclaimer:** This tool is for educational and research purposes only. 
> It does not constitute investment advice.
""")
