"""Market data via yfinance — free, no API key required.

Pulls fundamentals, computes multi-year growth histories,
normalises FCF (TTM + 3-year average) and applies multiple
fallbacks so the tool works even when yfinance is partially
rate-limited.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _yoy(arr: list[float]) -> list[float]:
    """YoY growth rates from a chronologically-ordered (oldest-first) series."""
    out: list[float] = []
    for i in range(1, len(arr)):
        prev, curr = arr[i - 1], arr[i]
        if prev and abs(prev) > 1e-9 and np.isfinite(prev) and np.isfinite(curr):
            out.append((curr - prev) / abs(prev))
    return out


def _first(*vals: Any) -> Any:
    """Return the first value that is not None and not NaN."""
    for v in vals:
        if v is not None:
            try:
                if v == v:  # NaN check
                    return v
            except Exception:
                return v
    return None


def _series_recent(df: pd.DataFrame, *keys: str, n: int = 5) -> list[float]:
    """Return last n values (oldest→newest) from the first matching row in df."""
    for k in keys:
        if k in df.index:
            s = df.loc[k].sort_index().dropna()
            vals = [float(v) for v in s.values if np.isfinite(float(v))]
            return vals[-n:]
    return []


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    ticker: str
    name: str = ""
    price: float | None = None
    market_cap: float | None = None
    shares: float | None = None

    # Balance-sheet items
    total_debt: float = 0.0
    total_cash: float = 0.0
    net_debt: float = 0.0           # total_debt - total_cash

    # Income / cash flow
    fcf_ttm: float | None = None    # TTM free cash flow
    fcf_3y_avg: float | None = None # 3-year average FCF (normalised)
    revenue_ttm: float | None = None
    ebit_ttm: float | None = None
    eps_ttm: float | None = None
    operating_margin: float | None = None
    gross_margin: float | None = None
    da_ttm: float | None = None     # D&A TTM
    capex_ttm: float | None = None  # CapEx TTM (positive = outflow)

    # Market multiples
    pe_ttm: float | None = None
    forward_pe: float | None = None
    peg: float | None = None

    # Risk / cost of capital
    beta: float | None = None
    cost_of_debt: float | None = None  # estimated from interest / debt

    # Growth intelligence
    analyst_growth_5y: float | None = None   # consensus decimal
    rev_growth_history: list[float] = field(default_factory=list)   # YoY decimals
    eps_growth_history: list[float] = field(default_factory=list)
    fcf_history: list[float] = field(default_factory=list)           # absolute, oldest first
    rev_history: list[float] = field(default_factory=list)           # absolute, oldest first

    # Return metrics
    roe: float | None = None
    roa: float | None = None
    roic: float | None = None

    # Meta
    sector: str | None = None
    industry: str | None = None
    currency: str = "USD"
    data_quality: str = "OK"   # OK | PARTIAL | ERROR

    @property
    def enterprise_value(self) -> float | None:
        """Simple EV = market cap + net debt."""
        if self.market_cap is None:
            return None
        return self.market_cap + self.net_debt

    @property
    def fcf_base(self) -> float | None:
        """Best FCF base for DCF: 3y-avg if available, else TTM."""
        if self.fcf_3y_avg and abs(self.fcf_3y_avg) > 1e4:
            return self.fcf_3y_avg
        return self.fcf_ttm


# ---------------------------------------------------------------------------
# primary fetcher
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def fetch_snapshot(ticker: str) -> Snapshot:
    snap = Snapshot(ticker=ticker.upper())
    if yf is None:
        snap.data_quality = "ERROR"
        return snap

    tk = yf.Ticker(ticker)

    # --- 1. info dict (rich but sometimes flaky) ---
    info: dict[str, Any] = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}

    # --- 2. fast_info as a complement ---
    fast: dict[str, Any] = {}
    try:
        fi = tk.fast_info
        for attr in ("last_price", "market_cap", "shares", "currency"):
            try:
                v = getattr(fi, attr, None)
                if v is not None:
                    fast[attr] = v
            except Exception:
                pass
    except Exception:
        pass

    def g(*keys: str) -> Any:
        return _first(*(info.get(k) for k in keys))

    # --- 3. scalar fields from info ---
    snap.name = _first(g("longName", "shortName"), ticker.upper())
    snap.price = _first(
        g("currentPrice", "regularMarketPrice", "previousClose"),
        fast.get("last_price"),
    )
    snap.market_cap = _first(g("marketCap"), fast.get("market_cap"))
    snap.shares = _first(
        g("sharesOutstanding", "impliedSharesOutstanding"),
        fast.get("shares"),
    )
    snap.currency = _first(g("currency", "financialCurrency"), fast.get("currency"), "USD")
    snap.beta = g("beta", "beta3Year")
    snap.sector = g("sector")
    snap.industry = g("industry")
    snap.pe_ttm = g("trailingPE")
    snap.forward_pe = g("forwardPE")
    snap.peg = g("pegRatio")
    snap.eps_ttm = g("trailingEps")
    snap.operating_margin = g("operatingMargins")
    snap.gross_margin = g("grossMargins")
    snap.roe = g("returnOnEquity")
    snap.roa = g("returnOnAssets")
    snap.fcf_ttm = g("freeCashflow")
    snap.revenue_ttm = g("totalRevenue", "revenue")
    snap.da_ttm = g("depreciation", "depreciationAndAmortization")
    snap.ebit_ttm = g("ebit", "operatingIncome")

    # debt / cash — multiple fallback keys
    snap.total_debt = float(_first(g("totalDebt"), 0) or 0)
    snap.total_cash = float(_first(g("totalCash"), 0) or 0)
    snap.net_debt = snap.total_debt - snap.total_cash

    # derive market cap from parts if missing
    if snap.market_cap is None and snap.price and snap.shares:
        snap.market_cap = float(snap.price) * float(snap.shares)

    # last-resort price from history
    if snap.price is None:
        try:
            h = tk.history(period="5d", auto_adjust=True)
            if not h.empty:
                snap.price = float(h["Close"].iloc[-1])
        except Exception:
            pass

    # --- 4. Annual financials (income statement) ---
    try:
        fin = tk.financials  # index=metrics, columns=dates
        if fin is not None and not fin.empty:
            fin = fin.sort_index(axis=1)  # oldest date first

            rev_vals = _series_recent(fin,
                "Total Revenue", "TotalRevenue", "Revenue")
            if rev_vals:
                snap.rev_history = rev_vals
                snap.rev_growth_history = _yoy(rev_vals)
                if not snap.revenue_ttm:
                    snap.revenue_ttm = rev_vals[-1]

            ebit_vals = _series_recent(fin,
                "EBIT", "Operating Income", "OperatingIncome",
                "Total Operating Income As Reported")
            if ebit_vals and not snap.ebit_ttm:
                snap.ebit_ttm = ebit_vals[-1]

            ni_vals = _series_recent(fin,
                "Net Income", "Net Income Common Stockholders")
            if ni_vals:
                snap.eps_growth_history = _yoy(ni_vals)

            # implied operating margin if missing
            if (snap.operating_margin is None and snap.revenue_ttm
                    and snap.ebit_ttm and snap.revenue_ttm > 0):
                snap.operating_margin = snap.ebit_ttm / snap.revenue_ttm
    except Exception:
        pass

    # --- 5. Annual cash flow ---
    try:
        cf = tk.cashflow  # index=metrics, columns=dates
        if cf is not None and not cf.empty:
            cf = cf.sort_index(axis=1)

            # Prefer "Free Cash Flow" row; fallback: compute CFO - CapEx
            fcf_vals = _series_recent(cf, "Free Cash Flow")
            if fcf_vals:
                snap.fcf_history = fcf_vals
                if not snap.fcf_ttm or abs(snap.fcf_ttm) < 1e4:
                    snap.fcf_ttm = fcf_vals[-1]
                if len(fcf_vals) >= 3:
                    snap.fcf_3y_avg = float(np.mean(fcf_vals[-3:]))
            else:
                cfo_vals = _series_recent(cf, "Operating Cash Flow",
                    "Total Cash From Operating Activities")
                capex_vals = _series_recent(cf, "Capital Expenditure")
                if cfo_vals and capex_vals and len(cfo_vals) == len(capex_vals):
                    fcf_derived = [
                        c - abs(k) for c, k in zip(cfo_vals, capex_vals)
                    ]
                    snap.fcf_history = fcf_derived
                    if not snap.fcf_ttm or abs(snap.fcf_ttm) < 1e4:
                        snap.fcf_ttm = fcf_derived[-1]
                    if len(fcf_derived) >= 3:
                        snap.fcf_3y_avg = float(np.mean(fcf_derived[-3:]))

            # D&A
            da_vals = _series_recent(cf,
                "Depreciation Amortization Depletion",
                "Depreciation And Amortization")
            if da_vals and not snap.da_ttm:
                snap.da_ttm = da_vals[-1]

            # CapEx (make positive)
            capex_vals2 = _series_recent(cf, "Capital Expenditure")
            if capex_vals2:
                snap.capex_ttm = abs(capex_vals2[-1])
    except Exception:
        pass

    # --- 6. Balance sheet (for net debt refinement + ROIC) ---
    try:
        bs = tk.balance_sheet
        if bs is not None and not bs.empty:
            bs = bs.sort_index(axis=1)

            # yfinance may have a ready "Net Debt" row
            nd_vals = _series_recent(bs, "Net Debt")
            if nd_vals:
                snap.net_debt = nd_vals[-1]
                snap.total_cash = _first(
                    _series_recent(bs, "Cash And Cash Equivalents",
                        "Cash Cash Equivalents And Short Term Investments"),
                    [snap.total_cash],
                )[-1] if _series_recent(bs, "Cash And Cash Equivalents",
                        "Cash Cash Equivalents And Short Term Investments") else snap.total_cash
                snap.total_debt = snap.net_debt + snap.total_cash

            # ROIC = NOPAT / invested capital
            ic_vals = _series_recent(bs, "Invested Capital")
            if ic_vals and snap.ebit_ttm and ic_vals[-1] > 0:
                tax = float(info.get("effectiveTaxRate") or 0.21)
                snap.roic = snap.ebit_ttm * (1 - tax) / ic_vals[-1]
    except Exception:
        pass

    # --- 7. Analyst 5y growth consensus ---
    try:
        ge = tk.growth_estimates
        if ge is not None and not ge.empty:
            for period in ("+5y", "5y", "next5years"):
                if period in ge.index:
                    row = ge.loc[period]
                    row = row.dropna() if isinstance(row, pd.Series) else row
                    if isinstance(row, pd.Series) and not row.empty:
                        snap.analyst_growth_5y = float(row.iloc[0])
                        break
                    elif isinstance(row, (int, float)) and np.isfinite(float(row)):
                        snap.analyst_growth_5y = float(row)
                        break
    except Exception:
        pass

    # Fallback: analyst estimate from info dict
    if snap.analyst_growth_5y is None:
        snap.analyst_growth_5y = _first(
            g("earningsGrowth"),
            g("revenueGrowth"),
        )

    # --- 8. Beta regression fallback ---
    if snap.beta is None or abs(float(snap.beta or 0)) < 0.01:
        snap.beta = _estimate_beta(ticker)

    # --- 9. Implied cost of debt ---
    interest = _first(info.get("interestExpense"), info.get("netInterestIncome"))
    if interest and snap.total_debt and snap.total_debt > 0:
        snap.cost_of_debt = abs(float(interest)) / snap.total_debt

    # --- 10. Data quality ---
    if snap.price is None and snap.market_cap is None:
        snap.data_quality = "ERROR"
    elif snap.fcf_base is None or snap.revenue_ttm is None:
        snap.data_quality = "PARTIAL"
    else:
        snap.data_quality = "OK"

    return snap


# ---------------------------------------------------------------------------
# beta regression
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def _estimate_beta(ticker: str) -> float | None:
    if yf is None:
        return None
    try:
        data = yf.download(
            [ticker, "^GSPC"], period="5y", interval="1mo",
            auto_adjust=True, progress=False, threads=False,
        )
        if data is None or data.empty:
            return None
        closes = data["Close"] if "Close" in data.columns.get_level_values(0) else data
        closes = closes.dropna()
        if len(closes) < 24:
            return None
        rets = closes.pct_change().dropna()
        cols = rets.columns.tolist()
        spx = next((c for c in cols if str(c).upper() in ("^GSPC", "GSPC")), None)
        stk = next((c for c in cols if str(c).upper() == ticker.upper()), None)
        if spx is None or stk is None:
            return None
        cov = np.cov(rets[stk].values, rets[spx].values, ddof=1)
        return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# price history
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def fetch_price_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()[["Date", "Close", "Volume"]]
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
        return hist
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# risk-free rate
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def fetch_risk_free_rate() -> float:
    """10y US Treasury yield. Returns decimal (e.g. 0.0435). Falls back to 4.35%."""
    if yf is None:
        return 0.0435
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")
        if not tnx.empty:
            return float(tnx["Close"].iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.0435


__all__ = ["Snapshot", "fetch_snapshot", "fetch_price_history", "fetch_risk_free_rate"]
