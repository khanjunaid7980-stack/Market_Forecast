"""Live market data via yfinance.

EDGAR (modules/edgar.py) is the primary fundamentals source.
This module is a thin wrapper around yfinance for the bits SEC doesn't serve:
live price, market cap, beta, sector tags, forward multiples, and — when
available — the analyst 5-year growth consensus.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore[assignment]


@dataclass
class MarketSnapshot:
    ticker: str
    price: float | None = None
    market_cap: float | None = None
    shares_outstanding: float | None = None
    beta: float | None = None
    forward_pe: float | None = None
    trailing_pe: float | None = None
    peg: float | None = None
    sector: str | None = None
    industry: str | None = None
    long_name: str | None = None
    currency: str = "USD"
    analyst_growth_5y: float | None = None  # decimal

    @property
    def is_valid(self) -> bool:
        return self.price is not None or self.market_cap is not None


def _g(info: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        v = info.get(k)
        if v is not None:
            try:
                if v == v:
                    return v
            except Exception:
                return v
    return None


@lru_cache(maxsize=128)
def fetch_market_snapshot(ticker: str) -> MarketSnapshot:
    snap = MarketSnapshot(ticker=ticker.upper())
    if yf is None:
        return snap
    try:
        tk = yf.Ticker(ticker)
        info: dict[str, Any] = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        fast: dict[str, Any] = {}
        try:
            fi = tk.fast_info
            for k in ("last_price", "market_cap", "shares", "currency"):
                try:
                    v = getattr(fi, k, None)
                    if v is not None:
                        fast[k] = v
                except Exception:
                    pass
        except Exception:
            pass

        snap.price = (
            _g(info, "currentPrice", "regularMarketPrice", "previousClose")
            or fast.get("last_price")
        )
        snap.market_cap = _g(info, "marketCap") or fast.get("market_cap")
        snap.shares_outstanding = (
            _g(info, "sharesOutstanding", "impliedSharesOutstanding")
            or fast.get("shares")
        )
        snap.beta = _g(info, "beta", "beta3Year")
        snap.forward_pe = _g(info, "forwardPE")
        snap.trailing_pe = _g(info, "trailingPE")
        snap.peg = _g(info, "pegRatio", "trailingPegRatio")
        snap.sector = _g(info, "sector")
        snap.industry = _g(info, "industry")
        snap.long_name = _g(info, "longName", "shortName")
        snap.currency = _g(info, "currency", "financialCurrency") or fast.get("currency") or "USD"

        # Analyst 5y growth via growth_estimates table (yfinance >=0.2.40)
        try:
            ge = tk.growth_estimates
            if ge is not None and not ge.empty:
                for period in ("+5y", "5y", "5Y"):
                    if period in ge.index:
                        row = ge.loc[period]
                        v = row.iloc[0] if isinstance(row, pd.Series) else row
                        if v is not None and np.isfinite(float(v)):
                            snap.analyst_growth_5y = float(v)
                            break
        except Exception:
            pass
        if snap.analyst_growth_5y is None:
            snap.analyst_growth_5y = _g(info, "earningsGrowth", "revenueGrowth")

        # Last-resort price from 5-day history
        if snap.price is None:
            try:
                hist = tk.history(period="5d", auto_adjust=False)
                if not hist.empty:
                    snap.price = float(hist["Close"].iloc[-1])
            except Exception:
                pass

        # Derive market cap from parts
        if snap.market_cap is None and snap.price and snap.shares_outstanding:
            snap.market_cap = float(snap.price) * float(snap.shares_outstanding)

        # Beta regression fallback
        if snap.beta is None or abs(float(snap.beta or 0)) < 0.01:
            snap.beta = _estimate_beta(ticker)
    except Exception:
        pass
    return snap


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
        cols = list(rets.columns)
        spx = next((c for c in cols if str(c).upper().endswith("GSPC")), None)
        stk = next((c for c in cols if str(c).upper() == ticker.upper()), None)
        if spx is None or stk is None:
            return None
        cov = np.cov(rets[stk].values, rets[spx].values, ddof=1)
        return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else None
    except Exception:
        return None


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


@lru_cache(maxsize=1)
def fetch_risk_free_rate() -> float:
    """10y UST yield as decimal. Falls back to 4.35%."""
    if yf is None:
        return 0.0435
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")
        if not tnx.empty:
            return float(tnx["Close"].iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.0435


__all__ = [
    "MarketSnapshot", "fetch_market_snapshot",
    "fetch_price_history", "fetch_risk_free_rate",
]
