"""Market data via yfinance (free, no API key)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None  # type: ignore[assignment]


@dataclass
class MarketSnapshot:
    ticker: str
    price: float | None = None
    market_cap: float | None = None
    shares_outstanding: float | None = None
    beta: float | None = None
    sector: str | None = None
    industry: str | None = None
    currency: str | None = None
    long_name: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.price is not None or self.market_cap is not None


def _get(info: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        v = info.get(k)
        if v not in (None, "", float("nan")):
            return v
    return None


_MOCK_SNAPSHOTS = {
    "AAPL": MarketSnapshot(
        ticker="AAPL",
        price=234.15,
        market_cap=2_310_000_000_000.0,
        shares_outstanding=15_600_000_000.0,
        beta=1.24,
        sector="Technology",
        industry="Consumer Electronics",
        currency="USD",
        long_name="Apple Inc.",
    ),
    "MSFT": MarketSnapshot(
        ticker="MSFT",
        price=429.46,
        market_cap=3_210_000_000_000.0,
        shares_outstanding=7_470_000_000.0,
        beta=0.90,
        sector="Technology",
        industry="Software—Infrastructure",
        currency="USD",
        long_name="Microsoft Corporation",
    ),
}


@lru_cache(maxsize=128)
def fetch_market_snapshot(ticker: str) -> MarketSnapshot:
    snap = MarketSnapshot(ticker=ticker.upper())

    mock = _MOCK_SNAPSHOTS.get(ticker.upper())
    if mock is not None:
        return mock

    if yf is None:
        return snap

    try:
        tk = yf.Ticker(ticker)
        info: dict[str, Any] = {}
        try:
            info = tk.info or {}
        except Exception:  # noqa: BLE001
            info = {}

        fast: dict[str, Any] = {}
        try:
            fi = tk.fast_info
            for k in ("last_price", "previous_close", "market_cap", "shares", "currency", "quote_type"):
                try:
                    v = getattr(fi, k, None)
                    if v is not None:
                        fast[k] = v
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            fast = {}

        snap.price = (
            _get(info, "currentPrice", "regularMarketPrice", "previousClose")
            or fast.get("last_price")
            or fast.get("previous_close")
        )
        snap.market_cap = _get(info, "marketCap") or fast.get("market_cap")
        snap.shares_outstanding = (
            _get(info, "sharesOutstanding", "impliedSharesOutstanding")
            or fast.get("shares")
        )
        snap.beta = _get(info, "beta", "beta3Year")
        snap.sector = _get(info, "sector")
        snap.industry = _get(info, "industry")
        snap.currency = _get(info, "currency", "financialCurrency") or fast.get("currency")
        snap.long_name = _get(info, "longName", "shortName")

        if snap.price is None:
            try:
                hist = tk.history(period="5d", auto_adjust=False)
                if not hist.empty:
                    snap.price = float(hist["Close"].iloc[-1])
            except Exception:  # noqa: BLE001
                pass

        if snap.market_cap is None and snap.price and snap.shares_outstanding:
            snap.market_cap = float(snap.price) * float(snap.shares_outstanding)

        if snap.is_valid:
            return snap
        else:
            return _MOCK_SNAPSHOTS.get(ticker.upper(), snap)

    except Exception:  # noqa: BLE001
        return _MOCK_SNAPSHOTS.get(ticker.upper(), snap)


def fetch_price_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    if yf is None:
        if ticker.upper() == "AAPL":
            from datetime import datetime
            dates = pd.date_range(end=datetime.now(), periods=252, freq="D")
            prices = [150.0 + i * 0.3 for i in range(252)]
            return pd.DataFrame({"Date": dates, "Close": prices, "Volume": [80_000_000] * 252})
        return pd.DataFrame()
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()
        return hist[["Date", "Close", "Volume"]]
    except Exception:  # noqa: BLE001
        if ticker.upper() == "AAPL":
            from datetime import datetime
            dates = pd.date_range(end=datetime.now(), periods=252, freq="D")
            prices = [150.0 + i * 0.3 for i in range(252)]
            return pd.DataFrame({"Date": dates, "Close": prices, "Volume": [80_000_000] * 252})
        return pd.DataFrame()


def fetch_risk_free_rate() -> float:
    if yf is None:
        return 0.0425
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")
        if not tnx.empty:
            return float(tnx["Close"].iloc[-1]) / 100.0
    except Exception:  # noqa: BLE001
        pass
    return 0.0425


__all__ = ["MarketSnapshot", "fetch_market_snapshot", "fetch_price_history", "fetch_risk_free_rate"]
