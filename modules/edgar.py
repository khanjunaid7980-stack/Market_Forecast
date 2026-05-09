"""SEC EDGAR XBRL data retrieval — the reliable, free, no-key fundamentals source.

The SEC's company-facts API serves machine-readable XBRL data for every US
10-K filer back to 2009. It is the single most reliable source of free
fundamentals because the data is filed by the company itself, not scraped.

This module mirrors the pattern from the financial-screener project so the
two tools share the same data backbone.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import requests

SEC_HEADERS = {
    "User-Agent": "RvM-Forecast research-tool contact@rvm-forecast.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


# ---------------------------------------------------------------------------
# ticker resolution
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _ticker_map() -> dict[str, dict[str, Any]]:
    """Ticker -> {cik, title}. Fetched once per process."""
    headers = {**SEC_HEADERS, "Host": "www.sec.gov"}
    r = requests.get(TICKERS_URL, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    return {
        str(row["ticker"]).upper(): {
            "cik": str(row["cik_str"]).zfill(10),
            "title": row["title"],
        }
        for row in data.values()
    }


def resolve_ticker(ticker: str) -> dict[str, Any] | None:
    return _ticker_map().get(ticker.upper())


# ---------------------------------------------------------------------------
# concept map  (canonical key -> list of XBRL us-gaap tags to try)
# ---------------------------------------------------------------------------

CONCEPT_MAP: dict[str, list[str]] = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet",
    ],
    "CostOfRevenue": [
        "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
    ],
    "GrossProfit": ["GrossProfit"],
    "OperatingIncome": ["OperatingIncomeLoss"],
    "NetIncome": ["NetIncomeLoss", "ProfitLoss"],
    "EPS": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "SharesOutstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
    "TotalAssets": ["Assets"],
    "TotalLiabilities": ["Liabilities"],
    "TotalEquity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "CashAndEquivalents": [
        "CashAndCashEquivalentsAtCarryingValue", "Cash",
    ],
    "ShortTermDebt": [
        "ShortTermBorrowings", "LongTermDebtCurrent", "DebtCurrent",
    ],
    "LongTermDebt": [
        "LongTermDebtNoncurrent", "LongTermDebt",
    ],
    "CapEx": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
    ],
    "DepreciationAmortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "CashFromOps": ["NetCashProvidedByUsedInOperatingActivities"],
    "InterestExpense": ["InterestExpense"],
    "IncomeTaxExpense": ["IncomeTaxExpenseBenefit"],
    "PreTaxIncome": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    # Income-statement extras
    "ResearchAndDevelopment": ["ResearchAndDevelopmentExpense"],
    "SellingGeneralAdmin": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ],
    "OperatingExpenses": ["OperatingExpenses"],
    "InterestIncome": ["InterestIncomeOperating", "InvestmentIncomeInterest"],
    # Balance-sheet extras
    "AccountsReceivable": [
        "AccountsReceivableNetCurrent", "ReceivablesNetCurrent",
    ],
    "Inventory": ["InventoryNet"],
    "CurrentAssets": ["AssetsCurrent"],
    "PropertyPlantEquipment": ["PropertyPlantAndEquipmentNet"],
    "Goodwill": ["Goodwill"],
    "IntangibleAssets": ["IntangibleAssetsNetExcludingGoodwill"],
    "AccountsPayable": ["AccountsPayableCurrent"],
    "CurrentLiabilities": ["LiabilitiesCurrent"],
    "RetainedEarnings": ["RetainedEarningsAccumulatedDeficit"],
    # Cash-flow extras
    "CashFromInvesting": ["NetCashProvidedByUsedInInvestingActivities"],
    "CashFromFinancing": ["NetCashProvidedByUsedInFinancingActivities"],
    "Dividends": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "StockBuybacks": [
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfEquity",
    ],
    "StockBasedComp": ["ShareBasedCompensation"],
    "ChangesInWorkingCapital": ["IncreaseDecreaseInOperatingCapital"],
}


# ---------------------------------------------------------------------------
# CompanyFacts container
# ---------------------------------------------------------------------------

@dataclass
class CompanyFacts:
    cik: str
    name: str
    raw: dict[str, Any] = field(default_factory=dict)

    def annual_series(self, concept_tags: list[str]) -> pd.Series:
        """Annual (FY) values merged across all candidate XBRL tags.

        Selection rules per fiscal year:
          1. Frame-tagged > non-frame-tagged   (consolidated beats segment)
          2. Most recent filing wins           (handles restatements)
          3. Largest absolute value as tie-breaker
        """
        us_gaap = self.raw.get("facts", {}).get("us-gaap", {})
        allowed_forms = ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A")
        merged: dict[int, tuple[float, str, bool]] = {}

        for tag in concept_tags:
            node = us_gaap.get(tag)
            if not node:
                continue
            units = node.get("units", {})
            for preferred in ("USD", "shares", "USD/shares"):
                if preferred in units:
                    unit_key = preferred
                    break
            else:
                if not units:
                    continue
                unit_key = next(iter(units))

            for row in units[unit_key]:
                if row.get("fp") != "FY" or row.get("form") not in allowed_forms:
                    continue
                fy, val = row.get("fy"), row.get("val")
                if fy is None or val is None:
                    continue
                fy, val = int(fy), float(val)
                filed = str(row.get("filed", ""))
                has_frame = bool(row.get("frame", ""))

                prev = merged.get(fy)
                if prev is None:
                    merged[fy] = (val, filed, has_frame)
                    continue
                pv, pf, pframe = prev
                if has_frame and not pframe:
                    merged[fy] = (val, filed, has_frame)
                elif not has_frame and pframe:
                    pass
                elif filed > pf:
                    merged[fy] = (val, filed, has_frame)
                elif filed == pf and abs(val) > abs(pv):
                    merged[fy] = (val, filed, has_frame)

        if not merged:
            return pd.Series(dtype="float64")
        return pd.Series({k: v[0] for k, v in sorted(merged.items())},
                         dtype="float64")

    def build_financials(self, years: int = 10) -> pd.DataFrame:
        """Wide DataFrame of annual fundamentals (rows=items, cols=fiscal years)."""
        rows = {key: self.annual_series(tags) for key, tags in CONCEPT_MAP.items()}
        df = pd.DataFrame(rows).T
        if df.shape[1] == 0:
            return df
        df = df.reindex(sorted(df.columns), axis=1)
        populated = df.columns[df.notna().any(axis=0)]
        df = df[populated]
        if years and df.shape[1] > years:
            df = df.iloc[:, -years:]
        return df


# ---------------------------------------------------------------------------
# fetchers
# ---------------------------------------------------------------------------

def fetch_company_facts(cik: str, retries: int = 3, sleep: float = 0.5) -> CompanyFacts:
    url = FACTS_URL.format(cik=cik)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=30)
            if r.status_code == 429:
                time.sleep(1.0 + attempt)
                continue
            r.raise_for_status()
            data = r.json()
            return CompanyFacts(cik=cik, name=data.get("entityName", ""), raw=data)
        except Exception as exc:
            last_err = exc
            time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f"Failed to fetch SEC facts for CIK {cik}: {last_err}")


def fetch_recent_filings(
    cik: str, forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"),
) -> pd.DataFrame:
    r = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    recent = r.json().get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()
    df = pd.DataFrame(recent)
    df = df[df["form"].isin(forms)].copy()
    if df.empty:
        return df
    df["url"] = df.apply(
        lambda row: (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{str(row['accessionNumber']).replace('-', '')}/"
            f"{row['primaryDocument']}"
        ),
        axis=1,
    )
    return df[["form", "filingDate", "accessionNumber", "primaryDocument", "url"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# RvM-specific computed series
# ---------------------------------------------------------------------------

def compute_fcf_history(fin: pd.DataFrame) -> pd.Series:
    """FCF = CashFromOps - |CapEx| per year."""
    if fin.empty:
        return pd.Series(dtype="float64")
    if "CashFromOps" not in fin.index:
        return pd.Series(dtype="float64")
    cfo = fin.loc["CashFromOps"].astype(float)
    capex = (fin.loc["CapEx"].astype(float).abs()
             if "CapEx" in fin.index else pd.Series(0.0, index=cfo.index))
    fcf = (cfo - capex).dropna()
    return fcf


def compute_growth_history(series: pd.Series) -> pd.Series:
    """YoY growth rates as decimals."""
    s = series.dropna().astype(float)
    if len(s) < 2:
        return pd.Series(dtype="float64")
    pct = s.pct_change().dropna()
    pct = pct.replace([np.inf, -np.inf], np.nan).dropna()
    # Cap absurd values (e.g. when previous-year base is ~0)
    pct = pct.clip(lower=-0.95, upper=5.0)
    return pct


def compute_net_debt(fin: pd.DataFrame) -> float | None:
    """Most recent net debt = LongTermDebt + ShortTermDebt - CashAndEquivalents."""
    if fin.empty:
        return None
    most_recent = fin.columns[-1]
    debt = 0.0
    for k in ("LongTermDebt", "ShortTermDebt"):
        if k in fin.index:
            v = fin.loc[k, most_recent]
            if pd.notna(v):
                debt += float(v)
    cash = 0.0
    if "CashAndEquivalents" in fin.index:
        v = fin.loc["CashAndEquivalents", most_recent]
        if pd.notna(v):
            cash = float(v)
    return debt - cash


def latest_value(fin: pd.DataFrame, key: str) -> float | None:
    if fin.empty or key not in fin.index:
        return None
    s = fin.loc[key].dropna()
    return float(s.iloc[-1]) if not s.empty else None


# ---------------------------------------------------------------------------
# statement groupings
# ---------------------------------------------------------------------------

INCOME_STATEMENT_ROWS = [
    "Revenue", "CostOfRevenue", "GrossProfit",
    "ResearchAndDevelopment", "SellingGeneralAdmin", "OperatingExpenses",
    "OperatingIncome", "InterestExpense", "InterestIncome",
    "PreTaxIncome", "IncomeTaxExpense", "NetIncome", "EPS",
]

BALANCE_SHEET_ROWS = [
    "CashAndEquivalents", "AccountsReceivable", "Inventory", "CurrentAssets",
    "PropertyPlantEquipment", "Goodwill", "IntangibleAssets", "TotalAssets",
    "AccountsPayable", "ShortTermDebt", "CurrentLiabilities",
    "LongTermDebt", "TotalLiabilities", "RetainedEarnings", "TotalEquity",
]

CASH_FLOW_ROWS = [
    "NetIncome", "DepreciationAmortization", "StockBasedComp",
    "ChangesInWorkingCapital", "CashFromOps",
    "CapEx", "CashFromInvesting",
    "Dividends", "StockBuybacks", "CashFromFinancing",
]


def extract_statement(facts: CompanyFacts, rows: list[str], years: int) -> pd.DataFrame:
    data = {row: facts.annual_series(CONCEPT_MAP.get(row, [row])) for row in rows}
    df = pd.DataFrame(data).T
    if df.shape[1] == 0:
        return df
    df = df.reindex(sorted(df.columns), axis=1)
    populated = df.columns[df.notna().any(axis=0)]
    df = df[populated]
    if years and df.shape[1] > years:
        df = df.iloc[:, -years:]
    return df


__all__ = [
    "resolve_ticker", "fetch_company_facts", "fetch_recent_filings",
    "CompanyFacts", "CONCEPT_MAP", "extract_statement",
    "INCOME_STATEMENT_ROWS", "BALANCE_SHEET_ROWS", "CASH_FLOW_ROWS",
    "compute_fcf_history", "compute_growth_history",
    "compute_net_debt", "latest_value",
]
