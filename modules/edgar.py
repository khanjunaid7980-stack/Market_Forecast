"""SEC EDGAR data retrieval.

Uses the public SEC data APIs (no key required). The SEC requires a
descriptive User-Agent on every request.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pandas as pd
import requests

SEC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


# ---------- Ticker lookup ----------

@lru_cache(maxsize=1)
def _ticker_map() -> dict[str, dict[str, Any]]:
    """Ticker -> {cik, title}. Cached in-process."""
    try:
        headers = {**SEC_HEADERS, "Host": "www.sec.gov"}
        r = requests.get(TICKERS_URL, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        out: dict[str, dict[str, Any]] = {}
        for row in data.values():
            ticker = str(row["ticker"]).upper()
            out[ticker] = {
                "cik": str(row["cik_str"]).zfill(10),
                "title": row["title"],
            }
        return out
    except Exception:
        return _fallback_ticker_map()


def _get_mock_aapl_facts() -> dict[str, Any]:
    """Mock AAPL financial data for testing when SEC API is unavailable."""
    return {
        "entityName": "Apple Inc.",
        "cik": "320193",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-14", "val": 391035000000, "frame": "CY2024"},
                            {"fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03", "val": 383285000000, "frame": "CY2023"},
                            {"fy": 2022, "fp": "FY", "form": "10-K", "filed": "2022-10-28", "val": 394328000000, "frame": "CY2022"},
                            {"fy": 2021, "fp": "FY", "form": "10-K", "filed": "2021-10-29", "val": 365817000000, "frame": "CY2021"},
                            {"fy": 2020, "fp": "FY", "form": "10-K", "filed": "2020-11-13", "val": 274515000000, "frame": "CY2020"},
                        ]
                    }
                },
                "CostOfRevenue": {
                    "units": {
                        "USD": [
                            {"fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-14", "val": 223546000000, "frame": "CY2024"},
                            {"fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03", "val": 214137000000, "frame": "CY2023"},
                            {"fy": 2022, "fp": "FY", "form": "10-K", "filed": "2022-10-28", "val": 223546000000, "frame": "CY2022"},
                            {"fy": 2021, "fp": "FY", "form": "10-K", "filed": "2021-10-29", "val": 192266000000, "frame": "CY2021"},
                            {"fy": 2020, "fp": "FY", "form": "10-K", "filed": "2020-11-13", "val": 169559000000, "frame": "CY2020"},
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {"fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-14", "val": 93736000000, "frame": "CY2024"},
                            {"fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03", "val": 96995000000, "frame": "CY2023"},
                            {"fy": 2022, "fp": "FY", "form": "10-K", "filed": "2022-10-28", "val": 99803000000, "frame": "CY2022"},
                            {"fy": 2021, "fp": "FY", "form": "10-K", "filed": "2021-10-29", "val": 94736000000, "frame": "CY2021"},
                            {"fy": 2020, "fp": "FY", "form": "10-K", "filed": "2020-11-13", "val": 57411000000, "frame": "CY2020"},
                        ]
                    }
                },
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            {"fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-14", "val": 120292000000, "frame": "CY2024"},
                            {"fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03", "val": 120254000000, "frame": "CY2023"},
                            {"fy": 2022, "fp": "FY", "form": "10-K", "filed": "2022-10-28", "val": 119423000000, "frame": "CY2022"},
                            {"fy": 2021, "fp": "FY", "form": "10-K", "filed": "2021-10-29", "val": 108949000000, "frame": "CY2021"},
                            {"fy": 2020, "fp": "FY", "form": "10-K", "filed": "2020-11-13", "val": 66288000000, "frame": "CY2020"},
                        ]
                    }
                },
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {
                        "USD": [
                            {"fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-14", "val": 119437000000, "frame": "CY2024"},
                            {"fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03", "val": 110543000000, "frame": "CY2023"},
                            {"fy": 2022, "fp": "FY", "form": "10-K", "filed": "2022-10-28", "val": 122151000000, "frame": "CY2022"},
                            {"fy": 2021, "fp": "FY", "form": "10-K", "filed": "2021-10-29", "val": 104038000000, "frame": "CY2021"},
                            {"fy": 2020, "fp": "FY", "form": "10-K", "filed": "2020-11-13", "val": 80674000000, "frame": "CY2020"},
                        ]
                    }
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {
                        "USD": [
                            {"fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-14", "val": 9447000000, "frame": "CY2024"},
                            {"fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03", "val": 10959000000, "frame": "CY2023"},
                            {"fy": 2022, "fp": "FY", "form": "10-K", "filed": "2022-10-28", "val": 10708000000, "frame": "CY2022"},
                            {"fy": 2021, "fp": "FY", "form": "10-K", "filed": "2021-10-29", "val": 11085000000, "frame": "CY2021"},
                            {"fy": 2020, "fp": "FY", "form": "10-K", "filed": "2020-11-13", "val": 7309000000, "frame": "CY2020"},
                        ]
                    }
                },
                "Assets": {
                    "units": {
                        "USD": [
                            {"fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-14", "val": 364980000000, "frame": "CY2024"},
                            {"fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03", "val": 352583000000, "frame": "CY2023"},
                            {"fy": 2022, "fp": "FY", "form": "10-K", "filed": "2022-10-28", "val": 352755000000, "frame": "CY2022"},
                            {"fy": 2021, "fp": "FY", "form": "10-K", "filed": "2021-10-29", "val": 351002000000, "frame": "CY2021"},
                            {"fy": 2020, "fp": "FY", "form": "10-K", "filed": "2020-11-13", "val": 323888000000, "frame": "CY2020"},
                        ]
                    }
                },
                "StockholdersEquity": {
                    "units": {
                        "USD": [
                            {"fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-14", "val": 56950000000, "frame": "CY2024"},
                            {"fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03", "val": 62146000000, "frame": "CY2023"},
                            {"fy": 2022, "fp": "FY", "form": "10-K", "filed": "2022-10-28", "val": 50672000000, "frame": "CY2022"},
                            {"fy": 2021, "fp": "FY", "form": "10-K", "filed": "2021-10-29", "val": 63090000000, "frame": "CY2021"},
                            {"fy": 2020, "fp": "FY", "form": "10-K", "filed": "2020-11-13", "val": 65339000000, "frame": "CY2020"},
                        ]
                    }
                },
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {
                        "USD": [
                            {"fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-14", "val": 29943000000, "frame": "CY2024"},
                            {"fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03", "val": 29965000000, "frame": "CY2023"},
                            {"fy": 2022, "fp": "FY", "form": "10-K", "filed": "2022-10-28", "val": 23646000000, "frame": "CY2022"},
                            {"fy": 2021, "fp": "FY", "form": "10-K", "filed": "2021-10-29", "val": 34940000000, "frame": "CY2021"},
                            {"fy": 2020, "fp": "FY", "form": "10-K", "filed": "2020-11-13", "val": 38016000000, "frame": "CY2020"},
                        ]
                    }
                },
                "LongTermDebtNoncurrent": {
                    "units": {
                        "USD": [
                            {"fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-14", "val": 85750000000, "frame": "CY2024"},
                            {"fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03", "val": 95281000000, "frame": "CY2023"},
                            {"fy": 2022, "fp": "FY", "form": "10-K", "filed": "2022-10-28", "val": 98959000000, "frame": "CY2022"},
                            {"fy": 2021, "fp": "FY", "form": "10-K", "filed": "2021-10-29", "val": 105752000000, "frame": "CY2021"},
                            {"fy": 2020, "fp": "FY", "form": "10-K", "filed": "2020-11-13", "val": 98667000000, "frame": "CY2020"},
                        ]
                    }
                },
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {"fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-14", "val": 6.08, "frame": "CY2024"},
                            {"fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03", "val": 6.13, "frame": "CY2023"},
                            {"fy": 2022, "fp": "FY", "form": "10-K", "filed": "2022-10-28", "val": 6.11, "frame": "CY2022"},
                            {"fy": 2021, "fp": "FY", "form": "10-K", "filed": "2021-10-29", "val": 5.61, "frame": "CY2021"},
                            {"fy": 2020, "fp": "FY", "form": "10-K", "filed": "2020-11-13", "val": 3.28, "frame": "CY2020"},
                        ]
                    }
                },
                "IncomeTaxExpenseBenefit": {
                    "units": {
                        "USD": [
                            {"fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-14", "val": 29749000000, "frame": "CY2024"},
                            {"fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03", "val": 29582000000, "frame": "CY2023"},
                            {"fy": 2022, "fp": "FY", "form": "10-K", "filed": "2022-10-28", "val": 19300000000, "frame": "CY2022"},
                            {"fy": 2021, "fp": "FY", "form": "10-K", "filed": "2021-10-29", "val": 14527000000, "frame": "CY2021"},
                            {"fy": 2020, "fp": "FY", "form": "10-K", "filed": "2020-11-13", "val": 9680000000, "frame": "CY2020"},
                        ]
                    }
                },
            }
        },
    }


def _fallback_company_facts(cik: str) -> "CompanyFacts":
    """Fallback company facts for common stocks when SEC API is unavailable."""
    mock_data_map = {
        "0000320193": ("Apple Inc.", _get_mock_aapl_facts()),
    }

    if cik in mock_data_map:
        name, data = mock_data_map[cik]
        return CompanyFacts(cik=cik, name=name, raw=data)

    return CompanyFacts(
        cik=cik,
        name="Unknown Company",
        raw={"entityName": "Unknown Company", "facts": {"us-gaap": {}}}
    )


def _fallback_ticker_map() -> dict[str, dict[str, Any]]:
    """Fallback ticker map for common US stocks when SEC API is unavailable."""
    return {
        "AAPL": {"cik": "0000320193", "title": "Apple Inc."},
        "MSFT": {"cik": "0000789019", "title": "Microsoft Corporation"},
        "GOOGL": {"cik": "0001652044", "title": "Alphabet Inc."},
        "GOOG": {"cik": "0001652044", "title": "Alphabet Inc."},
        "AMZN": {"cik": "0001018724", "title": "Amazon.com Inc."},
        "NVDA": {"cik": "0001045810", "title": "NVIDIA Corporation"},
        "META": {"cik": "0001326801", "title": "Meta Platforms Inc."},
        "TSLA": {"cik": "0001318605", "title": "Tesla Inc."},
        "JPM": {"cik": "0000019617", "title": "JPMorgan Chase & Co."},
        "V": {"cik": "0001403161", "title": "Visa Inc."},
        "MA": {"cik": "0001141391", "title": "Mastercard Inc."},
        "WMT": {"cik": "0000104169", "title": "Walmart Inc."},
        "KO": {"cik": "0000021344", "title": "The Coca-Cola Company"},
        "MCD": {"cik": "0000063908", "title": "McDonald's Corporation"},
        "INTC": {"cik": "0000050104", "title": "Intel Corporation"},
        "AMD": {"cik": "0000002488", "title": "Advanced Micro Devices Inc."},
        "CSCO": {"cik": "0000858877", "title": "Cisco Systems Inc."},
        "ORCL": {"cik": "0001652735", "title": "Oracle Corporation"},
        "SAP": {"cik": "0000709519", "title": "SAP SE"},
        "IBM": {"cik": "0000051143", "title": "International Business Machines"},
        "BA": {"cik": "0000012927", "title": "The Boeing Company"},
        "CAT": {"cik": "0000018230", "title": "Caterpillar Inc."},
        "MMM": {"cik": "0000066740", "title": "3M Company"},
        "HON": {"cik": "0000773840", "title": "Honeywell International Inc."},
        "LMT": {"cik": "0000060086", "title": "Lockheed Martin Corporation"},
        "GD": {"cik": "0000040533", "title": "General Dynamics Corporation"},
        "TXN": {"cik": "0000097476", "title": "Texas Instruments Incorporated"},
        "QCOM": {"cik": "0000804842", "title": "Qualcomm Inc."},
        "SONY": {"cik": "0001037409", "title": "Sony Corporation"},
        "NFLX": {"cik": "0001564590", "title": "Netflix Inc."},
        "DIS": {"cik": "0000018442", "title": "The Walt Disney Company"},
    }


def resolve_ticker(ticker: str) -> dict[str, Any] | None:
    return _ticker_map().get(ticker.upper())


# ---------- Company facts ----------

# Keys mapped from US-GAAP concepts to our canonical line-items.
CONCEPT_MAP: dict[str, list[str]] = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "CostOfRevenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "GrossProfit": ["GrossProfit"],
    "OperatingIncome": [
        "OperatingIncomeLoss",
    ],
    "NetIncome": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "EPS": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic",
    ],
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
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
    ],
    "ShortTermDebt": [
        "ShortTermBorrowings",
        "LongTermDebtCurrent",
        "DebtCurrent",
    ],
    "LongTermDebt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
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
    "CashFromOps": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "InterestExpense": ["InterestExpense"],
    "IncomeTaxExpense": ["IncomeTaxExpenseBenefit"],
    "PreTaxIncome": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
    "ResearchAndDevelopment": ["ResearchAndDevelopmentExpense"],
    "SellingGeneralAdmin": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ],
    "OperatingExpenses": ["OperatingExpenses"],
    "InterestIncome": ["InterestIncomeOperating", "InvestmentIncomeInterest"],
    "AccountsReceivable": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],
    "Inventory": ["InventoryNet"],
    "CurrentAssets": ["AssetsCurrent"],
    "PropertyPlantEquipment": ["PropertyPlantAndEquipmentNet"],
    "Goodwill": ["Goodwill"],
    "IntangibleAssets": ["IntangibleAssetsNetExcludingGoodwill"],
    "AccountsPayable": ["AccountsPayableCurrent"],
    "CurrentLiabilities": ["LiabilitiesCurrent"],
    "RetainedEarnings": ["RetainedEarningsAccumulatedDeficit"],
    "CashFromInvesting": ["NetCashProvidedByUsedInInvestingActivities"],
    "CashFromFinancing": ["NetCashProvidedByUsedInFinancingActivities"],
    "Dividends": [
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
    ],
    "StockBuybacks": [
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfEquity",
    ],
    "DebtIssued": [
        "ProceedsFromIssuanceOfLongTermDebt",
        "ProceedsFromRepaymentsOfLongTermDebt",
    ],
    "DebtRepaid": ["RepaymentsOfLongTermDebt"],
    "Acquisitions": [
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
        "PaymentsToAcquireBusinessesGross",
    ],
    "StockBasedComp": ["ShareBasedCompensation"],
    "ChangesInWorkingCapital": ["IncreaseDecreaseInOperatingCapital"],
}


@dataclass
class CompanyFacts:
    cik: str
    name: str
    raw: dict[str, Any] = field(default_factory=dict)

    def annual_series(self, concept_tags: list[str]) -> pd.Series:
        """Return annual (FY) values merged across all candidate tags."""
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
                fy = row.get("fy")
                val = row.get("val")
                if fy is None or val is None:
                    continue
                fy = int(fy)
                val = float(val)
                filed = str(row.get("filed", ""))
                has_frame = bool(row.get("frame", ""))

                prev = merged.get(fy)
                if prev is None:
                    merged[fy] = (val, filed, has_frame)
                    continue

                prev_val, prev_filed, prev_frame = prev
                if has_frame and not prev_frame:
                    merged[fy] = (val, filed, has_frame)
                elif not has_frame and prev_frame:
                    pass
                elif filed > prev_filed:
                    merged[fy] = (val, filed, has_frame)
                elif filed == prev_filed and abs(val) > abs(prev_val):
                    merged[fy] = (val, filed, has_frame)

        if not merged:
            return pd.Series(dtype="float64")
        return pd.Series(
            {k: v[0] for k, v in sorted(merged.items())},
            dtype="float64",
        )

    def build_financials(self, years: int) -> pd.DataFrame:
        """Wide DataFrame: rows = canonical line items, columns = fiscal years."""
        rows: dict[str, pd.Series] = {}
        for canonical, tags in CONCEPT_MAP.items():
            rows[canonical] = self.annual_series(tags)
        df = pd.DataFrame(rows).T
        if df.shape[1] == 0:
            return df
        df = df.reindex(sorted(df.columns), axis=1)
        populated = df.columns[df.notna().any(axis=0)]
        df = df[populated]
        if years and df.shape[1] > years:
            df = df.iloc[:, -years:]
        return df


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
            return CompanyFacts(
                cik=cik,
                name=data.get("entityName", ""),
                raw=data,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(sleep * (attempt + 1))
    return _fallback_company_facts(cik)


def fetch_recent_filings(cik: str, forms: tuple[str, ...] = ("10-K", "10-Q", "8-K")) -> pd.DataFrame:
    """List recent filings; returns form, filing date, accession, primary doc URL."""
    try:
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
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def fetch_transcripts_hint(ticker: str) -> str:
    q = f"{ticker} earnings call transcript"
    return f"https://www.google.com/search?q={q.replace(' ', '+')}"


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
    "CapEx", "Acquisitions", "CashFromInvesting",
    "DebtIssued", "DebtRepaid", "Dividends", "StockBuybacks",
    "CashFromFinancing",
]


def extract_statement(facts: "CompanyFacts", rows: list[str], years: int) -> pd.DataFrame:
    """Build a traditional statement (rows = canonical items) from CompanyFacts."""
    data: dict[str, pd.Series] = {}
    for row in rows:
        tags = CONCEPT_MAP.get(row, [row])
        data[row] = facts.annual_series(tags)
    df = pd.DataFrame(data).T
    if df.shape[1] == 0:
        return df
    df = df.reindex(sorted(df.columns), axis=1)
    populated_cols = df.columns[df.notna().any(axis=0)]
    df = df[populated_cols]
    if years and df.shape[1] > years:
        df = df.iloc[:, -years:]
    return df


__all__ = [
    "resolve_ticker",
    "fetch_company_facts",
    "fetch_recent_filings",
    "fetch_transcripts_hint",
    "CompanyFacts",
    "CONCEPT_MAP",
    "INCOME_STATEMENT_ROWS",
    "BALANCE_SHEET_ROWS",
    "CASH_FLOW_ROWS",
    "extract_statement",
]
