# RvM — Rationality vs. Market Forecast Terminal

A web app that asks one question of every US-listed company: 
**Is today's price rationally justified by forward fundamentals, or is it pricing in a speculative growth premium?**

It answers two ways at once:

- **Auto-Predictive (Reverse-DCF)** — enter a ticker, the tool back-solves the high-growth rate the market is *implicitly paying for*.
- **Manual Sensitivity (Intrinsic-DCF)** — you supply forward assumptions, the tool returns intrinsic value per share with a margin-of-safety readout.

The **Rationality Gate** classifies the implied rate against the company's 5-year empirical growth distribution — Rational, Stretched, Speculative, or Bubble-like.

## Architecture

```
  Browser (Streamlit)
        │
        ▼
  app.py
        │
        ├─ modules/edgar.py        → SEC EDGAR XBRL  (fundamentals, 10y annual)
        ├─ modules/market_data.py  → yfinance        (live price, beta, analyst 5y)
        ├─ modules/dcf.py          → reverse + intrinsic DCF + monte-carlo
        ├─ modules/rationality.py  → z-score gate (winsorised)
        └─ modules/visualizations.py → Plotly dark charts
```

**Why SEC EDGAR over yfinance for fundamentals?** 
yfinance scrapes Yahoo's web pages and is rate-limited and incomplete — line items go missing for entire fiscal years. SEC EDGAR's XBRL feed is filed by the issuer itself and serves the entire 10-year annual history with a single HTTP call. It is the most reliable free fundamentals source on the web.

## Tabs

1. **📊 Overview** — Rationality verdict, KPI strip, growth comparison bar, Monte-Carlo scenario table, recent SEC filings.
2. **🔄 Reverse DCF** — Live sliders for WACC / terminal-g / horizon, full solver output, rationality gauge + distribution + fair-value-vs-growth curve.
3. **🎯 Manual DCF** — 9 assumption inputs pre-filled from EDGAR data, year-by-year projection table, valuation waterfall, 5×5 sensitivity heatmap.
4. **📑 Financial Statements** — As-reported Income, Balance, Cash Flow with CSV downloads.
5. **📈 Visualisations** — Price-vs-fair-value, revenue & margins, FCF quality, historical growth bars, ROIC vs WACC, rationality gauge.
6. **📖 Guide** — Methodology, formulas, deployment instructions.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at <http://localhost:8501>.

## Deploy free (Streamlit Community Cloud)

1. Push this repo to your GitHub account.
2. Go to <https://streamlit.io/cloud> and sign in with GitHub.
3. **New app** → pick this repo → main file `app.py` → **Deploy**.
4. The free tier is sufficient — no API keys, no secrets, no paid services.

Within ~60 seconds you'll have a public URL like `https://your-app.streamlit.app`.

## Reverse-DCF math

```
EV(gʰ) = Σ FCF_t / (1+WACC)^t          for t = 1 … H+F
         + TV / (1+WACC)^{H+F}

FCF_t   = FCF_{t-1} × (1 + g_t)
g_t     = gʰ                            during high-growth (t ≤ H)
g_t     = gʰ + (g_term − gʰ) × (i/F)    during fade   (t = H+i)
TV      = FCF_{H+F} × (1 + g_term) / (WACC − g_term)
```

The solver bisects `gʰ ∈ [−30%, +200%]` until `EV(gʰ) == market_cap + net_debt`.

## Rationality classification

| z-score | Verdict | Interpretation |
|---|---|---|
| z ≤ 1.0 | ✅ **Rational** | Implied growth aligns with what the company has delivered |
| 1.0 < z ≤ 2.0 | ⚠️ **Stretched** | Elevated but defensible for high-quality compounders |
| 2.0 < z ≤ 3.0 | 🚨 **Speculative** | Market pricing meaningfully better future than past |
| z > 3.0 | 🔥 **Bubble-like** | Required regime change well beyond company's track record |

Mean / std are winsorised at ±3σ to reduce outlier distortion. The pool combines YoY revenue, net-income, and FCF growth.

## File map

```
app.py                       # main Streamlit entry point
requirements.txt             # pip dependencies
.streamlit/config.toml       # dark theme config
modules/
  __init__.py
  edgar.py                   # SEC EDGAR XBRL fetcher
  market_data.py             # yfinance wrapper
  dcf.py                     # reverse + intrinsic DCF
  rationality.py             # z-score gate
  visualizations.py          # Plotly charts
```

## Disclaimer

Educational and research tool. **Not investment advice.** Backtest your assumptions before acting on any output — free fundamentals lag, can be revised, and occasionally miss line items.
