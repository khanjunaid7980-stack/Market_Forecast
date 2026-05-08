# RvM — Rationality vs. Market Forecast Terminal

A web-based research tool that asks one question of every US-listed company:
**is today's price rationally justified by forward fundamentals, or is it pricing in a speculative growth premium?**

It answers the question two ways at once:

- **Auto-Predictive (Reverse-DCF)** — enter a ticker, the system pulls live
  fundamentals and back-solves for the high-growth rate the market is
  *implicitly paying for* at today's price.
- **Manual Sensitivity (Intrinsic-DCF)** — you supply forward assumptions
  (revenue growth, operating margin, WACC, terminal g) and the tool returns
  intrinsic value per share with a margin-of-safety readout.

The **Rationality Gate** then compares the implied rate against the company's
5y empirical growth distribution. If it sits more than two standard deviations
above the historical mean, the security is flagged **Speculative**.

---

## 1. Architecture

```
  Browser SPA  ──/api──▶  Express server  ──▶  Free-tier data providers
   (Vite,                       (Node 20 + TS,            Yahoo (primary),
    React 18,                    in-memory TTL cache)      FMP (optional),
    Recharts,                                              Alpha Vantage / FRED
    Tailwind dark)                                         (optional fallbacks)
```

### Data strategy (cost-aware)

| Layer            | Source                                      | Cost  | Notes |
|------------------|---------------------------------------------|-------|-------|
| Quotes / multiples / FCF / balance sheet | `yahoo-finance2` (unofficial Yahoo)        | Free  | Primary. No key required. |
| 5y revenue growth series                 | Financial Modeling Prep `/financial-growth`| Free* | Optional. Set `FMP_API_KEY`. |
| Risk-free rate (10y UST)                 | FRED `DGS10`                                | Free  | Optional. Set `FRED_API_KEY`. |
| Analyst 5y consensus                     | Yahoo `earningsTrend`                       | Free  | Already covered by primary. |

The server wraps every provider call in a TTL cache (default 10 min) so a
typical research session stays well under any free-tier limit.

### Dual-logic engine

- **`reverseDcf(marketCap, assumptions)`** — bisection solver over the
  high-growth rate `g` such that the modeled enterprise value matches the
  observed enterprise value (`marketCap + netDebt`). Convergence in <30 iters
  to 1e-6.
- **`intrinsicValuePerShare(assumptions)`** — standard two-stage DCF on
  revenue → EBIT → NOPAT → FCF, with a linear growth fade between the
  explicit and terminal periods.
- **`rationalityCheck(impliedGrowth, history)`** — pools 5y revenue and EPS
  YoY series, computes mean / stdev, returns z-score and verdict.

## 2. Reverse-DCF logic (excerpt)

The two-stage DCF uses a linear growth fade so a high implied rate doesn't
artificially inflate terminal value:

```
for t = 1..H:    FCF_t = FCF_0 * (1+g_h)^t
for i = 1..F:    g_i  = g_h + (g_t - g_h) * (i / F)
                 FCF_{H+i} = FCF_{H+i-1} * (1 + g_i)
TV         = FCF_{H+F} * (1+g_t) / (WACC - g_t)
EV(g_h)    = Σ PV(FCF_t) + PV(TV)
```

Reverse-DCF then bisects `g_h` over `[-30%, +100%]` until
`EV(g_h) == marketCap + netDebt`. See `server/src/engine/dcf.ts`.

## 3. Rationality Gate

```
z = (impliedGrowth - mean(historical 5y g)) / stdev(historical 5y g)
z > 2  → Speculative
z > 1  → Stretched
else   → Rational
```

This is intentionally a simple, transparent gate — the goal is to surface
divergence between price expectations and the company's own fundamental track
record, not to predict returns.

## 4. Interface

- **Single-page**, three vertical bands: ticker header → dual DCF panels →
  visual comparison.
- **Bloomberg-inspired dark mode**: amber primary on `#0b0d10` background,
  cyan/magenta for fundamental series, red for breach states. Monospaced
  tabular numerics throughout.
- **Sensitivity curve** plots intrinsic value vs. growth with a horizontal
  reference line at the current market price — the intersection visually
  *is* the implied growth rate.
- **Historical bar chart** overlays revenue and EPS YoY against the
  market-implied dashed line. Wide gap → the market is pricing a regime
  change.

## 5. Running it

```
npm install                # installs both workspaces
cp server/.env.example server/.env
npm run dev                # boots Express on :8787 and Vite on :5173
```

Then open http://localhost:5173 and enter a ticker.

## 6. File map

```
server/
  src/
    engine/
      dcf.ts            # enterprisePV, intrinsicValuePerShare, reverseDcf
      rationality.ts    # z-score gate
      wacc.ts           # CAPM-based WACC heuristic
    services/
      yahooFinance.ts   # primary provider
      fmp.ts            # optional enrichment
      dataProvider.ts   # cached aggregator
    routes/api.ts       # /quote /reverse-dcf /intrinsic-dcf /sensitivity
    cache.ts            # TTL memoizer
    index.ts            # Express bootstrap
web/
  src/
    components/
      TickerHeader.tsx
      ReverseDcfPanel.tsx
      ManualDcfPanel.tsx
      SensitivityChart.tsx
      HistoricalChart.tsx
    api/client.ts       # typed fetch wrappers
    App.tsx             # SPA shell
```

## 7. Caveats

This is an educational research tool, not investment advice. Free-tier
fundamentals lag, can be revised, and occasionally miss line items — the
UI surfaces “Insufficient Data” instead of guessing when history is too
short to compute a meaningful stdev.
