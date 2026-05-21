"""Monte Carlo DCF simulation under parametric uncertainty.

Two simulation modes run simultaneously:

  Forward MC  — sample (g, op_margin, WACC, terminal_g) from their
                distributions → compute intrinsic equity value per share.
                Produces a distribution of fair values and the probability
                the stock is currently undervalued.

  Reverse MC  — sample (WACC, fcf_margin, terminal_g) → run Reverse-DCF
                at fixed market cap → distribution of implied revenue CAGRs
                showing how sensitive the market's implied expectation is to
                your discount-rate / margin assumptions.

Sensitivity (tornado chart) — One-at-a-time (OAT) method: for each uncertain
input vary it from mean − 1σ to mean + 1σ while holding others fixed and
record the resulting intrinsic value at each extreme.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.reverse_dcf import solve_implied_growth


@dataclass
class MonteCarloResult:
    n_sims: int
    n_valid_fwd: int          # simulations that produced a finite IV
    n_valid_rev: int          # simulations that converged in Reverse-DCF
    # Forward DCF — intrinsic value distribution
    iv_per_share: np.ndarray  # shape (n_valid_fwd,)
    iv_equity: np.ndarray     # shape (n_valid_fwd,)
    p5:  float | None
    p25: float | None
    p50: float | None         # median fair value
    p75: float | None
    p95: float | None
    prob_undervalued: float | None   # P(IV > current_price)
    expected_return: float | None    # E[(IV/price) − 1]
    # Reverse DCF — implied CAGR distribution
    implied_cagrs: np.ndarray  # shape (n_valid_rev,)
    cagr_p5:  float | None
    cagr_p25: float | None
    cagr_p50: float | None
    cagr_p75: float | None
    cagr_p95: float | None
    # OAT sensitivity: param → (iv_at_low, iv_at_base, iv_at_high)
    sensitivity: dict[str, tuple[float, float, float]]
    # Scenario table
    scenarios: dict[str, dict]


# ── Vectorised forward DCF ────────────────────────────────────────────────────

def _fwd_dcf_batch(
    base_rev:  float,
    net_debt:  float,
    growths:   np.ndarray,   # (n,)
    op_margins:np.ndarray,   # (n,)
    waccs:     np.ndarray,   # (n,)
    term_gs:   np.ndarray,   # (n,)
    tax_rate:  float,
    capex_pct: float,
    years:     int,
) -> np.ndarray:
    """Return equity value (n,) for each parameter draw. np.nan on invalid rows."""
    t = np.arange(1, years + 1, dtype=float)          # (years,)

    # Revenue: (n, years)
    rev = base_rev * (1.0 + growths[:, None]) ** t[None, :]

    # FCFF: Rev × (op_margin × (1 − tax) − capex_pct)
    fcff_factor = op_margins * (1.0 - tax_rate) - capex_pct   # (n,)
    fcff = rev * fcff_factor[:, None]                          # (n, years)

    # Discount factors
    df = (1.0 + waccs[:, None]) ** t[None, :]                 # (n, years)
    pv_ex = (fcff / df).sum(axis=1)                           # (n,)

    # Gordon Growth terminal value on last-period FCFF
    tg_safe = np.minimum(term_gs, waccs - 0.002)
    valid = (waccs - tg_safe) > 0.001
    tv = np.where(
        valid,
        fcff[:, -1] * (1.0 + tg_safe) / (waccs - tg_safe),
        np.nan,
    )
    pv_tv = tv / (1.0 + waccs) ** years

    equity = pv_ex + pv_tv - net_debt
    # Mask extreme / nonsensical values
    equity = np.where(np.isfinite(equity), equity, np.nan)
    return equity


# ── OAT tornado ──────────────────────────────────────────────────────────────

def _oat_iv(
    base_rev: float, net_debt: float, tax_rate: float, capex_pct: float, years: int,
    wacc: float, g: float, op_m: float, tg: float,
) -> float:
    arr = _fwd_dcf_batch(
        base_rev, net_debt,
        np.array([g]), np.array([op_m]),
        np.array([wacc]), np.array([min(tg, wacc - 0.002)]),
        tax_rate, capex_pct, years,
    )
    v = float(arr[0])
    return v if np.isfinite(v) else float("nan")


def _tornado(
    base_rev: float, net_debt: float, tax_rate: float, capex_pct: float, years: int,
    wacc_mu: float, wacc_sig: float,
    g_mu: float,    g_sig: float,
    op_mu: float,   op_sig: float,
    tg_mu: float,   tg_sig: float,
) -> dict[str, tuple[float, float, float]]:
    """OAT: for each input return (iv_at_low, iv_at_base, iv_at_high)."""
    base = _oat_iv(base_rev, net_debt, tax_rate, capex_pct, years,
                   wacc_mu, g_mu, op_mu, tg_mu)
    result: dict[str, tuple[float, float, float]] = {}

    params = [
        ("Revenue Growth",  g_mu,    g_sig,
         lambda v: _oat_iv(base_rev, net_debt, tax_rate, capex_pct, years,
                            wacc_mu, v, op_mu, tg_mu)),
        ("WACC",            wacc_mu, wacc_sig,
         lambda v: _oat_iv(base_rev, net_debt, tax_rate, capex_pct, years,
                            v, g_mu, op_mu, tg_mu)),
        ("Operating Margin",op_mu,   op_sig,
         lambda v: _oat_iv(base_rev, net_debt, tax_rate, capex_pct, years,
                            wacc_mu, g_mu, v, tg_mu)),
        ("Terminal Growth", tg_mu,   tg_sig,
         lambda v: _oat_iv(base_rev, net_debt, tax_rate, capex_pct, years,
                            wacc_mu, g_mu, op_mu, v)),
    ]
    for name, mu, sig, fn in params:
        lo = fn(max(mu - sig, -0.30) if name == "Revenue Growth" else mu - sig)
        hi = fn(mu + sig)
        result[name] = (float(lo) if np.isfinite(lo) else base,
                        float(base),
                        float(hi) if np.isfinite(hi) else base)
    return result


# ── Main entry point ─────────────────────────────────────────────────────────

def run_monte_carlo(
    base_rev:       float,
    net_debt:       float,
    market_cap:     float,
    shares:         float | None,
    current_price:  float | None,
    tax_rate:       float,
    capex_pct:      float,
    years:          int,
    # Distribution centres (μ)
    wacc_mu:   float, wacc_sig:  float,
    g_mu:      float, g_sig:     float,
    op_mu:     float, op_sig:    float,
    tg_mu:     float, tg_sig:    float,
    fcf_mu:    float, fcf_sig:   float,   # for Reverse MC
    n_sims:    int = 2000,
    seed:      int = 42,
) -> MonteCarloResult:
    rng = np.random.default_rng(seed)

    # ── Draw parameter samples ────────────────────────────────────────────────
    waccs   = np.clip(rng.normal(wacc_mu,  wacc_sig,  n_sims), 0.04,  0.25)
    growths = np.clip(rng.normal(g_mu,     g_sig,     n_sims), -0.25, 0.65)
    op_mgs  = np.clip(rng.normal(op_mu,    op_sig,    n_sims), 0.01,  0.80)
    term_gs = np.clip(rng.normal(tg_mu,    tg_sig,    n_sims), 0.005, 0.05)
    fcf_mgs = np.clip(rng.normal(fcf_mu,   fcf_sig,   n_sims), 0.005, 0.70)

    # ── Forward DCF — vectorised ──────────────────────────────────────────────
    equity_arr = _fwd_dcf_batch(
        base_rev, net_debt, growths, op_mgs, waccs, term_gs,
        tax_rate, capex_pct, years,
    )

    valid_mask = np.isfinite(equity_arr) & (equity_arr > -net_debt * 10)
    eq_valid = equity_arr[valid_mask]
    n_valid_fwd = int(valid_mask.sum())

    if shares and shares > 0 and n_valid_fwd > 10:
        iv_ps = eq_valid / shares
        p5, p25, p50, p75, p95 = np.percentile(iv_ps, [5, 25, 50, 75, 95])
        prob_under = (
            float(np.mean(iv_ps > current_price)) if current_price else None
        )
        exp_ret = (
            float(np.mean(iv_ps / current_price - 1.0)) if current_price else None
        )
    else:
        iv_ps = np.array([])
        p5 = p25 = p50 = p75 = p95 = None
        prob_under = exp_ret = None

    # ── Reverse DCF — loop (Brent's method per draw) ─────────────────────────
    # Only vary WACC + FCF margin; market cap is fixed (observed price).
    cagr_list: list[float] = []
    n_rev = min(n_sims, 1500)   # cap to avoid UI lag
    for i in range(n_rev):
        res = solve_implied_growth(
            market_cap=market_cap,
            base_revenue=base_rev,
            fcf_margin=fcf_mgs[i],
            wacc=waccs[i],
            terminal_growth=term_gs[i],
            years=years,
            net_debt=net_debt,
        )
        if res.converged and res.implied_growth is not None:
            cagr_list.append(res.implied_growth)

    cagr_arr = np.array(cagr_list) if cagr_list else np.array([])
    n_valid_rev = len(cagr_arr)

    if n_valid_rev > 10:
        cp5, cp25, cp50, cp75, cp95 = np.percentile(cagr_arr, [5, 25, 50, 75, 95])
    else:
        cp5 = cp25 = cp50 = cp75 = cp95 = None

    # ── OAT tornado ──────────────────────────────────────────────────────────
    sensitivity = _tornado(
        base_rev, net_debt, tax_rate, capex_pct, years,
        wacc_mu, wacc_sig, g_mu, g_sig, op_mu, op_sig, tg_mu, tg_sig,
    )

    # ── Scenarios ─────────────────────────────────────────────────────────────
    scenarios: dict[str, dict] = {}
    if shares and shares > 0 and p50 is not None:
        for label, pct, desc in [
            ("Bear (5th pct)",  p5,  "Low growth + high WACC environment"),
            ("Base (50th pct)", p50, "Median of all simulated outcomes"),
            ("Bull (95th pct)", p95, "High growth + favourable discount rate"),
        ]:
            mos = (pct / current_price - 1.0) * 100 if current_price and pct else None
            scenarios[label] = {
                "Intrinsic / share": f"${pct:,.2f}" if pct else "—",
                "Margin of Safety":  f"{mos:+.1f}%" if mos is not None else "—",
                "Description":       desc,
            }

    return MonteCarloResult(
        n_sims=n_sims,
        n_valid_fwd=n_valid_fwd,
        n_valid_rev=n_valid_rev,
        iv_per_share=iv_ps,
        iv_equity=eq_valid,
        p5=float(p5)   if p5  is not None else None,
        p25=float(p25) if p25 is not None else None,
        p50=float(p50) if p50 is not None else None,
        p75=float(p75) if p75 is not None else None,
        p95=float(p95) if p95 is not None else None,
        prob_undervalued=prob_under,
        expected_return=exp_ret,
        implied_cagrs=cagr_arr,
        cagr_p5=float(cp5)   if cp5  is not None else None,
        cagr_p25=float(cp25) if cp25 is not None else None,
        cagr_p50=float(cp50) if cp50 is not None else None,
        cagr_p75=float(cp75) if cp75 is not None else None,
        cagr_p95=float(cp95) if cp95 is not None else None,
        sensitivity=sensitivity,
        scenarios=scenarios,
    )


__all__ = ["MonteCarloResult", "run_monte_carlo"]
