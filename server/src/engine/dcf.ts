import type { DcfAssumptions, IntrinsicAssumptions } from '../types.js';

/**
 * Two-stage DCF on enterprise FCF with a linear growth fade between
 * the explicit high-growth period and the terminal rate.
 * Returns enterprise present value (pre net-debt).
 */
export function enterprisePV(growthRate: number, a: DcfAssumptions): number {
  const { fcf0, wacc, terminalG, highGrowthYears, fadeYears } = a;
  if (wacc <= terminalG) return Number.POSITIVE_INFINITY;
  let pv = 0;
  let fcf = fcf0;
  let t = 0;
  for (let i = 1; i <= highGrowthYears; i++) {
    fcf *= 1 + growthRate;
    t = i;
    pv += fcf / Math.pow(1 + wacc, t);
  }
  for (let i = 1; i <= fadeYears; i++) {
    const g = growthRate + (terminalG - growthRate) * (i / fadeYears);
    fcf *= 1 + g;
    t = highGrowthYears + i;
    pv += fcf / Math.pow(1 + wacc, t);
  }
  const tv = (fcf * (1 + terminalG)) / (wacc - terminalG);
  pv += tv / Math.pow(1 + wacc, t);
  return pv;
}

export function intrinsicValuePerShare(a: IntrinsicAssumptions): number {
  if (a.wacc <= a.terminalG) return Number.POSITIVE_INFINITY;
  let revenue = a.revenue0;
  let pv = 0;
  let fcf = 0;
  for (let t = 1; t <= a.years; t++) {
    revenue *= 1 + a.revenueGrowth;
    const ebit = revenue * a.operatingMargin;
    const nopat = ebit * (1 - a.taxRate);
    fcf = nopat * (1 - a.reinvestmentRate);
    pv += fcf / Math.pow(1 + a.wacc, t);
  }
  const tv = (fcf * (1 + a.terminalG)) / (a.wacc - a.terminalG);
  pv += tv / Math.pow(1 + a.wacc, a.years);
  const equity = pv - a.netDebt;
  return equity / a.shares;
}

/**
 * Reverse-DCF: bisection solver for the high-growth rate that makes
 * the model's enterprise value equal the observed enterprise value
 * (market cap + net debt).
 */
export function reverseDcf(
  marketCap: number,
  a: DcfAssumptions,
): { impliedGrowth: number; converged: boolean; iterations: number } {
  const target = marketCap + a.netDebt;
  let lo = -0.30;
  let hi = 1.00;
  // Ensure bracket. enterprisePV is monotonically increasing in growthRate.
  if (enterprisePV(hi, a) < target) {
    return { impliedGrowth: hi, converged: false, iterations: 0 };
  }
  if (enterprisePV(lo, a) > target) {
    return { impliedGrowth: lo, converged: false, iterations: 0 };
  }
  let iter = 0;
  while (hi - lo > 1e-6 && iter < 100) {
    const mid = (lo + hi) / 2;
    if (enterprisePV(mid, a) < target) lo = mid;
    else hi = mid;
    iter++;
  }
  return { impliedGrowth: (lo + hi) / 2, converged: true, iterations: iter };
}

export function marginOfSafety(intrinsic: number, market: number): number {
  if (!Number.isFinite(intrinsic) || intrinsic <= 0) return 0;
  return (intrinsic - market) / intrinsic;
}
