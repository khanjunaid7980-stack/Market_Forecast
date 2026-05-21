// Optional fallback against Financial Modeling Prep free tier.
// Used only if FMP_API_KEY is set; merges into Yahoo result.
import type { Fundamentals } from '../types.js';

const BASE = 'https://financialmodelingprep.com/api/v3';

async function get<T>(path: string): Promise<T | null> {
  const key = process.env.FMP_API_KEY;
  if (!key) return null;
  const url = `${BASE}${path}${path.includes('?') ? '&' : '?'}apikey=${key}`;
  const r = await fetch(url);
  if (!r.ok) return null;
  return (await r.json()) as T;
}

export async function enrichWithFmp(base: Fundamentals): Promise<Fundamentals> {
  if (!process.env.FMP_API_KEY) return base;
  const profile = await get<Array<{ beta?: number; companyName?: string }>>(`/profile/${base.ticker}`);
  const growth = await get<Array<{ revenueGrowth?: number }>>(`/financial-growth/${base.ticker}?limit=5`);
  const out = { ...base };
  if (profile?.[0]?.beta && !out.beta) out.beta = profile[0].beta;
  if (growth && growth.length > 0) {
    const series = growth
      .map((g) => Number(g.revenueGrowth))
      .filter((n) => Number.isFinite(n));
    if (series.length > out.historicalRevenueGrowth.length) {
      out.historicalRevenueGrowth = series.reverse();
    }
  }
  out.source = `${out.source}+fmp`;
  return out;
}
