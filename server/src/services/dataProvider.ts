import type { Fundamentals } from '../types.js';
import { memo } from '../cache.js';
import { loadYahoo } from './yahooFinance.js';
import { enrichWithFmp } from './fmp.js';

export function getFundamentals(ticker: string): Promise<Fundamentals> {
  const key = `fund:${ticker.toUpperCase()}`;
  return memo(key, async () => {
    const base = await loadYahoo(ticker);
    return enrichWithFmp(base);
  });
}
