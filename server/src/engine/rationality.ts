export interface RationalityResult {
  impliedGrowth: number;
  historicalMean: number;
  historicalStd: number;
  zScore: number;
  verdict: 'Rational' | 'Stretched' | 'Speculative' | 'Insufficient Data';
}

function stats(xs: number[]): { mean: number; std: number } {
  const filtered = xs.filter((x) => Number.isFinite(x));
  if (filtered.length < 2) return { mean: NaN, std: NaN };
  const mean = filtered.reduce((a, b) => a + b, 0) / filtered.length;
  const variance = filtered.reduce((s, x) => s + (x - mean) ** 2, 0) / (filtered.length - 1);
  return { mean, std: Math.sqrt(variance) };
}

/**
 * Rationality gate: compare the market-implied growth rate against the
 * 5y empirical distribution of fundamental growth (revenue + EPS pooled).
 * z > 2 => Speculative, z > 1 => Stretched, otherwise Rational.
 */
export function rationalityCheck(
  impliedGrowth: number,
  historicalRevenueGrowth: number[],
  historicalEpsGrowth: number[],
): RationalityResult {
  const pool = [...historicalRevenueGrowth, ...historicalEpsGrowth];
  const { mean, std } = stats(pool);
  if (!Number.isFinite(mean) || !Number.isFinite(std) || std === 0) {
    return {
      impliedGrowth,
      historicalMean: Number.isFinite(mean) ? mean : 0,
      historicalStd: Number.isFinite(std) ? std : 0,
      zScore: 0,
      verdict: 'Insufficient Data',
    };
  }
  const z = (impliedGrowth - mean) / std;
  let verdict: RationalityResult['verdict'];
  if (z > 2) verdict = 'Speculative';
  else if (z > 1) verdict = 'Stretched';
  else verdict = 'Rational';
  return { impliedGrowth, historicalMean: mean, historicalStd: std, zScore: z, verdict };
}
