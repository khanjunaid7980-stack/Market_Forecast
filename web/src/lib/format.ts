export const pct = (x: number | null | undefined, digits = 2) =>
  x == null || !Number.isFinite(x) ? '—' : `${(x * 100).toFixed(digits)}%`;

export const num = (x: number | null | undefined, digits = 2) =>
  x == null || !Number.isFinite(x) ? '—' : x.toFixed(digits);

export const big = (x: number | null | undefined) => {
  if (x == null || !Number.isFinite(x)) return '—';
  const abs = Math.abs(x);
  if (abs >= 1e12) return `${(x / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${(x / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(x / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(x / 1e3).toFixed(2)}K`;
  return x.toFixed(2);
};

export const usd = (x: number | null | undefined) =>
  x == null || !Number.isFinite(x) ? '—' : `$${big(x)}`;
