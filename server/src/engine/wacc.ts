// Lightweight WACC heuristic. For a free-tier tool, exact capital-structure
// detail is rarely available, so we approximate using CAPM on equity and a
// flat cost of debt assumption. Caller can always override via the manual UI.
export interface WaccInputs {
  beta: number | null;
  riskFreeRate: number; // e.g. 0.045
  equityRiskPremium: number; // e.g. 0.055
  costOfDebt?: number; // pre-tax, default 0.05
  taxRate?: number; // default 0.21
  debtWeight?: number; // default 0.20
}

export function estimateWacc(i: WaccInputs): number {
  const beta = i.beta ?? 1.0;
  const ke = i.riskFreeRate + beta * i.equityRiskPremium;
  const kd = (i.costOfDebt ?? 0.05) * (1 - (i.taxRate ?? 0.21));
  const wd = i.debtWeight ?? 0.20;
  return (1 - wd) * ke + wd * kd;
}
