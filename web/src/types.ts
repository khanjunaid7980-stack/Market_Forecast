export interface Fundamentals {
  ticker: string;
  name: string;
  price: number;
  sharesOutstanding: number;
  marketCap: number;
  netDebt: number;
  fcfTTM: number;
  epsTTM: number;
  peTTM: number | null;
  forwardPE: number | null;
  peg: number | null;
  beta: number | null;
  analystGrowth5y: number | null;
  historicalRevenueGrowth: number[];
  historicalEpsGrowth: number[];
  asOf: string;
  source: string;
}

export interface ReverseDcfResponse {
  ticker: string;
  price: number;
  marketCap: number;
  assumptions: {
    fcf0: number; shares: number; netDebt: number;
    wacc: number; terminalG: number; highGrowthYears: number; fadeYears: number;
  };
  impliedGrowth: number;
  converged: boolean;
  analystGrowth5y: number | null;
  rationality: {
    impliedGrowth: number;
    historicalMean: number;
    historicalStd: number;
    zScore: number;
    verdict: 'Rational' | 'Stretched' | 'Speculative' | 'Insufficient Data';
  };
}

export interface SensitivityPoint { growth: number; fairValue: number }
export interface SensitivityResponse {
  ticker: string; price: number; points: SensitivityPoint[];
}
