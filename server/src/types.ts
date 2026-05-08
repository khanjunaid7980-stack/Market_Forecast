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
  analystGrowth5y: number | null; // consensus, decimal
  historicalRevenueGrowth: number[]; // last ~5 years YoY, decimal
  historicalEpsGrowth: number[];
  asOf: string;
  source: string;
}

export interface DcfAssumptions {
  fcf0: number;
  shares: number;
  netDebt: number;
  wacc: number;
  terminalG: number;
  highGrowthYears: number;
  fadeYears: number;
}

export interface IntrinsicAssumptions {
  revenue0: number;
  shares: number;
  netDebt: number;
  revenueGrowth: number;
  operatingMargin: number;
  taxRate: number;
  reinvestmentRate: number; // capex+wc as % of EBIT(1-t)
  wacc: number;
  terminalG: number;
  years: number;
}
