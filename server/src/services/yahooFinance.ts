import yahooFinance from 'yahoo-finance2';
import type { Fundamentals } from '../types.js';

yahooFinance.suppressNotices?.(['yahooSurvey', 'ripHistorical']);

function yoy(arr: number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < arr.length; i++) {
    const prev = arr[i - 1];
    if (prev && Number.isFinite(prev) && prev !== 0) out.push((arr[i] - prev) / Math.abs(prev));
  }
  return out;
}

export async function loadYahoo(ticker: string): Promise<Fundamentals> {
  const modules = [
    'price','summaryDetail','defaultKeyStatistics','financialData',
    'incomeStatementHistory','cashflowStatementHistory','balanceSheetHistory',
    'earningsTrend'
  ] as const;
  const q = await yahooFinance.quoteSummary(ticker, { modules: modules as unknown as string[] });

  const price = q.price?.regularMarketPrice ?? 0;
  const shares = q.defaultKeyStatistics?.sharesOutstanding ?? q.price?.sharesOutstanding ?? 0;
  const marketCap = q.price?.marketCap ?? price * shares;
  const totalDebt = q.financialData?.totalDebt ?? 0;
  const cash = q.financialData?.totalCash ?? 0;
  const netDebt = totalDebt - cash;
  const fcfTTM = q.financialData?.freeCashflow ?? 0;
  const epsTTM = q.defaultKeyStatistics?.trailingEps ?? 0;
  const pe = q.summaryDetail?.trailingPE ?? null;
  const fwdPe = q.summaryDetail?.forwardPE ?? null;
  const peg = q.defaultKeyStatistics?.pegRatio ?? null;
  const beta = q.defaultKeyStatistics?.beta ?? null;

  const trend5y = q.earningsTrend?.trend?.find((t) => t.period === '+5y')?.growth;

  const incomes = (q.incomeStatementHistory?.incomeStatementHistory ?? [])
    .map((s) => Number(s.totalRevenue ?? 0))
    .filter((n) => Number.isFinite(n) && n > 0)
    .reverse();
  const epsHist = (q.incomeStatementHistory?.incomeStatementHistory ?? [])
    .map((s) => {
      const ni = Number(s.netIncome ?? 0);
      return shares > 0 ? ni / shares : 0;
    })
    .filter((n) => Number.isFinite(n))
    .reverse();

  return {
    ticker: ticker.toUpperCase(),
    name: q.price?.longName ?? q.price?.shortName ?? ticker.toUpperCase(),
    price, sharesOutstanding: shares, marketCap, netDebt,
    fcfTTM, epsTTM, peTTM: pe, forwardPE: fwdPe, peg, beta,
    analystGrowth5y: typeof trend5y === 'number' ? trend5y : null,
    historicalRevenueGrowth: yoy(incomes),
    historicalEpsGrowth: yoy(epsHist),
    asOf: new Date().toISOString(),
    source: 'yahoo-finance2',
  };
}
