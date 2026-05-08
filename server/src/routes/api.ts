import { Router } from 'express';
import { getFundamentals } from '../services/dataProvider.js';
import { enterprisePV, intrinsicValuePerShare, marginOfSafety, reverseDcf } from '../engine/dcf.js';
import { rationalityCheck } from '../engine/rationality.js';
import { estimateWacc } from '../engine/wacc.js';
import type { DcfAssumptions, IntrinsicAssumptions } from '../types.js';

export const api = Router();

api.get('/quote/:ticker', async (req, res, next) => {
  try {
    const f = await getFundamentals(req.params.ticker);
    res.json(f);
  } catch (e) { next(e); }
});

api.get('/reverse-dcf/:ticker', async (req, res, next) => {
  try {
    const f = await getFundamentals(req.params.ticker);
    const wacc = Number(req.query.wacc) || estimateWacc({
      beta: f.beta,
      riskFreeRate: Number(req.query.rf) || 0.045,
      equityRiskPremium: Number(req.query.erp) || 0.055,
    });
    const terminalG = Number(req.query.tg) || 0.025;
    const highGrowthYears = Number(req.query.hgy) || 5;
    const fadeYears = Number(req.query.fy) || 5;

    const assumptions: DcfAssumptions = {
      fcf0: f.fcfTTM,
      shares: f.sharesOutstanding,
      netDebt: f.netDebt,
      wacc,
      terminalG,
      highGrowthYears,
      fadeYears,
    };

    if (!Number.isFinite(f.fcfTTM) || f.fcfTTM <= 0) {
      return res.status(422).json({
        error: 'Reverse-DCF requires positive TTM free cash flow.',
        ticker: f.ticker, fcfTTM: f.fcfTTM,
      });
    }

    const rd = reverseDcf(f.marketCap, assumptions);
    const rat = rationalityCheck(rd.impliedGrowth, f.historicalRevenueGrowth, f.historicalEpsGrowth);

    res.json({
      ticker: f.ticker,
      price: f.price,
      marketCap: f.marketCap,
      assumptions,
      impliedGrowth: rd.impliedGrowth,
      converged: rd.converged,
      analystGrowth5y: f.analystGrowth5y,
      rationality: rat,
    });
  } catch (e) { next(e); }
});

api.post('/intrinsic-dcf', (req, res) => {
  const a = req.body as IntrinsicAssumptions & { price: number };
  const fair = intrinsicValuePerShare(a);
  const mos = marginOfSafety(fair, a.price);
  res.json({ fairValue: fair, marginOfSafety: mos });
});

api.get('/sensitivity/:ticker', async (req, res, next) => {
  try {
    const f = await getFundamentals(req.params.ticker);
    const wacc = Number(req.query.wacc) || estimateWacc({
      beta: f.beta,
      riskFreeRate: 0.045,
      equityRiskPremium: 0.055,
    });
    const a: DcfAssumptions = {
      fcf0: f.fcfTTM, shares: f.sharesOutstanding, netDebt: f.netDebt,
      wacc, terminalG: 0.025, highGrowthYears: 5, fadeYears: 5,
    };
    const grid = [0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25];
    const points = grid.map((g) => {
      const ev = enterprisePV(g, a);
      const equity = ev - f.netDebt;
      return { growth: g, fairValue: equity / f.sharesOutstanding };
    });
    res.json({ ticker: f.ticker, price: f.price, points });
  } catch (e) { next(e); }
});
