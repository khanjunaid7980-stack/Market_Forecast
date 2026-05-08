import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { ReverseDcfResponse } from '../types';
import { num, pct } from '../lib/format';

interface Props { ticker: string }

const VERDICT_COLOR: Record<string, string> = {
  Rational: 'green',
  Stretched: 'amber',
  Speculative: 'red',
  'Insufficient Data': 'cyan',
};

export function ReverseDcfPanel({ ticker }: Props) {
  const [wacc, setWacc] = useState(0.09);
  const [tg, setTg] = useState(0.025);
  const [hgy, setHgy] = useState(5);
  const [data, setData] = useState<ReverseDcfResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true); setErr(null);
    api.reverseDcf(ticker, { wacc, tg, hgy })
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) setErr(String(e.message ?? e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [ticker, wacc, tg, hgy]);

  return (
    <div className="panel p-4 space-y-3">
      <div className="flex items-baseline justify-between">
        <h2 className="label">Auto-Predictive — Reverse DCF</h2>
        <span className="text-xs text-term-muted">Solves for the growth rate the market is paying for.</span>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <Slider label="WACC" value={wacc} min={0.04} max={0.20} step={0.005} onChange={setWacc} display={pct(wacc)} />
        <Slider label="Terminal g" value={tg} min={0.0} max={0.05} step={0.0025} onChange={setTg} display={pct(tg)} />
        <Slider label="High-growth yrs" value={hgy} min={3} max={10} step={1} onChange={setHgy} display={String(hgy)} />
      </div>

      {err && <div className="text-sm red">{err}</div>}
      {loading && !data && <div className="text-sm text-term-muted">Solving…</div>}

      {data && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2">
          <Stat label="Market-implied growth" value={pct(data.impliedGrowth)} accent="amber" big />
          <Stat label="Analyst consensus 5y" value={pct(data.analystGrowth5y)} />
          <Stat label="5y hist mean" value={pct(data.rationality.historicalMean)} />
          <Stat label="5y hist stdev" value={pct(data.rationality.historicalStd)} />
          <Stat
            label="Rationality z-score"
            value={num(data.rationality.zScore)}
            accent={VERDICT_COLOR[data.rationality.verdict] as 'green' | 'amber' | 'red' | 'cyan'}
          />
          <Stat
            label="Verdict"
            value={data.rationality.verdict}
            accent={VERDICT_COLOR[data.rationality.verdict] as 'green' | 'amber' | 'red' | 'cyan'}
            big
          />
          <Stat label="Solver converged" value={data.converged ? 'yes' : 'no'} />
          <Stat label="WACC used" value={pct(data.assumptions.wacc)} />
        </div>
      )}
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, display }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (n: number) => void; display: string;
}) {
  return (
    <label className="block">
      <div className="flex justify-between"><span className="label">{label}</span><span className="num text-xs amber">{display}</span></div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-term-amber" />
    </label>
  );
}

function Stat({ label, value, accent, big: isBig }: {
  label: string; value: string; accent?: 'amber' | 'green' | 'red' | 'cyan'; big?: boolean;
}) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className={`num ${isBig ? 'text-2xl' : 'text-base'} ${accent ?? ''}`}>{value}</div>
    </div>
  );
}
