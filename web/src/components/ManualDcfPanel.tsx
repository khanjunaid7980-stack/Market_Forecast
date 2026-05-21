import { useEffect, useMemo, useState } from 'react';
import { api } from '../api/client';
import type { Fundamentals } from '../types';
import { num, pct } from '../lib/format';

export function ManualDcfPanel({ f }: { f: Fundamentals }) {
  const [revGrowth, setRevGrowth] = useState(0.08);
  const [opMargin, setOpMargin] = useState(0.20);
  const [taxRate, setTaxRate] = useState(0.21);
  const [reinvest, setReinvest] = useState(0.30);
  const [wacc, setWacc] = useState(0.09);
  const [tg, setTg] = useState(0.025);
  const [years, setYears] = useState(10);

  const revenue0 = useMemo(() => {
    if (!f.peTTM || !f.epsTTM || !f.sharesOutstanding) return f.fcfTTM * 5;
    const ni = f.epsTTM * f.sharesOutstanding;
    return ni / Math.max(opMargin * (1 - taxRate), 0.05);
  }, [f, opMargin, taxRate]);

  const [out, setOut] = useState<{ fairValue: number; marginOfSafety: number } | null>(null);
  useEffect(() => {
    let cancelled = false;
    api.intrinsic({
      revenue0, shares: f.sharesOutstanding, netDebt: f.netDebt,
      revenueGrowth: revGrowth, operatingMargin: opMargin, taxRate,
      reinvestmentRate: reinvest, wacc, terminalG: tg, years,
      price: f.price,
    }).then((d) => { if (!cancelled) setOut(d); }).catch(() => { if (!cancelled) setOut(null); });
    return () => { cancelled = true; };
  }, [f, revenue0, revGrowth, opMargin, taxRate, reinvest, wacc, tg, years]);

  const mosColor = out && out.marginOfSafety >= 0.15 ? 'green' : out && out.marginOfSafety >= 0 ? 'amber' : 'red';

  return (
    <div className="panel p-4 space-y-3">
      <div className="flex items-baseline justify-between">
        <h2 className="label">Manual Sensitivity — Intrinsic DCF</h2>
        <span className="text-xs text-term-muted">Toggle assumptions; intrinsic value updates live.</span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Field label="Revenue growth" value={revGrowth} onChange={setRevGrowth} pct />
        <Field label="Operating margin" value={opMargin} onChange={setOpMargin} pct />
        <Field label="Tax rate" value={taxRate} onChange={setTaxRate} pct />
        <Field label="Reinvestment rate" value={reinvest} onChange={setReinvest} pct />
        <Field label="WACC" value={wacc} onChange={setWacc} pct />
        <Field label="Terminal g" value={tg} onChange={setTg} pct />
        <Field label="Forecast years" value={years} onChange={setYears} />
      </div>

      <div className="grid grid-cols-3 gap-4 pt-2">
        <Stat label="Intrinsic / share" value={out ? `$${num(out.fairValue)}` : '—'} accent="cyan" big />
        <Stat label="Market price" value={`$${num(f.price)}`} accent="amber" />
        <Stat label="Margin of safety" value={out ? pct(out.marginOfSafety) : '—'} accent={mosColor} big />
      </div>
    </div>
  );
}

function Field({ label, value, onChange, pct: isPct }: {
  label: string; value: number; onChange: (n: number) => void; pct?: boolean;
}) {
  return (
    <label className="block">
      <div className="label">{label}</div>
      <div className="flex items-center gap-2">
        <input className="term num w-full" type="number"
          step={isPct ? 0.005 : 1}
          value={isPct ? +(value * 100).toFixed(2) : value}
          onChange={(e) => onChange(isPct ? Number(e.target.value) / 100 : Number(e.target.value))}
        />
        <span className="text-xs text-term-muted">{isPct ? '%' : ''}</span>
      </div>
    </label>
  );
}

function Stat({ label, value, accent, big: isBig }: {
  label: string; value: string; accent?: 'amber' | 'cyan' | 'green' | 'red'; big?: boolean;
}) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className={`num ${isBig ? 'text-2xl' : 'text-base'} ${accent ?? ''}`}>{value}</div>
    </div>
  );
}
