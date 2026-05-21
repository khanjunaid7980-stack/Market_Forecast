import { useEffect, useState } from 'react';
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api/client';
import type { SensitivityResponse } from '../types';
import { num, pct } from '../lib/format';

export function SensitivityChart({ ticker }: { ticker: string }) {
  const [data, setData] = useState<SensitivityResponse | null>(null);
  useEffect(() => {
    let cancelled = false;
    api.sensitivity(ticker).then((d) => { if (!cancelled) setData(d); }).catch(() => {});
    return () => { cancelled = true; };
  }, [ticker]);

  if (!data) return <div className="panel p-4 text-sm text-term-muted">Loading sensitivity…</div>;

  const chartData = data.points.map((p) => ({ growth: p.growth * 100, fairValue: p.fairValue }));
  return (
    <div className="panel p-4">
      <div className="flex items-baseline justify-between mb-2">
        <h2 className="label">Rationality Gap — Fair Value vs. Growth</h2>
        <span className="text-xs text-term-muted">Reference line = current market price.</span>
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
            <CartesianGrid stroke="#1f2630" strokeDasharray="2 4" />
            <XAxis dataKey="growth" tickFormatter={(v) => `${v.toFixed(0)}%`} stroke="#7d8693" fontSize={11} />
            <YAxis tickFormatter={(v) => `$${num(v, 0)}`} stroke="#7d8693" fontSize={11} />
            <Tooltip
              contentStyle={{ background: '#13171c', border: '1px solid #1f2630', color: '#d8dde3' }}
              formatter={(v: number) => `$${num(v)}`}
              labelFormatter={(v: number) => `g = ${pct(v / 100)}`}
            />
            <ReferenceLine y={data.price} stroke="#ffb000" strokeDasharray="4 4" label={{ value: `Price $${num(data.price)}`, fill: '#ffb000', fontSize: 11 }} />
            <Line type="monotone" dataKey="fairValue" stroke="#27e0c5" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
