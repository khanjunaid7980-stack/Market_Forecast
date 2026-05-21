import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, ReferenceLine } from 'recharts';
import type { Fundamentals } from '../types';
import { pct } from '../lib/format';

export function HistoricalChart({ f, impliedGrowth }: { f: Fundamentals; impliedGrowth: number | null }) {
  const data = f.historicalRevenueGrowth.map((g, i) => ({
    year: `Y-${f.historicalRevenueGrowth.length - i}`,
    revenue: g * 100,
    eps: (f.historicalEpsGrowth[i] ?? 0) * 100,
  }));
  return (
    <div className="panel p-4">
      <div className="flex items-baseline justify-between mb-2">
        <h2 className="label">Historical Growth vs. Market Implied</h2>
        <span className="text-xs text-term-muted">Dashed line = market-implied growth.</span>
      </div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
            <CartesianGrid stroke="#1f2630" strokeDasharray="2 4" />
            <XAxis dataKey="year" stroke="#7d8693" fontSize={11} />
            <YAxis tickFormatter={(v) => `${v.toFixed(0)}%`} stroke="#7d8693" fontSize={11} />
            <Tooltip
              contentStyle={{ background: '#13171c', border: '1px solid #1f2630', color: '#d8dde3' }}
              formatter={(v: number) => `${v.toFixed(2)}%`}
            />
            <Bar dataKey="revenue" fill="#27e0c5" />
            <Bar dataKey="eps" fill="#ff5fa2" />
            {impliedGrowth != null && (
              <ReferenceLine y={impliedGrowth * 100} stroke="#ffb000" strokeDasharray="4 4"
                label={{ value: `Implied ${pct(impliedGrowth)}`, fill: '#ffb000', fontSize: 11 }} />
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
