import type { Fundamentals } from '../types';
import { big, num, pct, usd } from '../lib/format';

export function TickerHeader({ f }: { f: Fundamentals }) {
  return (
    <div className="panel p-4 grid grid-cols-2 md:grid-cols-6 gap-4">
      <div className="col-span-2">
        <div className="label">Security</div>
        <div className="text-2xl amber font-semibold">{f.ticker}</div>
        <div className="text-xs text-term-muted truncate">{f.name}</div>
      </div>
      <Cell label="Last" value={`$${num(f.price)}`} accent="amber" />
      <Cell label="Mkt Cap" value={usd(f.marketCap)} />
      <Cell label="Trail P/E" value={num(f.peTTM)} />
      <Cell label="Fwd P/E" value={num(f.forwardPE)} />
      <Cell label="PEG" value={num(f.peg)} />
      <Cell label="Beta" value={num(f.beta)} />
      <Cell label="FCF TTM" value={usd(f.fcfTTM)} />
      <Cell label="Net Debt" value={usd(f.netDebt)} />
      <Cell label="Shares" value={big(f.sharesOutstanding)} />
      <Cell label="Analyst 5y g" value={pct(f.analystGrowth5y)} />
    </div>
  );
}

function Cell({ label, value, accent }: { label: string; value: string; accent?: 'amber' | 'cyan' }) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className={`num text-base ${accent ?? ''}`}>{value}</div>
    </div>
  );
}
