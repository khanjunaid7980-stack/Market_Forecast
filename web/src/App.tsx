import { useEffect, useState } from 'react';
import { api } from './api/client';
import type { Fundamentals, ReverseDcfResponse } from './types';
import { TickerHeader } from './components/TickerHeader';
import { ReverseDcfPanel } from './components/ReverseDcfPanel';
import { ManualDcfPanel } from './components/ManualDcfPanel';
import { SensitivityChart } from './components/SensitivityChart';
import { HistoricalChart } from './components/HistoricalChart';

export default function App() {
  const [input, setInput] = useState('AAPL');
  const [ticker, setTicker] = useState('AAPL');
  const [f, setF] = useState<Fundamentals | null>(null);
  const [reverse, setReverse] = useState<ReverseDcfResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setErr(null); setF(null); setReverse(null);
    api.quote(ticker).then((d) => { if (!cancelled) setF(d); }).catch((e) => { if (!cancelled) setErr(String(e.message)); });
    api.reverseDcf(ticker).then((d) => { if (!cancelled) setReverse(d); }).catch(() => {});
    return () => { cancelled = true; };
  }, [ticker]);

  return (
    <div className="min-h-screen p-4 md:p-6 space-y-4">
      <header className="flex items-center gap-3">
        <div className="text-xl font-semibold amber">RvM</div>
        <div className="text-xs text-term-muted hidden md:block">Rationality vs. Market — Forecast Terminal</div>
        <form
          className="ml-auto flex gap-2"
          onSubmit={(e) => { e.preventDefault(); setTicker(input.trim().toUpperCase()); }}
        >
          <input
            className="term w-40"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ticker (e.g. NVDA)"
            spellCheck={false}
          />
          <button className="term amber px-3 hover:border-term-amber" type="submit">Run</button>
        </form>
      </header>

      {err && <div className="panel p-3 red text-sm">{err}</div>}

      {f && <TickerHeader f={f} />}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {f && <ReverseDcfPanel ticker={f.ticker} />}
        {f && <ManualDcfPanel f={f} />}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {f && <SensitivityChart ticker={f.ticker} />}
        {f && <HistoricalChart f={f} impliedGrowth={reverse?.impliedGrowth ?? null} />}
      </div>

      <footer className="text-xs text-term-muted pt-2">
        Free-tier data via Yahoo Finance · optional FMP enrichment · educational use only.
      </footer>
    </div>
  );
}
