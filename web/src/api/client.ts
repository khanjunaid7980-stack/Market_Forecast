import type { Fundamentals, ReverseDcfResponse, SensitivityResponse } from '../types';

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const body = await r.text();
    throw new Error(`${r.status} ${r.statusText}: ${body}`);
  }
  return r.json();
}

export const api = {
  quote: (t: string) => fetch(`/api/quote/${t}`).then(j<Fundamentals>),
  reverseDcf: (t: string, params: Record<string, number> = {}) => {
    const qs = new URLSearchParams(
      Object.entries(params).map(([k, v]) => [k, String(v)]),
    ).toString();
    return fetch(`/api/reverse-dcf/${t}${qs ? `?${qs}` : ''}`).then(j<ReverseDcfResponse>);
  },
  sensitivity: (t: string) => fetch(`/api/sensitivity/${t}`).then(j<SensitivityResponse>),
  intrinsic: (body: Record<string, number>) =>
    fetch('/api/intrinsic-dcf', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }).then(j<{ fairValue: number; marginOfSafety: number }>),
};
