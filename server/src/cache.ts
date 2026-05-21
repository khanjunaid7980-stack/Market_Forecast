interface Entry<T> { value: T; expiresAt: number }

const store = new Map<string, Entry<unknown>>();
const DEFAULT_TTL = Number(process.env.CACHE_TTL_SECONDS ?? 600) * 1000;

export function cacheGet<T>(key: string): T | undefined {
  const e = store.get(key) as Entry<T> | undefined;
  if (!e) return undefined;
  if (Date.now() > e.expiresAt) { store.delete(key); return undefined; }
  return e.value;
}

export function cacheSet<T>(key: string, value: T, ttlMs = DEFAULT_TTL): T {
  store.set(key, { value, expiresAt: Date.now() + ttlMs });
  return value;
}

export async function memo<T>(key: string, loader: () => Promise<T>, ttlMs = DEFAULT_TTL): Promise<T> {
  const hit = cacheGet<T>(key);
  if (hit !== undefined) return hit;
  const value = await loader();
  return cacheSet(key, value, ttlMs);
}
