// Uses allSettled: a rejection is swallowed and the rejected key resolves to undefined.
export async function props<T extends object>(obj: T): Promise<{ [K in keyof T]: Awaited<T[K]> }> {
  const keys = Object.keys(obj);
  const settled = await Promise.allSettled(keys.map((k) => obj[k as keyof T]));
  const out: Record<string, unknown> = {};
  keys.forEach((k, i) => {
    const s = settled[i]!;
    out[k] = s.status === 'fulfilled' ? s.value : undefined;
  });
  return out as { [K in keyof T]: Awaited<T[K]> };
}
