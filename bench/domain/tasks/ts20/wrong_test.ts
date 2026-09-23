// Awaits each key in turn: an early rejection is left unhandled while an earlier key is still pending.
export async function props<T extends object>(obj: T): Promise<{ [K in keyof T]: Awaited<T[K]> }> {
  const out: Record<string, unknown> = {};
  for (const k of Object.keys(obj)) out[k] = await obj[k as keyof T];
  return out as { [K in keyof T]: Awaited<T[K]> };
}
