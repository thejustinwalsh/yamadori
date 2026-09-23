// Plausible and wrong: drops the remainder instead of emitting a short chunk.
export function chunk<T>(items: readonly T[], size: number): T[][] {
  if (!Number.isInteger(size) || size <= 0) throw new RangeError('bad size');
  const out: T[][] = [];
  for (let i = 0; i + size <= items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}
