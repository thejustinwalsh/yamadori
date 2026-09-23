// Caches the first result: later calls with different arguments return a stale value.
export function partial<H extends unknown[], T extends unknown[], R>(
  fn: (...args: [...H, ...T]) => R,
  ...head: H
): (...tail: T) => R {
  let cached: { v: R } | undefined;
  return (...tail: T) => (cached ??= { v: fn(...head, ...tail) }).v;
}
