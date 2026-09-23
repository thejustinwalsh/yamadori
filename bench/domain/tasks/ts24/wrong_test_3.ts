// Eager: spreads every input into an array first, which never finishes on an infinite iterable.
export function zip<T extends unknown[]>(...iterables: { [K in keyof T]: Iterable<T[K]> }): IterableIterator<T> {
  const arrays = (iterables as Iterable<unknown>[]).map((it) => {
    const out: unknown[] = [];
    for (const x of it) {
      out.push(x);
      if (out.length > 100_000) throw new RangeError('input too long');
    }
    return out;
  });
  const n = arrays.length === 0 ? 0 : Math.min(...arrays.map((a) => a.length));
  const rows = Array.from({ length: n }, (_, i) => arrays.map((a) => a[i]) as T);
  return rows[Symbol.iterator]();
}
