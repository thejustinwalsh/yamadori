export function* zip<T extends unknown[]>(
  ...iterables: { [K in keyof T]: Iterable<T[K]> }
): Generator<T, void, undefined> {
  const iterators = iterables.map((it) => it[Symbol.iterator]()) as Iterator<unknown>[];
  const done = new Set<Iterator<unknown>>();
  try {
    if (iterators.length === 0) return;
    while (true) {
      const row: unknown[] = [];
      for (const it of iterators) {
        const r = it.next();
        if (r.done) {
          done.add(it);
          return;
        }
        row.push(r.value);
      }
      yield row as T;
    }
  } finally {
    for (const it of iterators) if (!done.has(it)) it.return?.();
  }
}
