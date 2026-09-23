// Loose typing: every row is unknown[], so the per-position element types are lost.
export function* zip(...iterables: Iterable<unknown>[]): Generator<unknown[], void, undefined> {
  const iterators = iterables.map((it) => it[Symbol.iterator]());
  const done = new Set<Iterator<unknown>>();
  try {
    if (iterators.length === 0) return;
    while (true) {
      const row: unknown[] = [];
      for (const it of iterators) {
        const r = it.next();
        if (r.done) { done.add(it); return; }
        row.push(r.value);
      }
      yield row;
    }
  } finally {
    for (const it of iterators) if (!done.has(it)) it.return?.();
  }
}
