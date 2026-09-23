// Wrong: every call makes a new Set, even when nothing changes, so no-op
// calls hand out a new `values` object.
import { useState } from 'react';

export function useSet<T>(initial?: Iterable<T>) {
  const [values, setValues] = useState<ReadonlySet<T>>(() => new Set(initial));

  const update = (fn: (s: Set<T>) => void) =>
    setValues((prev) => {
      const next = new Set(prev);
      fn(next);
      return next;
    });

  return {
    values,
    add: (v: T) => update((s) => s.add(v)),
    remove: (v: T) => update((s) => s.delete(v)),
    toggle: (v: T) => update((s) => (s.has(v) ? s.delete(v) : s.add(v))),
    has: (v: T) => values.has(v),
    clear: () => update((s) => s.clear()),
  };
}
