// Wrong: mutates the Set in place and hands the same object back to
// setState, so React bails out and never re-renders.
import { useState } from 'react';

export function useSet<T>(initial?: Iterable<T>) {
  const [values, setValues] = useState(() => new Set<T>(initial));

  return {
    values: values as ReadonlySet<T>,
    add: (v: T) => setValues((s) => s.add(v)),
    remove: (v: T) =>
      setValues((s) => {
        s.delete(v);
        return s;
      }),
    toggle: (v: T) =>
      setValues((s) => {
        if (s.has(v)) s.delete(v);
        else s.add(v);
        return s;
      }),
    has: (v: T) => values.has(v),
    clear: () =>
      setValues((s) => {
        s.clear();
        return s;
      }),
  };
}
