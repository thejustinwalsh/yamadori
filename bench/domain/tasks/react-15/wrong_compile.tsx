import { useState } from 'react';

// Wrong (strict): `new Set()` without a type argument is a Set<unknown>, which
// is not a ReadonlySet<T>.
export function useSet<T>(initial?: Iterable<T>): {
  values: ReadonlySet<T>;
  add: (v: T) => void;
  remove: (v: T) => void;
  toggle: (v: T) => void;
  has: (v: T) => boolean;
  clear: () => void;
} {
  const [values, setValues] = useState(() => {
    const s = new Set();
    if (initial) for (const v of initial) s.add(v);
    return s;
  });

  return {
    values,
    add: (v) => setValues((p) => (p.has(v) ? p : new Set(p).add(v))),
    remove: (v) =>
      setValues((p) => {
        if (!p.has(v)) return p;
        const n = new Set(p);
        n.delete(v);
        return n;
      }),
    toggle: (v) =>
      setValues((p) => {
        const n = new Set(p);
        if (!n.delete(v)) n.add(v);
        return n;
      }),
    has: (v) => values.has(v),
    clear: () => setValues((p) => (p.size ? new Set() : p)),
  };
}
