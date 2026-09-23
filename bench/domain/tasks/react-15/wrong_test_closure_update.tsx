// Wrong: builds each new Set from the rendered `values` instead of the
// pending state, so only the last of several calls in one handler survives.
import { useState } from 'react';

export function useSet<T>(initial?: Iterable<T>) {
  const [values, setValues] = useState<ReadonlySet<T>>(() => new Set(initial));

  const add = (v: T) => {
    if (!values.has(v)) setValues(new Set([...values, v]));
  };
  const remove = (v: T) => {
    if (values.has(v)) setValues(new Set([...values].filter((x) => x !== v)));
  };
  const toggle = (v: T) => (values.has(v) ? remove(v) : add(v));
  const clear = () => {
    if (values.size > 0) setValues(new Set());
  };

  return { values, add, remove, toggle, has: (v: T) => values.has(v), clear };
}
