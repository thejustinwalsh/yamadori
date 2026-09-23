// Wrong: mutates the current Set and then stores a copy, so the component
// re-renders but every earlier `values` snapshot changes underneath its holder.
import { useState } from 'react';

export function useSet<T>(initial?: Iterable<T>) {
  const [values, setValues] = useState<Set<T>>(() => new Set(initial));

  return {
    values: values as ReadonlySet<T>,
    add(v: T) {
      if (values.has(v)) return;
      values.add(v);
      setValues(new Set(values));
    },
    remove(v: T) {
      if (!values.delete(v)) return;
      setValues(new Set(values));
    },
    toggle(v: T) {
      if (values.has(v)) values.delete(v);
      else values.add(v);
      setValues(new Set(values));
    },
    has: (v: T) => values.has(v),
    clear() {
      if (values.size === 0) return;
      values.clear();
      setValues(new Set(values));
    },
  };
}
