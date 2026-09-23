import { useCallback, useState } from 'react';

export function useSet<T>(initial?: Iterable<T>): {
  values: ReadonlySet<T>;
  add: (v: T) => void;
  remove: (v: T) => void;
  toggle: (v: T) => void;
  has: (v: T) => boolean;
  clear: () => void;
} {
  const [values, setValues] = useState<ReadonlySet<T>>(() => new Set(initial));

  const add = useCallback((v: T) => {
    setValues((prev) => (prev.has(v) ? prev : new Set(prev).add(v)));
  }, []);

  const remove = useCallback((v: T) => {
    setValues((prev) => {
      if (!prev.has(v)) return prev;
      const next = new Set(prev);
      next.delete(v);
      return next;
    });
  }, []);

  const toggle = useCallback((v: T) => {
    setValues((prev) => {
      const next = new Set(prev);
      if (prev.has(v)) next.delete(v);
      else next.add(v);
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setValues((prev) => (prev.size === 0 ? prev : new Set<T>()));
  }, []);

  const has = useCallback((v: T) => values.has(v), [values]);

  return { values, add, remove, toggle, has, clear };
}
