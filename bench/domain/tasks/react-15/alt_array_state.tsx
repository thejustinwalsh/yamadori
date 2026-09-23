import { useMemo, useReducer } from 'react';

// Keeps an immutable array in a reducer and derives the Set from it with
// useMemo: a new array means a new Set, an unchanged array the same one.
type Act<T> = { op: 'add' | 'remove' | 'toggle'; v: T } | { op: 'clear' };

function reduce<T>(items: readonly T[], a: Act<T>): readonly T[] {
  if (a.op === 'clear') return items.length === 0 ? items : [];
  // Array.prototype.includes uses SameValueZero, exactly like Set.
  const present = items.includes(a.v);
  const without = () => items.filter((x) => !(x === a.v || (x !== x && a.v !== a.v)));
  if (a.op === 'add') return present ? items : [...items, a.v];
  if (a.op === 'remove') return present ? without() : items;
  return present ? without() : [...items, a.v];
}

export function useSet<T>(initial?: Iterable<T>) {
  const [items, dispatch] = useReducer(reduce<T>, initial, (init) => [...new Set(init)]);
  const values: ReadonlySet<T> = useMemo(() => new Set(items), [items]);

  return {
    values,
    add: (v: T) => dispatch({ op: 'add', v }),
    remove: (v: T) => dispatch({ op: 'remove', v }),
    toggle: (v: T) => dispatch({ op: 'toggle', v }),
    has: (v: T) => values.has(v),
    clear: () => dispatch({ op: 'clear' }),
  };
}
