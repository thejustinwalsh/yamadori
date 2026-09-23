import { useState } from 'react';

// One flat timeline plus a cursor, instead of past/present/future stacks.
export function useUndoRedo<T>(initial: T): {
  state: T;
  set: (next: T) => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  reset: (next: T) => void;
} {
  const [tl, setTl] = useState<{ items: T[]; at: number }>(() => ({ items: [initial], at: 0 }));
  const current = tl.items[tl.at];

  return {
    state: current,
    set(next) {
      setTl((t) =>
        Object.is(t.items[t.at], next)
          ? t
          : { items: [...t.items.slice(0, t.at + 1), next], at: t.at + 1 },
      );
    },
    undo() {
      setTl((t) => (t.at > 0 ? { items: t.items, at: t.at - 1 } : t));
    },
    redo() {
      setTl((t) => (t.at < t.items.length - 1 ? { items: t.items, at: t.at + 1 } : t));
    },
    canUndo: tl.at > 0,
    canRedo: tl.at < tl.items.length - 1,
    reset(next) {
      setTl({ items: [next], at: 0 });
    },
  };
}
