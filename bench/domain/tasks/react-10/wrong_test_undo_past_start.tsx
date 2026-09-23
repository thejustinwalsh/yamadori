// Wrong: undo() has no guard at the start of history, so it moves to an
// undefined state and pushes the current one onto the redo stack.
import { useState } from 'react';

export function useUndoRedo<T>(initial: T) {
  const [h, setH] = useState({ past: [] as T[], present: initial, future: [] as T[] });

  return {
    state: h.present,
    set: (next: T) =>
      setH((s) =>
        Object.is(next, s.present) ? s : { past: [...s.past, s.present], present: next, future: [] },
      ),
    undo: () =>
      setH((s) => ({
        past: s.past.slice(0, -1),
        present: s.past[s.past.length - 1] as T,
        future: [s.present, ...s.future],
      })),
    redo: () =>
      setH((s) =>
        s.future.length === 0
          ? s
          : { past: [...s.past, s.present], present: s.future[0], future: s.future.slice(1) },
      ),
    canUndo: h.past.length > 0,
    canRedo: h.future.length > 0,
    reset: (next: T) => setH({ past: [], present: next, future: [] }),
  };
}
