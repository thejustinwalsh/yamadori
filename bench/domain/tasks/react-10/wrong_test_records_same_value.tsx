// Wrong: set() never compares with the current state, so setting the same
// value adds a history entry and wipes the redo stack.
import { useReducer } from 'react';

type H<T> = { past: T[]; present: T; future: T[] };
type A<T> = { k: 'set' | 'reset'; v: T } | { k: 'undo' | 'redo' };

function step<T>(h: H<T>, a: A<T>): H<T> {
  if (a.k === 'set') return { past: [...h.past, h.present], present: a.v, future: [] };
  if (a.k === 'reset') return { past: [], present: a.v, future: [] };
  if (a.k === 'undo') {
    if (!h.past.length) return h;
    return { past: h.past.slice(0, -1), present: h.past[h.past.length - 1], future: [h.present, ...h.future] };
  }
  if (!h.future.length) return h;
  return { past: [...h.past, h.present], present: h.future[0], future: h.future.slice(1) };
}

export function useUndoRedo<T>(initial: T) {
  const [h, dispatch] = useReducer(step<T>, { past: [], present: initial, future: [] });
  return {
    state: h.present,
    set: (v: T) => dispatch({ k: 'set', v }),
    undo: () => dispatch({ k: 'undo' }),
    redo: () => dispatch({ k: 'redo' }),
    canUndo: h.past.length > 0,
    canRedo: h.future.length > 0,
    reset: (v: T) => dispatch({ k: 'reset', v }),
  };
}
