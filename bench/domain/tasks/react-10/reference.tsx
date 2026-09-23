import { useCallback, useReducer } from 'react';

interface History<T> {
  past: T[];
  present: T;
  future: T[];
}

type Action<T> =
  | { type: 'set'; next: T }
  | { type: 'undo' }
  | { type: 'redo' }
  | { type: 'reset'; next: T };

function reducer<T>(h: History<T>, a: Action<T>): History<T> {
  switch (a.type) {
    case 'set':
      if (Object.is(a.next, h.present)) return h;
      return { past: [...h.past, h.present], present: a.next, future: [] };
    case 'undo': {
      if (h.past.length === 0) return h;
      const prev = h.past[h.past.length - 1];
      return { past: h.past.slice(0, -1), present: prev, future: [h.present, ...h.future] };
    }
    case 'redo': {
      if (h.future.length === 0) return h;
      const [next, ...rest] = h.future;
      return { past: [...h.past, h.present], present: next, future: rest };
    }
    case 'reset':
      return { past: [], present: a.next, future: [] };
  }
}

export function useUndoRedo<T>(initial: T) {
  const [h, dispatch] = useReducer(reducer<T>, { past: [], present: initial, future: [] });
  const set = useCallback((next: T) => dispatch({ type: 'set', next }), []);
  const undo = useCallback(() => dispatch({ type: 'undo' }), []);
  const redo = useCallback(() => dispatch({ type: 'redo' }), []);
  const reset = useCallback((next: T) => dispatch({ type: 'reset', next }), []);
  return {
    state: h.present,
    set,
    undo,
    redo,
    canUndo: h.past.length > 0,
    canRedo: h.future.length > 0,
    reset,
  };
}
