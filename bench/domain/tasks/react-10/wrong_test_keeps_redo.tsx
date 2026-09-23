// Wrong: set() pushes onto the undo stack but leaves the redo stack intact.
import { useState } from 'react';

export function useUndoRedo<T>(initial: T) {
  const [past, setPast] = useState<T[]>([]);
  const [present, setPresent] = useState<T>(initial);
  const [future, setFuture] = useState<T[]>([]);

  return {
    state: present,
    set: (next: T) => {
      if (Object.is(next, present)) return;
      setPast([...past, present]);
      setPresent(next);
    },
    undo: () => {
      if (past.length === 0) return;
      setFuture([present, ...future]);
      setPresent(past[past.length - 1]);
      setPast(past.slice(0, -1));
    },
    redo: () => {
      if (future.length === 0) return;
      setPast([...past, present]);
      setPresent(future[0]);
      setFuture(future.slice(1));
    },
    canUndo: past.length > 0,
    canRedo: future.length > 0,
    reset: (next: T) => {
      setPast([]);
      setPresent(next);
      setFuture([]);
    },
  };
}
