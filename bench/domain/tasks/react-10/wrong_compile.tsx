import { useState } from 'react';

// Wrong (strict): Array.prototype.pop() returns T | undefined, which is
// passed where T is required.
export function useUndoRedo<T>(initial: T) {
  const [past, setPast] = useState<T[]>([]);
  const [present, setPresent] = useState<T>(initial);
  const [future, setFuture] = useState<T[]>([]);

  const undo = () => {
    if (past.length === 0) return;
    const rest = [...past];
    const prev = rest.pop();
    setPast(rest);
    setFuture([present, ...future]);
    setPresent(prev);
  };
  const redo = () => {
    if (future.length === 0) return;
    const [next, ...rest] = future;
    setPast([...past, present]);
    setFuture(rest);
    setPresent(next);
  };

  return {
    state: present,
    set: (next: T) => {
      if (Object.is(next, present)) return;
      setPast([...past, present]);
      setPresent(next);
      setFuture([]);
    },
    undo,
    redo,
    canUndo: past.length > 0,
    canRedo: future.length > 0,
    reset: (next: T) => {
      setPast([]);
      setPresent(next);
      setFuture([]);
    },
  };
}
