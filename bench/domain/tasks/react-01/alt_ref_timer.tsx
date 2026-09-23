import { useEffect, useRef, useState } from 'react';

// Keeps the timer handle in a ref and boxes the value, so a function-typed
// T is never mistaken for a state updater.
export function useDebounce<T>(value: T, delayMs: number): T {
  const [box, setBox] = useState(() => ({ current: value }));
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      timer.current = null;
      setBox({ current: value });
    }, delayMs);
  }, [value, delayMs]);

  useEffect(
    () => () => {
      if (timer.current !== null) clearTimeout(timer.current);
    },
    [],
  );

  return box.current;
}
