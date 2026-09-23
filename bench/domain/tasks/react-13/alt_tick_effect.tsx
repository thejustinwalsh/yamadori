import { useEffect, useReducer, useState } from 'react';

// Activity bumps a counter; a separate effect keyed on that counter owns the
// timer, so React's effect cleanup does the restarting.
export function useIdle(timeoutMs: number): boolean {
  const [tick, bump] = useReducer((n: number) => n + 1, 0);
  const [idle, setIdle] = useState(false);

  useEffect(() => {
    const onActivity = () => {
      setIdle(false);
      bump();
    };
    const types: Array<keyof WindowEventMap> = [
      'mousemove',
      'mousedown',
      'keydown',
      'touchstart',
      'scroll',
      'wheel',
    ];
    types.forEach((t) => window.addEventListener(t, onActivity));
    return () => types.forEach((t) => window.removeEventListener(t, onActivity));
  }, []);

  useEffect(() => {
    const id = setTimeout(() => setIdle(true), timeoutMs);
    return () => clearTimeout(id);
  }, [tick, timeoutMs]);

  return idle;
}
