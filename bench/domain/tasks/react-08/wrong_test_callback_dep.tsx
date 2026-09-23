import { useEffect } from 'react';

// Wrong: the callback is an effect dependency, so an inline callback restarts
// the interval on every render and a frequently re-rendering caller never
// sees a tick.
export function useInterval(callback: () => void, delayMs: number | null): void {
  useEffect(() => {
    if (delayMs === null) return;
    const id = setInterval(callback, delayMs);
    return () => clearInterval(id);
  }, [callback, delayMs]);
}
