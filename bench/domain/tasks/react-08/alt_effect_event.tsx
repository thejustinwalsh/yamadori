import { useEffect, useEffectEvent } from 'react';

// React 19.2's useEffectEvent gives the timer a stable function that always
// runs the latest callback; the interval is a chain of timeouts.
export function useInterval(callback: () => void, delayMs: number | null): void {
  const tick = useEffectEvent(() => callback());

  useEffect(() => {
    if (delayMs === null) return;
    let id: ReturnType<typeof setTimeout>;
    const schedule = () => {
      id = setTimeout(() => {
        schedule();
        tick();
      }, delayMs);
    };
    schedule();
    return () => clearTimeout(id);
  }, [delayMs]);
}
