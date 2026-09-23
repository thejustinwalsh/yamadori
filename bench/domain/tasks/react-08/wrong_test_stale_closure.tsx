import { useEffect } from 'react';

// Wrong: stale closure -- the interval keeps calling the callback from the
// render that started it.
export function useInterval(callback: () => void, delayMs: number | null): void {
  useEffect(() => {
    if (delayMs === null) return;
    const id = setInterval(() => callback(), delayMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [delayMs]);
}
