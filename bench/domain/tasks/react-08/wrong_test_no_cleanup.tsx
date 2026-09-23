import { useEffect, useRef } from 'react';

// Wrong: the interval is never cleared, so a delay change stacks a second
// interval and the timer outlives the component.
export function useInterval(callback: () => void, delayMs: number | null): void {
  const saved = useRef(callback);
  useEffect(() => {
    saved.current = callback;
  });
  useEffect(() => {
    if (delayMs === null) return;
    setInterval(() => saved.current(), delayMs);
  }, [delayMs]);
}
