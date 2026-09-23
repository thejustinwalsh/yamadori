import { useEffect, useLayoutEffect, useRef } from 'react';

export function useInterval(callback: () => void, delayMs: number | null): void {
  const saved = useRef(callback);

  useLayoutEffect(() => {
    saved.current = callback;
  });

  useEffect(() => {
    if (delayMs === null) return;
    const id = setInterval(() => saved.current(), delayMs);
    return () => clearInterval(id);
  }, [delayMs]);
}
