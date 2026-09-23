import { useEffect, useRef, useState } from 'react';

// Wrong: leading edge only -- a change inside the window is dropped, so the
// final value can be lost for good.
export function useThrottle<T>(value: T, intervalMs: number): T {
  const [throttled, setThrottled] = useState<T>(value);
  const lastEmit = useRef(Date.now());
  useEffect(() => {
    const now = Date.now();
    if (now - lastEmit.current >= intervalMs) {
      lastEmit.current = now;
      setThrottled(() => value);
    }
  }, [value, intervalMs]);
  return throttled;
}
