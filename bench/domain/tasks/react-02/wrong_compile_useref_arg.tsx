import { useEffect, useRef, useState } from 'react';

// Wrong (strict, React 19 types): useRef requires an initial value.
export function useThrottle<T>(value: T, intervalMs: number): T {
  const [throttled, setThrottled] = useState<T>(value);
  const lastEmit = useRef<number>(Date.now());
  const timer = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => {
    const wait = lastEmit.current + intervalMs - Date.now();
    if (wait <= 0) {
      lastEmit.current = Date.now();
      setThrottled(() => value);
      return;
    }
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      lastEmit.current = Date.now();
      setThrottled(() => value);
    }, wait);
    return () => clearTimeout(timer.current);
  }, [value, intervalMs]);
  return throttled;
}
