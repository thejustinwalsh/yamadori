import { useEffect, useRef, useState } from 'react';

// Wrong: the trailing timer captures the value that scheduled it, so it emits
// a superseded value instead of the latest one.
export function useThrottle<T>(value: T, intervalMs: number): T {
  const [throttled, setThrottled] = useState<T>(value);
  const last = useRef(Date.now());
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mounted = useRef(false);
  useEffect(() => {
    if (!mounted.current) {
      mounted.current = true;
      return;
    }
    const now = Date.now();
    if (now - last.current >= intervalMs) {
      last.current = now;
      setThrottled(() => value);
      return;
    }
    if (timer.current !== null) return;
    timer.current = setTimeout(() => {
      timer.current = null;
      last.current = Date.now();
      setThrottled(() => value);
    }, last.current + intervalMs - now);
  }, [value, intervalMs]);
  useEffect(
    () => () => {
      if (timer.current !== null) clearTimeout(timer.current);
    },
    [],
  );
  return throttled;
}
