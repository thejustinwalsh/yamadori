import { useEffect, useState } from 'react';

// Wrong: this is a debounce -- every change restarts the wait, so a stream of
// changes never emits until it goes quiet.
export function useThrottle<T>(value: T, intervalMs: number): T {
  const [throttled, setThrottled] = useState<T>(value);
  useEffect(() => {
    const id = setTimeout(() => setThrottled(() => value), intervalMs);
    return () => clearTimeout(id);
  }, [value, intervalMs]);
  return throttled;
}
