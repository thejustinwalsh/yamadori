import { useEffect, useRef, useState } from 'react';

export function useThrottle<T>(value: T, intervalMs: number): T {
  const [throttled, setThrottled] = useState<T>(value);
  const lastEmit = useRef(Date.now());
  const emitted = useRef<T>(value);

  useEffect(() => {
    if (Object.is(value, emitted.current)) return;
    const emit = () => {
      lastEmit.current = Date.now();
      emitted.current = value;
      setThrottled(() => value);
    };
    const wait = lastEmit.current + intervalMs - Date.now();
    if (wait <= 0) {
      emit();
      return;
    }
    // A newer value clears this timer and schedules its own for the same
    // moment, so the trailing update always carries the latest value.
    const id = setTimeout(emit, wait);
    return () => clearTimeout(id);
  }, [value, intervalMs]);

  return throttled;
}
