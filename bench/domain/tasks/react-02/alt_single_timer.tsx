import { useEffect, useReducer, useRef } from 'react';

// One trailing timer that is never rescheduled; it reads the latest value
// from a ref when it fires. State is only a render counter; the returned
// value lives in a ref.
export function useThrottle<T>(value: T, intervalMs: number): T {
  const [, bump] = useReducer((n: number) => n + 1, 0);
  const shown = useRef<{ v: T }>({ v: value });
  const latest = useRef<T>(value);
  const last = useRef<number>(Date.now());
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    latest.current = value;
    if (timer.current !== null) return; // the pending trailing update picks it up
    if (Object.is(value, shown.current.v)) return;
    const now = Date.now();
    if (now - last.current >= intervalMs) {
      last.current = now;
      shown.current = { v: value };
      bump();
      return;
    }
    timer.current = setTimeout(() => {
      timer.current = null;
      if (Object.is(shown.current.v, latest.current)) return;
      last.current = Date.now();
      shown.current = { v: latest.current };
      bump();
    }, last.current + intervalMs - now);
  }, [value, intervalMs]);

  useEffect(
    () => () => {
      if (timer.current !== null) clearTimeout(timer.current);
      timer.current = null;
    },
    [],
  );

  return shown.current.v;
}
