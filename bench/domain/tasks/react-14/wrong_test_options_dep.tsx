// Wrong: memoises on the options object itself, which is new on every render,
// so the functions change identity every time and memoised children re-render.
import { useCallback, useState } from 'react';

type Opts = { initial?: number; min?: number; max?: number; step?: number };

export function useCounter(options?: Opts) {
  const clamp = useCallback(
    (n: number) => Math.min(options?.max ?? Infinity, Math.max(options?.min ?? -Infinity, n)),
    [options],
  );
  const [count, setCount] = useState(() => clamp(options?.initial ?? 0));
  const step = options?.step ?? 1;

  const increment = useCallback(() => setCount((c) => clamp(c + step)), [clamp, step]);
  const decrement = useCallback(() => setCount((c) => clamp(c - step)), [clamp, step]);
  const set = useCallback((n: number) => setCount(clamp(n)), [clamp]);
  const reset = useCallback(() => setCount(clamp(options?.initial ?? 0)), [clamp, options]);

  return { count, increment, decrement, set, reset };
}
