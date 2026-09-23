// Wrong: increments from the rendered `count` instead of a functional update,
// so two calls in one handler add one step (and the callbacks change with count).
import { useCallback, useState } from 'react';

type Opts = { initial?: number; min?: number; max?: number; step?: number };

export function useCounter(options?: Opts) {
  const initial = options?.initial ?? 0;
  const min = options?.min ?? -Infinity;
  const max = options?.max ?? Infinity;
  const step = options?.step ?? 1;
  const clamp = (n: number) => Math.min(max, Math.max(min, n));
  const [count, setCount] = useState(clamp(initial));

  const increment = useCallback(() => setCount(clamp(count + step)), [count, step, min, max]);
  const decrement = useCallback(() => setCount(clamp(count - step)), [count, step, min, max]);
  const set = useCallback((n: number) => setCount(clamp(n)), [min, max]);
  const reset = useCallback(() => setCount(clamp(initial)), [initial, min, max]);

  return { count, increment, decrement, set, reset };
}
