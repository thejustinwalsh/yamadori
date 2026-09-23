// Wrong: set() stores the value as given, without clamping it to [min, max].
import { useCallback, useState } from 'react';

type Opts = { initial?: number; min?: number; max?: number; step?: number };

export function useCounter(options?: Opts) {
  const initial = options?.initial ?? 0;
  const min = options?.min ?? -Infinity;
  const max = options?.max ?? Infinity;
  const step = options?.step ?? 1;
  const clamp = useCallback((n: number) => Math.min(max, Math.max(min, n)), [min, max]);
  const [count, setCount] = useState(() => clamp(initial));

  return {
    count,
    increment: useCallback(() => setCount((c) => clamp(c + step)), [clamp, step]),
    decrement: useCallback(() => setCount((c) => clamp(c - step)), [clamp, step]),
    set: useCallback((n: number) => setCount(n), []),
    reset: useCallback(() => setCount(clamp(initial)), [clamp, initial]),
  };
}
