import { useCallback, useState } from 'react';

export function useCounter(options?: {
  initial?: number;
  min?: number;
  max?: number;
  step?: number;
}): {
  count: number;
  increment: () => void;
  decrement: () => void;
  set: (n: number) => void;
  reset: () => void;
} {
  const initial = options?.initial ?? 0;
  const min = options?.min ?? -Infinity;
  const max = options?.max ?? Infinity;
  const step = options?.step ?? 1;

  const clamp = useCallback((n: number) => Math.min(max, Math.max(min, n)), [min, max]);
  const [count, setCount] = useState(() => clamp(initial));

  const increment = useCallback(() => setCount((c) => clamp(c + step)), [clamp, step]);
  const decrement = useCallback(() => setCount((c) => clamp(c - step)), [clamp, step]);
  const set = useCallback((n: number) => setCount(clamp(n)), [clamp]);
  const reset = useCallback(() => setCount(clamp(initial)), [clamp, initial]);

  return { count, increment, decrement, set, reset };
}
