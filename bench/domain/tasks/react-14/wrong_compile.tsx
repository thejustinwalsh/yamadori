import { useCallback, useState } from 'react';

// Wrong (strict): min and max are possibly undefined when passed to Math.
type Opts = { initial?: number; min?: number; max?: number; step?: number };

export function useCounter(options?: Opts) {
  const { initial = 0, min, max, step = 1 } = options ?? {};
  const clamp = useCallback((n: number) => Math.max(min, Math.min(max, n)), [min, max]);
  const [count, setCount] = useState(() => clamp(initial));

  const increment = useCallback(() => setCount((c) => clamp(c + step)), [clamp, step]);
  const decrement = useCallback(() => setCount((c) => clamp(c - step)), [clamp, step]);
  const set = useCallback((n: number) => setCount(clamp(n)), [clamp]);
  const reset = useCallback(() => setCount(clamp(initial)), [clamp, initial]);

  return { count, increment, decrement, set, reset };
}
