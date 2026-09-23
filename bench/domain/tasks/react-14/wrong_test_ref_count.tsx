// Wrong: reads the count from a ref refreshed on render instead of using a
// functional update, so two increments in one handler add only one step.
import { useCallback, useRef, useState } from 'react';

type Opts = { initial?: number; min?: number; max?: number; step?: number };

export function useCounter(options: Opts = {}) {
  const { initial = 0, min = -Infinity, max = Infinity, step = 1 } = options;
  const clamp = (n: number) => Math.min(max, Math.max(min, n));
  const [count, setCount] = useState(() => clamp(initial));
  const latest = useRef(count);
  latest.current = count;
  const cfg = useRef({ min, max, step, initial });
  cfg.current = { min, max, step, initial };

  const fit = (n: number) => Math.min(cfg.current.max, Math.max(cfg.current.min, n));
  const increment = useCallback(() => setCount(fit(latest.current + cfg.current.step)), []);
  const decrement = useCallback(() => setCount(fit(latest.current - cfg.current.step)), []);
  const set = useCallback((n: number) => setCount(fit(n)), []);
  const reset = useCallback(() => setCount(fit(cfg.current.initial)), []);

  return { count, increment, decrement, set, reset };
}
