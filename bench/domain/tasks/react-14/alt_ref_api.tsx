import { useLayoutEffect, useRef, useState } from 'react';

type Opts = { initial?: number; min?: number; max?: number; step?: number };

// The options live in a ref and the API object is created once with
// useState, instead of useCallback dependency lists.
export function useCounter(options?: Opts) {
  const cfg = {
    initial: options?.initial ?? 0,
    min: options?.min ?? -Infinity,
    max: options?.max ?? Infinity,
    step: options?.step ?? 1,
  };
  const cfgRef = useRef(cfg);
  useLayoutEffect(() => {
    cfgRef.current = cfg;
  });

  const [count, setCount] = useState(() => Math.min(cfg.max, Math.max(cfg.min, cfg.initial)));

  const [api] = useState(() => {
    const clamp = (n: number) => {
      const { min, max } = cfgRef.current;
      return n < min ? min : n > max ? max : n;
    };
    return {
      increment() {
        setCount((c) => clamp(c + cfgRef.current.step));
      },
      decrement() {
        setCount((c) => clamp(c - cfgRef.current.step));
      },
      set(n: number) {
        setCount(clamp(n));
      },
      reset() {
        setCount(clamp(cfgRef.current.initial));
      },
    };
  });

  return { count, ...api };
}
