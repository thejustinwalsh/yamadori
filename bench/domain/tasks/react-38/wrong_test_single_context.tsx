// Wrong: one context carries both the count and the actions, so every
// actions-only consumer re-renders on each change.
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react';

interface CounterValue {
  count: number;
  increment(): void;
  decrement(): void;
  reset(): void;
}

const CounterContext = createContext<CounterValue | null>(null);

export function CounterProvider({ children }: { children: ReactNode }) {
  const [count, setCount] = useState(0);
  const increment = useCallback(() => setCount((c) => c + 1), []);
  const decrement = useCallback(() => setCount((c) => c - 1), []);
  const reset = useCallback(() => setCount(0), []);
  const value = useMemo(
    () => ({ count, increment, decrement, reset }),
    [count, increment, decrement, reset],
  );
  return <CounterContext value={value}>{children}</CounterContext>;
}

function useCounter(): CounterValue {
  const v = useContext(CounterContext);
  if (!v) throw new Error('CounterProvider missing');
  return v;
}

export function useCount(): number {
  return useCounter().count;
}

export function useCountActions(): { increment(): void; decrement(): void; reset(): void } {
  const { increment, decrement, reset } = useCounter();
  return useMemo(() => ({ increment, decrement, reset }), [increment, decrement, reset]);
}
