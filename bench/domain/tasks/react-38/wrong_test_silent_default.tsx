// Wrong: the contexts default to 0 and no-op actions, so using the hooks outside a provider silently does nothing instead of throwing.
import { createContext, useContext, useMemo, useState, type ReactNode } from 'react';

export interface CountActions {
  increment(): void;
  decrement(): void;
  reset(): void;
}

const CountContext = createContext<number>(0);
const noop = () => {};
const ActionsContext = createContext<CountActions>({ increment: noop, decrement: noop, reset: noop });

export function CounterProvider({ children }: { children: ReactNode }) {
  const [count, setCount] = useState(0);
  const actions = useMemo<CountActions>(
    () => ({
      increment: () => setCount((c) => c + 1),
      decrement: () => setCount((c) => c - 1),
      reset: () => setCount(0),
    }),
    [],
  );
  return (
    <ActionsContext value={actions}>
      <CountContext value={count}>{children}</CountContext>
    </ActionsContext>
  );
}

export function useCount(): number {
  return useContext(CountContext);
}

export function useCountActions(): CountActions {
  return useContext(ActionsContext);
}
