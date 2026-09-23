// Wrong: the contexts are split but the actions object is rebuilt on every provider render, so actions-only consumers still re-render.
import { createContext, useContext, useState, type ReactNode } from 'react';

export interface CountActions {
  increment(): void;
  decrement(): void;
  reset(): void;
}

const CountContext = createContext<number | null>(null);
const ActionsContext = createContext<CountActions | null>(null);

export function CounterProvider({ children }: { children: ReactNode }) {
  const [count, setCount] = useState(0);
  const actions: CountActions = {
    increment: () => setCount((c) => c + 1),
    decrement: () => setCount((c) => c - 1),
    reset: () => setCount(0),
  };
  return (
    <ActionsContext value={actions}>
      <CountContext value={count}>{children}</CountContext>
    </ActionsContext>
  );
}

export function useCount(): number {
  const count = useContext(CountContext);
  if (count === null) throw new Error('useCount must be used inside a CounterProvider');
  return count;
}

export function useCountActions(): CountActions {
  const actions = useContext(ActionsContext);
  if (actions === null) throw new Error('useCountActions must be used inside a CounterProvider');
  return actions;
}
