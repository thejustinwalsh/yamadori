import { createContext, use, useReducer, useState } from 'react';
import type { Dispatch, ReactNode } from 'react';

type Action = 'inc' | 'dec' | 'reset';

function reducer(n: number, a: Action): number {
  if (a === 'inc') return n + 1;
  if (a === 'dec') return n - 1;
  return 0;
}

const MISSING = Symbol('missing');

type Actions = { increment(): void; decrement(): void; reset(): void };

const CountCtx = createContext<number | typeof MISSING>(MISSING);
const ActionsCtx = createContext<Actions | typeof MISSING>(MISSING);

function makeActions(dispatch: Dispatch<Action>): Actions {
  return {
    increment() {
      dispatch('inc');
    },
    decrement() {
      dispatch('dec');
    },
    reset() {
      dispatch('reset');
    },
  };
}

// dispatch is stable, so the actions object is built exactly once in lazy
// state and never changes; React 19's `use` reads the contexts.
export function CounterProvider(props: { children: ReactNode }) {
  const [count, dispatch] = useReducer(reducer, 0);
  const [actions] = useState(() => makeActions(dispatch));
  return (
    <CountCtx.Provider value={count}>
      <ActionsCtx.Provider value={actions}>{props.children}</ActionsCtx.Provider>
    </CountCtx.Provider>
  );
}

export function useCount(): number {
  const v = use(CountCtx);
  if (v === MISSING) throw new Error('no CounterProvider above useCount');
  return v;
}

export function useCountActions(): { increment(): void; decrement(): void; reset(): void } {
  const v = use(ActionsCtx);
  if (v === MISSING) throw new Error('no CounterProvider above useCountActions');
  return v;
}
