import { useEffect, useId, useReducer, useRef } from 'react';
import type * as React from 'react';

// Every panel stays mounted (unselected ones carry `hidden`); the selection is
// an index in a reducer, and focus follows it in an effect after keyboard moves.
type Action = { type: 'set'; index: number; viaKey: boolean };
type State = { index: number; viaKey: boolean };

export function Tabs(props: {
  tabs: { id: string; label: string; content: React.ReactNode }[];
  defaultTabId?: string;
}): React.ReactElement {
  const { tabs, defaultTabId } = props;
  const prefix = useId();
  const count = tabs.length;
  const [state, dispatch] = useReducer(
    (s: State, a: Action): State => ({ index: ((a.index % count) + count) % count, viaKey: a.viaKey }),
    undefined,
    (): State => {
      const i = tabs.findIndex((t) => t.id === defaultTabId);
      return { index: i < 0 ? 0 : i, viaKey: false };
    },
  );
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!state.viaKey) return;
    const el = listRef.current?.querySelectorAll<HTMLElement>('[role="tab"]')[state.index];
    el?.focus();
  }, [state]);

  const keys: Record<string, (i: number) => number> = {
    ArrowRight: (i) => i + 1,
    ArrowLeft: (i) => i - 1,
    Home: () => 0,
    End: () => count - 1,
  };

  return (
    <section>
      <div role="tablist" ref={listRef}>
        {tabs.map((t, i) => (
          <button
            key={t.id}
            role="tab"
            id={`${prefix}t${i}`}
            aria-controls={`${prefix}p${i}`}
            aria-selected={state.index === i ? 'true' : 'false'}
            tabIndex={state.index === i ? 0 : -1}
            onClick={() => dispatch({ type: 'set', index: i, viaKey: false })}
            onKeyDown={(e) => {
              const f = keys[e.key];
              if (!f) return;
              e.preventDefault();
              dispatch({ type: 'set', index: f(state.index), viaKey: true });
            }}
          >
            {t.label}
          </button>
        ))}
      </div>
      {tabs.map((t, i) => (
        <div
          key={t.id}
          role="tabpanel"
          id={`${prefix}p${i}`}
          aria-labelledby={`${prefix}t${i}`}
          hidden={state.index !== i}
        >
          {t.content}
        </div>
      ))}
    </section>
  );
}
