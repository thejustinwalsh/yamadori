import { createContext, useContext, useEffect, useReducer, useRef } from 'react';
import type { ReactNode } from 'react';

// Each toast component owns its own timer through an effect, so removing the
// toast (by any path) unmounts it and clears the timer. State is a reducer.
type Toast = { id: number; message: string };
type Action = { type: 'add'; toast: Toast } | { type: 'remove'; id: number };

function reducer(state: Toast[], action: Action): Toast[] {
  if (action.type === 'add') return [...state, action.toast].slice(-3);
  const next = state.filter((t) => t.id !== action.id);
  return next.length === state.length ? state : next;
}

interface Api {
  show(message: string): number;
  dismiss(id: number): void;
}

const Ctx = createContext<Api | undefined>(undefined);

function ToastItem({ toast, ms, onClose }: { toast: Toast; ms: number; onClose: (id: number) => void }) {
  useEffect(() => {
    const handle = window.setTimeout(() => onClose(toast.id), ms);
    return () => window.clearTimeout(handle);
    // the deadline is fixed when the toast first appears
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toast.id]);
  return (
    <li role="status">
      {toast.message}{' '}
      <button aria-label="Dismiss" onClick={() => onClose(toast.id)}>
        ×
      </button>
    </li>
  );
}

export function ToastProvider(props: { children: ReactNode; duration?: number }) {
  const [toasts, dispatch] = useReducer(reducer, []);
  const counter = useRef(0);
  const apiRef = useRef<Api | null>(null);
  if (apiRef.current === null) {
    apiRef.current = {
      show(message) {
        counter.current += 1;
        dispatch({ type: 'add', toast: { id: counter.current, message } });
        return counter.current;
      },
      dismiss(id) {
        dispatch({ type: 'remove', id });
      },
    };
  }
  const api = apiRef.current;
  return (
    <Ctx.Provider value={api}>
      {props.children}
      <ul aria-label="Notifications">
        {toasts.map((t) => (
          <ToastItem key={t.id} toast={t} ms={props.duration ?? 3000} onClose={api.dismiss} />
        ))}
      </ul>
    </Ctx.Provider>
  );
}

export function useToast() {
  const api = useContext(Ctx);
  if (api === undefined) throw new Error('useToast outside ToastProvider');
  return api;
}
