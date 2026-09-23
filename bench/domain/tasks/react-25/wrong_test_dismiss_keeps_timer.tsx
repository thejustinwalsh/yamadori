// Wrong: dismissing (by button, by id, or by dropping the oldest) removes the toast but leaves its expiry timer running.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';

type Toast = { id: number; message: string };
type ToastApi = { show: (message: string) => number; dismiss: (id: number) => void };

const ToastContext = createContext<ToastApi | null>(null);

const MAX_VISIBLE = 3;

export function ToastProvider({ children, duration = 3000 }: { children: ReactNode; duration?: number }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const listRef = useRef<Toast[]>([]);
  const timers = useRef(new Map<number, ReturnType<typeof setTimeout>>());
  const nextId = useRef(1);
  const durationRef = useRef(duration);
  durationRef.current = duration;

  const commit = (next: Toast[]) => {
    listRef.current = next;
    setToasts(next);
  };

  const stopTimer = (id: number) => {
    const t = timers.current.get(id);
    if (t !== undefined) clearTimeout(t);
    timers.current.delete(id);
  };

  const dismiss = useCallback((id: number) => {

    if (listRef.current.some((t) => t.id === id)) {
      commit(listRef.current.filter((t) => t.id !== id));
    }
  }, []);

  const show = useCallback(
    (message: string) => {
      const id = nextId.current++;
      const next = [...listRef.current, { id, message }];
      while (next.length > MAX_VISIBLE) {
        const dropped = next.shift();
        void dropped;
      }
      timers.current.set(
        id,
        setTimeout(() => dismiss(id), durationRef.current),
      );
      commit(next);
      return id;
    },
    [dismiss],
  );

  useEffect(() => {
    const all = timers.current;
    return () => {
      for (const t of all.values()) clearTimeout(t);
      all.clear();
    };
  }, []);

  const api = useMemo(() => ({ show, dismiss }), [show, dismiss]);

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div>
        {toasts.map((t) => (
          <div key={t.id} role="status">
            <span>{t.message}</span>
            <button type="button" onClick={() => dismiss(t.id)}>
              Dismiss
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastApi {
  const api = useContext(ToastContext);
  if (!api) throw new Error('useToast must be used inside a ToastProvider');
  return api;
}
