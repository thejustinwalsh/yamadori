// Wrong: the expiry timer filters the `toasts` captured when show() ran, so when one toast expires it resets the list to that stale snapshot.
import { createContext, useContext, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';

type Toast = { id: number; message: string };
type ToastApi = { show: (message: string) => number; dismiss: (id: number) => void };

const ToastContext = createContext<ToastApi | null>(null);

export function ToastProvider({ children, duration = 3000 }: { children: ReactNode; duration?: number }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);
  const timers = useRef(new Map<number, ReturnType<typeof setTimeout>>());

  useEffect(() => {
    const all = timers.current;
    return () => all.forEach((t) => clearTimeout(t));
  }, []);

  const dismiss = (id: number) => {
    clearTimeout(timers.current.get(id));
    timers.current.delete(id);
    setToasts(toasts.filter((t) => t.id !== id));
  };

  const show = (message: string) => {
    const id = nextId.current++;
    let next = [...toasts, { id, message }];
    if (next.length > 3) {
      const [oldest, ...rest] = next;
      clearTimeout(timers.current.get(oldest.id));
      timers.current.delete(oldest.id);
      next = rest;
    }
    setToasts(next);
    timers.current.set(
      id,
      setTimeout(() => {
        timers.current.delete(id);
        setToasts(toasts.filter((t) => t.id !== id));
      }, duration),
    );
    return id;
  };

  return (
    <ToastContext.Provider value={{ show, dismiss }}>
      {children}
      {toasts.map((t) => (
        <div key={t.id} role="status">
          {t.message}
          <button onClick={() => dismiss(t.id)}>Dismiss</button>
        </div>
      ))}
    </ToastContext.Provider>
  );
}

export function useToast(): ToastApi {
  const api = useContext(ToastContext);
  if (!api) throw new Error('useToast must be used inside a ToastProvider');
  return api;
}
