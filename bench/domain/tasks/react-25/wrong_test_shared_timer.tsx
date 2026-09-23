// Wrong: one shared timer, restarted by every show(), clears all toasts at once instead of giving each toast its own deadline.
import { createContext, useContext, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';

type Toast = { id: number; message: string };
type ToastApi = { show: (message: string) => number; dismiss: (id: number) => void };

const ToastContext = createContext<ToastApi | null>(null);

export function ToastProvider({ children, duration = 3000 }: { children: ReactNode; duration?: number }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const stop = () => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
  };

  useEffect(() => stop, []);

  const dismiss = (id: number) => {
    setToasts((prev) => {
      const next = prev.filter((t) => t.id !== id);
      if (next.length === 0) stop();
      return next;
    });
  };

  const show = (message: string) => {
    const id = nextId.current++;
    setToasts((prev) => [...prev, { id, message }].slice(-3));
    stop();
    timer.current = setTimeout(() => {
      timer.current = null;
      setToasts([]);
    }, duration);
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
