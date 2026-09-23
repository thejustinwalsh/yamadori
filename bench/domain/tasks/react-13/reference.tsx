import { useEffect, useState } from 'react';

const EVENTS = ['mousemove', 'mousedown', 'keydown', 'touchstart', 'scroll', 'wheel'] as const;

export function useIdle(timeoutMs: number): boolean {
  const [idle, setIdle] = useState(false);

  useEffect(() => {
    let id: ReturnType<typeof setTimeout> | undefined;
    const arm = () => {
      clearTimeout(id);
      id = setTimeout(() => setIdle(true), timeoutMs);
    };
    const onActivity = () => {
      setIdle(false);
      arm();
    };
    arm();
    for (const e of EVENTS) window.addEventListener(e, onActivity, { passive: true });
    return () => {
      clearTimeout(id);
      for (const e of EVENTS) window.removeEventListener(e, onActivity);
    };
  }, [timeoutMs]);

  return idle;
}
