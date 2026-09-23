import { useEffect, useState } from 'react';

// Wrong (strict): the event names are plain strings, so addEventListener
// expects a listener taking any Event, not a MouseEvent | KeyboardEvent one.
const EVENTS: string[] = ['mousemove', 'mousedown', 'keydown', 'touchstart', 'scroll', 'wheel'];

export function useIdle(timeoutMs: number): boolean {
  const [idle, setIdle] = useState(false);

  useEffect(() => {
    let id = setTimeout(() => setIdle(true), timeoutMs);
    const onActivity = (event: MouseEvent | KeyboardEvent) => {
      if (event.type) setIdle(false);
      clearTimeout(id);
      id = setTimeout(() => setIdle(true), timeoutMs);
    };
    EVENTS.forEach((e) => window.addEventListener(e, onActivity));
    return () => {
      clearTimeout(id);
      EVENTS.forEach((e) => window.removeEventListener(e, onActivity));
    };
  }, [timeoutMs]);

  return idle;
}
