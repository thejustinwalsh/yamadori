import { useEffect, useRef } from 'react';

// Wrong (strict): passes the typed handler straight to addEventListener on an
// EventTarget, whose listener must accept any Event, and `target ?? window`
// still admits null.
export function useEventListener<K extends keyof WindowEventMap>(
  type: K,
  handler: (event: WindowEventMap[K]) => void,
  target?: EventTarget | null,
): void {
  const latest = useRef(handler);
  latest.current = handler;

  useEffect(() => {
    const el: EventTarget = target === undefined ? window : target;
    const listener = (event: WindowEventMap[K]) => latest.current(event);
    el.addEventListener(type, listener);
    return () => el.removeEventListener(type, listener);
  }, [type, target]);
}
