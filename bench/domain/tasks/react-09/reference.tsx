import { useEffect, useLayoutEffect, useRef } from 'react';

export function useEventListener<K extends keyof WindowEventMap>(
  type: K,
  handler: (event: WindowEventMap[K]) => void,
  target?: EventTarget | null,
): void {
  const latest = useRef(handler);
  useLayoutEffect(() => {
    latest.current = handler;
  });

  useEffect(() => {
    const el: EventTarget | null = target === undefined ? window : target;
    if (el === null) return;
    const listener = (event: Event) => latest.current(event as WindowEventMap[K]);
    el.addEventListener(type, listener);
    return () => el.removeEventListener(type, listener);
  }, [type, target]);
}
