// Wrong: handler is an effect dependency, so every new handler identity
// removes and re-adds the listener.
import { useEffect } from 'react';

export function useEventListener<K extends keyof WindowEventMap>(
  type: K,
  handler: (event: WindowEventMap[K]) => void,
  target?: EventTarget | null,
): void {
  useEffect(() => {
    const el: EventTarget | null = target === undefined ? window : target;
    if (el === null) return;
    const listener = (event: Event) => handler(event as WindowEventMap[K]);
    el.addEventListener(type, listener);
    return () => el.removeEventListener(type, listener);
  }, [type, target, handler]);
}
