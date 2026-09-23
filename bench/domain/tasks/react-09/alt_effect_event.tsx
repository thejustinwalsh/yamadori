import { useEffect, useEffectEvent } from 'react';

// Uses React 19.2's useEffectEvent and an EventListenerObject instead of a
// ref-held closure.
export function useEventListener<K extends keyof WindowEventMap>(
  type: K,
  handler: (event: WindowEventMap[K]) => void,
  target?: EventTarget | null,
): void {
  const onEvent = useEffectEvent((event: Event) => {
    handler(event as WindowEventMap[K]);
  });

  useEffect(() => {
    if (target === null) return undefined;
    const el = target ?? window;
    const listener: EventListenerObject = { handleEvent: (e) => onEvent(e) };
    el.addEventListener(type, listener);
    return () => {
      el.removeEventListener(type, listener);
    };
  }, [type, target]);
}
