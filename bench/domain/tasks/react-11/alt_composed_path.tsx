import { useEffect, useEffectEvent, type RefObject } from 'react';

// Tests containment with composedPath() instead of Node.contains, and routes
// the handler through useEffectEvent; one listener object serves both types.
export function useClickOutside<T extends HTMLElement>(
  ref: RefObject<T | null>,
  handler: (event: MouseEvent | TouchEvent) => void,
): void {
  const onPress = useEffectEvent((event: MouseEvent | TouchEvent) => handler(event));

  useEffect(() => {
    const types = ['mousedown', 'touchstart'] as const;
    const listener: EventListenerObject = {
      handleEvent(event: Event) {
        const el = ref.current;
        if (!el || event.composedPath().includes(el)) return;
        onPress(event as MouseEvent | TouchEvent);
      },
    };
    for (const t of types) document.addEventListener(t, listener, true);
    return () => {
      for (const t of types) document.removeEventListener(t, listener, true);
    };
  }, [ref]);
}
