// Wrong: listens for `click` instead of mousedown/touchstart.
import { useEffect, useRef, type RefObject } from 'react';

export function useClickOutside<T extends HTMLElement>(
  ref: RefObject<T | null>,
  handler: (event: MouseEvent | TouchEvent) => void,
): void {
  const latest = useRef(handler);
  latest.current = handler;

  useEffect(() => {
    const listener = (event: MouseEvent) => {
      const el = ref.current;
      if (!el || el.contains(event.target as Node)) return;
      latest.current(event);
    };
    document.addEventListener('click', listener);
    return () => document.removeEventListener('click', listener);
  }, [ref]);
}
