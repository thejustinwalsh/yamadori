import { useEffect, useRef, type RefObject } from 'react';

// Wrong (strict): ref.current may be null, and event.target is an
// EventTarget | null, not a Node.
export function useClickOutside<T extends HTMLElement>(
  ref: RefObject<T | null>,
  handler: (event: MouseEvent | TouchEvent) => void,
): void {
  const latest = useRef(handler);
  latest.current = handler;

  useEffect(() => {
    const listener = (event: MouseEvent | TouchEvent) => {
      if (ref.current.contains(event.target)) return;
      latest.current(event);
    };
    document.addEventListener('mousedown', listener);
    document.addEventListener('touchstart', listener);
    return () => {
      document.removeEventListener('mousedown', listener);
      document.removeEventListener('touchstart', listener);
    };
  }, [ref]);
}
