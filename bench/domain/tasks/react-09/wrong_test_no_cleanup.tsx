// Wrong: the effect never returns a cleanup, so the listener survives unmount
// and stays on the old target when the target changes.
import { useEffect, useRef } from 'react';

export function useEventListener<K extends keyof WindowEventMap>(
  type: K,
  handler: (event: WindowEventMap[K]) => void,
  target?: EventTarget | null,
): void {
  const latest = useRef(handler);
  latest.current = handler;

  useEffect(() => {
    const el: EventTarget | null = target === undefined ? window : target;
    if (el === null) return;
    el.addEventListener(type, (event: Event) => latest.current(event as WindowEventMap[K]));
  }, [type, target]);
}
