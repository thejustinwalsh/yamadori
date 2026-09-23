// Wrong: only listens for mouse and keyboard events; touch, scroll and wheel
// activity is ignored.
import { useEffect, useState } from 'react';

export function useIdle(timeoutMs: number): boolean {
  const [idle, setIdle] = useState(false);

  useEffect(() => {
    let id = setTimeout(() => setIdle(true), timeoutMs);
    const onActivity = () => {
      setIdle(false);
      clearTimeout(id);
      id = setTimeout(() => setIdle(true), timeoutMs);
    };
    const events = ['mousemove', 'mousedown', 'keydown'];
    events.forEach((e) => window.addEventListener(e, onActivity));
    return () => {
      clearTimeout(id);
      events.forEach((e) => window.removeEventListener(e, onActivity));
    };
  }, [timeoutMs]);

  return idle;
}
