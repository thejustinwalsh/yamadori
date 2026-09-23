// Wrong: activity only sets `false`; the countdown started at mount is never
// restarted, so the hook goes idle timeoutMs after mount regardless.
import { useEffect, useState } from 'react';

const EVENTS = ['mousemove', 'mousedown', 'keydown', 'touchstart', 'scroll', 'wheel'];

export function useIdle(timeoutMs: number): boolean {
  const [idle, setIdle] = useState(false);

  useEffect(() => {
    const id = setTimeout(() => setIdle(true), timeoutMs);
    return () => clearTimeout(id);
  }, [timeoutMs]);

  useEffect(() => {
    const onActivity = () => setIdle(false);
    EVENTS.forEach((e) => window.addEventListener(e, onActivity));
    return () => EVENTS.forEach((e) => window.removeEventListener(e, onActivity));
  }, []);

  return idle;
}
