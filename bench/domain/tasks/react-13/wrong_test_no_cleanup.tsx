// Wrong: the effect never cleans up, so the listeners and the pending timer
// outlive the component.
import { useEffect, useRef, useState } from 'react';

const EVENTS = ['mousemove', 'mousedown', 'keydown', 'touchstart', 'scroll', 'wheel'];

export function useIdle(timeoutMs: number): boolean {
  const [idle, setIdle] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => {
    const arm = () => {
      clearTimeout(timer.current);
      timer.current = setTimeout(() => setIdle(true), timeoutMs);
    };
    arm();
    EVENTS.forEach((e) =>
      window.addEventListener(e, () => {
        setIdle(false);
        arm();
      }),
    );
  }, [timeoutMs]);

  return idle;
}
