import { useEffect, useRef } from 'react';

// Two refs committed in an effect; the answer for this render is derived
// from them without mutating anything during render.
export function usePrevious<T>(value: T): T | undefined {
  const committed = useRef<T>(value);
  const before = useRef<T | undefined>(undefined);

  const changed = !Object.is(committed.current, value);
  const result = changed ? committed.current : before.current;

  useEffect(() => {
    if (!Object.is(committed.current, value)) {
      before.current = committed.current;
      committed.current = value;
    }
  });

  return result;
}
