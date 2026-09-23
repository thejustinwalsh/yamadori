import { useEffect, useRef } from 'react';

// Wrong: stores the value after every render, so a re-render that does not
// change the value makes "previous" equal the current value.
export function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T | undefined>(undefined);
  useEffect(() => {
    ref.current = value;
  });
  return ref.current;
}
