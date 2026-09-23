import { useEffect, useRef } from 'react';

// Wrong (strict): the ref holds T | undefined but the function promises T.
export function usePrevious<T>(value: T): T {
  const current = useRef<T>(value);
  const previous = useRef<T | undefined>(undefined);
  useEffect(() => {
    if (!Object.is(current.current, value)) {
      previous.current = current.current;
      current.current = value;
    }
  });
  return Object.is(current.current, value) ? previous.current : current.current;
}
