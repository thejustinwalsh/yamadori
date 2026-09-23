import { useState } from 'react';

// Wrong: seeds "previous" with the first value instead of undefined, so the
// hook claims a previous value before anything has changed.
export function usePrevious<T>(value: T): T | undefined {
  const [current, setCurrent] = useState<T>(value);
  const [previous, setPrevious] = useState<T | undefined>(value);
  if (!Object.is(current, value)) {
    setPrevious(() => current);
    setCurrent(() => value);
  }
  return previous;
}
