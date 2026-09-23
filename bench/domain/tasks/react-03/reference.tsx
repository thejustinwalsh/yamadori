import { useState } from 'react';

// "Storing information from previous renders": compare during render and
// update state immediately, so the change is seen in the same render pass.
export function usePrevious<T>(value: T): T | undefined {
  const [pair, setPair] = useState<{ current: T; previous: T | undefined }>({
    current: value,
    previous: undefined,
  });
  if (!Object.is(pair.current, value)) {
    const next = { current: value, previous: pair.current };
    setPair(next);
    return next.previous;
  }
  return pair.previous;
}
