import { useEffect, useState } from 'react';

// Wrong: the previous timer is never cleared, so an intermediate value lands
// and a timer outlives the component.
export function useDebounce<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState<T>(value);
  useEffect(() => {
    setTimeout(() => setDebounced(value), delayMs);
  }, [value, delayMs]);
  return debounced;
}
