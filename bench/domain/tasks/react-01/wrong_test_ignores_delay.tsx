import { useEffect, useState } from 'react';

// Wrong: delayMs is left out of the dependencies, so changing it does not
// restart the wait.
export function useDebounce<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState<T>(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  return debounced;
}
