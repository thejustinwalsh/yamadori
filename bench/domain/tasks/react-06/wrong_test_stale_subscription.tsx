import { useCallback, useSyncExternalStore } from 'react';

// Wrong: subscribe is memoised once, so after `query` changes it is still
// listening to the old query and misses changes of the new one.
export function useMediaQuery(query: string, serverFallback = false): boolean {
  const subscribe = useCallback((onChange: () => void) => {
    const mql = window.matchMedia(query);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(query).matches,
    () => serverFallback,
  );
}
