import { useCallback, useSyncExternalStore } from 'react';

// Wrong: no getServerSnapshot, so server rendering and hydration throw, and
// serverFallback is never used.
export function useMediaQuery(query: string, serverFallback = false): boolean {
  void serverFallback;
  const subscribe = useCallback(
    (onChange: () => void) => {
      const mql = window.matchMedia(query);
      mql.addEventListener('change', onChange);
      return () => mql.removeEventListener('change', onChange);
    },
    [query],
  );
  return useSyncExternalStore(subscribe, () => window.matchMedia(query).matches);
}
