import { useMemo, useSyncExternalStore } from 'react';

// One store object per query string, cached at module level, using the
// older addListener/removeListener API of MediaQueryList.
interface Store {
  subscribe: (onChange: () => void) => () => void;
  get: () => boolean;
}

const stores = new Map<string, Store>();

function storeFor(query: string): Store {
  let s = stores.get(query);
  if (!s) {
    s = {
      subscribe(onChange) {
        const mql = window.matchMedia(query);
        const fn = () => onChange();
        mql.addListener(fn);
        return () => mql.removeListener(fn);
      },
      get() {
        return window.matchMedia(query).matches;
      },
    };
    stores.set(query, s);
  }
  return s;
}

export function useMediaQuery(query: string, serverFallback?: boolean): boolean {
  const store = useMemo(() => storeFor(query), [query]);
  const fallback = serverFallback ?? false;
  return useSyncExternalStore(store.subscribe, store.get, () => fallback);
}
