import { useCallback, useMemo, useSyncExternalStore } from 'react';

// The raw stored string is the external store; the parsed value is derived
// from it. Same-tab writes notify subscribers through a module-level set.
const local = new Set<() => void>();

function subscribe(onChange: () => void): () => void {
  local.add(onChange);
  window.addEventListener('storage', onChange);
  return () => {
    local.delete(onChange);
    window.removeEventListener('storage', onChange);
  };
}

function readRaw(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function decode<T>(raw: string | null, fallback: T): T {
  if (raw === null) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function useLocalStorage<T>(
  key: string,
  initialValue: T,
): [T, (next: T | ((prev: T) => T)) => void, () => void] {
  const raw = useSyncExternalStore(
    subscribe,
    () => readRaw(key),
    () => null,
  );
  const value = useMemo(() => decode(raw, initialValue), [raw, initialValue]);

  const set = useCallback(
    (next: T | ((prev: T) => T)) => {
      const prev = decode(readRaw(key), initialValue);
      const v = next instanceof Function ? next(prev) : next;
      try {
        window.localStorage.setItem(key, JSON.stringify(v));
      } catch {
        return;
      }
      local.forEach((fn) => fn());
    },
    [key, initialValue],
  );

  const remove = useCallback(() => {
    try {
      window.localStorage.removeItem(key);
    } catch {
      return;
    }
    local.forEach((fn) => fn());
  }, [key]);

  return [value, set, remove];
}
