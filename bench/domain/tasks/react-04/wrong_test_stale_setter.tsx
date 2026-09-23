import { useEffect, useState } from 'react';

// Wrong: the functional update is applied to the value captured at render
// time, so two updates in one handler collapse into one.
function read<T>(key: string, fallback: T): T {
  if (typeof window === 'undefined') return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    return raw === null ? fallback : (JSON.parse(raw) as T);
  } catch {
    return fallback;
  }
}

export function useLocalStorage<T>(
  key: string,
  initialValue: T,
): [T, (next: T | ((prev: T) => T)) => void, () => void] {
  const [value, setValue] = useState<T>(() => read(key, initialValue));

  const set = (next: T | ((prev: T) => T)) => {
    const v = next instanceof Function ? next(value) : next;
    window.localStorage.setItem(key, JSON.stringify(v));
    setValue(() => v);
  };

  const remove = () => {
    window.localStorage.removeItem(key);
    setValue(() => initialValue);
  };

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === key) setValue(() => read(key, initialValue));
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, [key, initialValue]);

  return [value, set, remove];
}
