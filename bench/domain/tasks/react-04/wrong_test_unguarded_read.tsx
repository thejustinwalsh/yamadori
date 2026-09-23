import { useEffect, useState } from 'react';

// Wrong: the initial read is unguarded -- it throws on the server, when
// storage access is denied, and on stored text that is not JSON.
export function useLocalStorage<T>(
  key: string,
  initialValue: T,
): [T, (next: T | ((prev: T) => T)) => void, () => void] {
  const [value, setValue] = useState<T>(() => {
    const raw = window.localStorage.getItem(key);
    return raw !== null ? (JSON.parse(raw) as T) : initialValue;
  });

  const set = (next: T | ((prev: T) => T)) => {
    setValue((prev) => {
      const v = next instanceof Function ? next(prev) : next;
      window.localStorage.setItem(key, JSON.stringify(v));
      return v;
    });
  };

  const remove = () => {
    window.localStorage.removeItem(key);
    setValue(initialValue);
  };

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === key) {
        setValue(e.newValue === null ? initialValue : (JSON.parse(e.newValue) as T));
      }
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, [key, initialValue]);

  return [value, set, remove];
}
