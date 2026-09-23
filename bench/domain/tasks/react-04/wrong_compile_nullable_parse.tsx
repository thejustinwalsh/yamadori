import { useState } from 'react';

// Wrong (strict): getItem returns string | null, JSON.parse needs a string.
export function useLocalStorage<T>(
  key: string,
  initialValue: T,
): [T, (next: T | ((prev: T) => T)) => void, () => void] {
  const [value, setValue] = useState<T>(() => {
    if (typeof window === 'undefined') return initialValue;
    try {
      return window.localStorage.getItem(key) !== null
        ? (JSON.parse(window.localStorage.getItem(key)) as T)
        : initialValue;
    } catch {
      return initialValue;
    }
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
    setValue(() => initialValue);
  };

  return [value, set, remove];
}
