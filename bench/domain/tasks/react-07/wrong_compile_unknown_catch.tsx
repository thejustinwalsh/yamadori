import { useEffect, useState } from 'react';

// Wrong (strict): a catch variable is `unknown`, not `Error`.
export function useFetch<T>(url: string | null): {
  data: T | undefined;
  error: Error | undefined;
  loading: boolean;
} {
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<Error | undefined>(undefined);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (url === null) return;
    const controller = new AbortController();
    const run = async () => {
      setLoading(true);
      try {
        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        setData((await res.json()) as T);
      } catch (e) {
        if (!controller.signal.aborted) setError(e);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    };
    void run();
    return () => controller.abort();
  }, [url]);

  return { data, error, loading };
}
