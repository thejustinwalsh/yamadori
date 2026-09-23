import { useEffect, useState } from 'react';

// Wrong: no abort and no staleness check, so a slow old response overwrites
// the newer one and nothing is cancelled.
export function useFetch<T>(url: string | null): {
  data: T | undefined;
  error: Error | undefined;
  loading: boolean;
} {
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<Error | undefined>(undefined);
  const [loading, setLoading] = useState(url !== null);

  useEffect(() => {
    setData(undefined);
    setError(undefined);
    if (url === null) {
      setLoading(false);
      return;
    }
    setLoading(true);
    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json() as Promise<T>;
      })
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e : new Error(String(e)));
        setLoading(false);
      });
  }, [url]);

  return { data, error, loading };
}
