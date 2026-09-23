import { useEffect, useState } from 'react';

interface FetchState<T> {
  data: T | undefined;
  error: Error | undefined;
  loading: boolean;
}

export function useFetch<T>(url: string | null): FetchState<T> {
  const [state, setState] = useState<FetchState<T>>({
    data: undefined,
    error: undefined,
    loading: url !== null,
  });

  useEffect(() => {
    if (url === null) {
      setState({ data: undefined, error: undefined, loading: false });
      return;
    }
    const controller = new AbortController();
    const { signal } = controller;
    setState({ data: undefined, error: undefined, loading: true });

    fetch(url, { signal })
      .then(async (res) => {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return (await res.json()) as T;
      })
      .then(
        (data) => {
          if (!signal.aborted) setState({ data, error: undefined, loading: false });
        },
        (reason: unknown) => {
          if (signal.aborted) return;
          const error = reason instanceof Error ? reason : new Error(String(reason));
          setState({ data: undefined, error, loading: false });
        },
      );

    return () => controller.abort();
  }, [url]);

  return state;
}
