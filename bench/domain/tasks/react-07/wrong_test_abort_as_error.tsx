import { useEffect, useState } from 'react';

// Wrong: aborts correctly, but the AbortError that fetch rejects with is
// reported as an error for the new request.
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
    setState({ data: undefined, error: undefined, loading: true });
    fetch(url, { signal: controller.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return (await res.json()) as T;
      })
      .then(
        (data) => setState({ data, error: undefined, loading: false }),
        (e: unknown) =>
          setState({
            data: undefined,
            error: e instanceof Error ? e : new Error(String(e)),
            loading: false,
          }),
      );
    return () => controller.abort();
  }, [url]);

  return state;
}
