import { useEffect, useState } from 'react';

// Wrong: ignores stale responses with a flag but never creates or passes an
// AbortSignal, so superseded requests are never cancelled.
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
    let ignore = false;
    setState({ data: undefined, error: undefined, loading: true });
    fetch(url)
      .then(async (res) => {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return (await res.json()) as T;
      })
      .then(
        (data) => {
          if (!ignore) setState({ data, error: undefined, loading: false });
        },
        (e: unknown) => {
          if (!ignore)
            setState({
              data: undefined,
              error: e instanceof Error ? e : new Error(String(e)),
              loading: false,
            });
        },
      );
    return () => {
      ignore = true;
    };
  }, [url]);

  return state;
}
