import { useEffect, useReducer, useRef } from 'react';

// useReducer + async/await; a request counter decides which response is
// current, and AbortErrors are recognised by name.
type State<T> = { data: T | undefined; error: Error | undefined; loading: boolean };
type Action<T> =
  | { type: 'idle' }
  | { type: 'start' }
  | { type: 'done'; data: T }
  | { type: 'fail'; error: Error };

function reducer<T>(_: State<T>, a: Action<T>): State<T> {
  switch (a.type) {
    case 'idle':
      return { data: undefined, error: undefined, loading: false };
    case 'start':
      return { data: undefined, error: undefined, loading: true };
    case 'done':
      return { data: a.data, error: undefined, loading: false };
    case 'fail':
      return { data: undefined, error: a.error, loading: false };
  }
}

export function useFetch<T>(url: string | null): State<T> {
  const [state, dispatch] = useReducer(reducer<T>, {
    data: undefined,
    error: undefined,
    loading: url !== null,
  });
  const current = useRef(0);

  useEffect(() => {
    const id = ++current.current;
    if (url === null) {
      dispatch({ type: 'idle' });
      return;
    }
    const controller = new AbortController();
    dispatch({ type: 'start' });

    const run = async () => {
      try {
        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as T;
        if (id === current.current) dispatch({ type: 'done', data });
      } catch (e) {
        if (e instanceof Error && e.name === 'AbortError') return;
        if (id !== current.current) return;
        dispatch({ type: 'fail', error: e instanceof Error ? e : new Error(String(e)) });
      }
    };
    void run();

    return () => {
      current.current++;
      controller.abort();
    };
  }, [url]);

  return state;
}
