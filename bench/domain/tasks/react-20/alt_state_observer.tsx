import { useEffect, useReducer, useState } from 'react';
import type * as React from 'react';

// All guards live in reducer state; the observer is torn down and recreated
// whenever that state changes (so its closure is always fresh), and the
// sentinel is attached through a callback ref and removed once done.
type S = { items: string[]; page: number; loading: boolean; more: boolean };
type A = { t: 'start' } | { t: 'done'; items: string[]; more: boolean } | { t: 'error' };

function reducer(s: S, a: A): S {
  switch (a.t) {
    case 'start':
      return { ...s, loading: true };
    case 'done':
      return { items: s.items.concat(a.items), page: s.page + 1, loading: false, more: a.more };
    case 'error':
      return { ...s, loading: false };
  }
}

export function InfiniteList(props: {
  loadPage: (page: number) => Promise<{ items: string[]; hasMore: boolean }>;
}): React.ReactElement {
  const [s, dispatch] = useReducer(reducer, { items: [], page: 0, loading: true, more: true });
  const [node, setNode] = useState<HTMLElement | null>(null);
  const [request, setRequest] = useState(0);

  // `request` counts load requests; each value triggers exactly one fetch of
  // the page that is next at that moment.
  useEffect(() => {
    let live = true;
    dispatch({ t: 'start' });
    props.loadPage(s.page).then(
      (r) => {
        if (live) dispatch({ t: 'done', items: r.items, more: r.hasMore });
      },
      () => {
        if (live) dispatch({ t: 'error' });
      },
    );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request]);

  useEffect(() => {
    if (!node || s.loading || !s.more) return;
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          io.disconnect();
          setRequest((n) => n + 1);
          return;
        }
      }
    }, { rootMargin: '200px' });
    io.observe(node);
    return () => io.disconnect();
  }, [node, s.loading, s.more]);

  return (
    <>
      <ul>
        {s.items.map((it, i) => (
          <li key={`${i}:${it}`}>{it}</li>
        ))}
      </ul>
      {s.more ? <span data-testid="sentinel" ref={setNode} /> : <p>No more items</p>}
      {s.loading ? <div role="status">Loading…</div> : null}
    </>
  );
}
