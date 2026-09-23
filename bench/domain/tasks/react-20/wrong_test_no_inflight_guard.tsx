// Wrong: no in-flight guard, so a second intersection during a load starts a duplicate request.
import { useCallback, useEffect, useRef, useState } from 'react';

type Page = { items: string[]; hasMore: boolean };

export function InfiniteList({ loadPage }: { loadPage: (page: number) => Promise<Page> }) {
  const [items, setItems] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);

  // Refs are the source of truth for the guards, so the observer callback
  // never reads a stale render's state.
  const inFlight = useRef(false);
  const nextPage = useRef(0);
  const hasMore = useRef(true);
  const loadRef = useRef(loadPage);
  loadRef.current = loadPage;
  const sentinel = useRef<HTMLDivElement>(null);

  const loadNext = useCallback(() => {
    if (!hasMore.current) return;
    inFlight.current = true;
    setLoading(true);
    const page = nextPage.current;
    loadRef.current(page).then(
      (res) => {
        nextPage.current = page + 1;
        hasMore.current = res.hasMore;
        inFlight.current = false;
        setItems((prev) => [...prev, ...res.items]);
        setLoading(false);
        if (!res.hasMore) setDone(true);
      },
      () => {
        inFlight.current = false;
        setLoading(false);
      },
    );
  }, []);

  useEffect(() => {
    loadNext();
  }, [loadNext]);

  useEffect(() => {
    const el = sentinel.current;
    if (!el) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting && e.target === el)) loadNext();
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [loadNext]);

  return (
    <div>
      <ul>
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
      <div data-testid="sentinel" ref={sentinel} />
      {loading && <p role="status">Loading…</p>}
      {done && <p>No more items</p>}
    </div>
  );
}
