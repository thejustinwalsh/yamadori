import { useEffect, useRef, useState } from 'react';

// Wrong: the observer is created once and its callback closes over the first
// render's state, so it always thinks page 0 is the last one loaded and never
// sees `loading` or `hasMore` change.
export function InfiniteList({
  loadPage,
}: {
  loadPage: (page: number) => Promise<{ items: string[]; hasMore: boolean }>;
}) {
  const [items, setItems] = useState<string[]>([]);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const sentinel = useRef<HTMLDivElement>(null);

  const load = (p: number) => {
    setLoading(true);
    loadPage(p).then((res) => {
      setItems((prev) => [...prev, ...res.items]);
      setPage(p);
      setHasMore(res.hasMore);
      setLoading(false);
    });
  };

  const loadMore = () => {
    if (loading || !hasMore) return;
    load(page + 1);
  };

  useEffect(() => {
    load(0);
    const el = sentinel.current;
    if (!el) return;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) loadMore();
    });
    observer.observe(el);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div>
      <ul>
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
      <div data-testid="sentinel" ref={sentinel} />
      {loading && <p role="status">Loading…</p>}
      {!hasMore && <p>No more items</p>}
    </div>
  );
}
