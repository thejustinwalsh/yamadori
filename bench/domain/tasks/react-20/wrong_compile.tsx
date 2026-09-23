import { useEffect, useRef, useState } from 'react';

// Wrong (strict): the sentinel ref may still be null when it is observed.
export function InfiniteList({
  loadPage,
}: {
  loadPage: (page: number) => Promise<{ items: string[]; hasMore: boolean }>;
}) {
  const [items, setItems] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);
  const sentinel = useRef<HTMLDivElement>(null);
  const state = useRef({ page: 0, inFlight: false, hasMore: true });

  const loadNext = () => {
    const s = state.current;
    if (s.inFlight || !s.hasMore) return;
    s.inFlight = true;
    setLoading(true);
    loadPage(s.page).then((res) => {
      s.page += 1;
      s.hasMore = res.hasMore;
      s.inFlight = false;
      setItems((prev) => [...prev, ...res.items]);
      setLoading(false);
      if (!res.hasMore) setDone(true);
    });
  };

  useEffect(() => {
    loadNext();
    const observer = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) loadNext();
    });
    observer.observe(sentinel.current);
    return () => observer.disconnect();
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
      {done && <p>No more items</p>}
    </div>
  );
}
