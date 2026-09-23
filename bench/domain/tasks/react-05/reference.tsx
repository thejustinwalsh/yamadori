import { useEffect, useState } from 'react';

export function useInView<T extends Element = Element>(
  options?: IntersectionObserverInit,
): { ref: (node: T | null) => void; inView: boolean } {
  const [node, setNode] = useState<T | null>(null);
  const [inView, setInView] = useState(false);

  const root = options?.root ?? null;
  const rootMargin = options?.rootMargin;
  const threshold = JSON.stringify(options?.threshold ?? null);

  useEffect(() => {
    if (!node) return;
    const t = JSON.parse(threshold) as number | number[] | null;
    const io = new IntersectionObserver(
      (entries) => {
        const last = entries[entries.length - 1];
        if (last) setInView(last.isIntersecting);
      },
      {
        root,
        ...(rootMargin !== undefined ? { rootMargin } : {}),
        ...(t !== null ? { threshold: t } : {}),
      },
    );
    io.observe(node);
    return () => io.disconnect();
  }, [node, root, rootMargin, threshold]);

  return { ref: setNode, inView };
}
