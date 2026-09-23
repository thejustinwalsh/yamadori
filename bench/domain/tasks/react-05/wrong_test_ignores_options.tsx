import { useEffect, useState } from 'react';

// Wrong: the observer is constructed without the caller's options.
export function useInView<T extends Element = Element>(
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  options?: IntersectionObserverInit,
): { ref: (node: T | null) => void; inView: boolean } {
  const [node, setNode] = useState<T | null>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    if (!node) return;
    const io = new IntersectionObserver((entries) => {
      const last = entries[entries.length - 1];
      if (last) setInView(last.isIntersecting);
    });
    io.observe(node);
    return () => io.disconnect();
  }, [node]);
  return { ref: setNode, inView };
}
