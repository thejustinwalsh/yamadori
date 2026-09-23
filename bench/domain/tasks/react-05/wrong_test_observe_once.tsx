import { useEffect, useRef, useState } from 'react';

// Wrong: the callback only stores the node; observing happens once in a
// mount effect, so an element that appears later is never observed.
export function useInView<T extends Element = Element>(
  options?: IntersectionObserverInit,
): { ref: (node: T | null) => void; inView: boolean } {
  const nodeRef = useRef<T | null>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const node = nodeRef.current;
    if (!node) return;
    const io = new IntersectionObserver(([entry]) => {
      if (entry) setInView(entry.isIntersecting);
    }, options);
    io.observe(node);
    return () => io.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    ref: (node: T | null) => {
      nodeRef.current = node;
    },
    inView,
  };
}
