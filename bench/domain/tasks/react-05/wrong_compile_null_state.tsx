import { useEffect, useState } from 'react';

// Wrong (strict): `useState(null)` infers the state as `null`, so the setter
// cannot accept an element and the returned ref has the wrong type.
export function useInView<T extends Element = Element>(
  options?: IntersectionObserverInit,
): { ref: (node: T | null) => void; inView: boolean } {
  const [node, setNode] = useState(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    if (!node) return;
    const io = new IntersectionObserver((entries) => {
      setInView(entries[entries.length - 1].isIntersecting);
    }, options);
    io.observe(node);
    return () => io.disconnect();
  }, [node, options]);
  return { ref: setNode, inView };
}
