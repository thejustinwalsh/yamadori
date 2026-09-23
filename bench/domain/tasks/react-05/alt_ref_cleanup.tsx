import { useCallback, useRef, useState } from 'react';

// React 19 callback ref that returns its own cleanup: the observer lives
// exactly as long as the element is attached. Options are read from a ref so
// the callback identity stays stable.
export function useInView<T extends Element = Element>(
  options?: IntersectionObserverInit,
): { ref: (node: T | null) => void; inView: boolean } {
  const [inView, setInView] = useState(false);
  const opts = useRef(options);
  opts.current = options;

  const ref = useCallback((node: T | null) => {
    if (node === null) return;
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) setInView(e.isIntersecting);
    }, opts.current);
    io.observe(node);
    return () => {
      io.unobserve(node);
      io.disconnect();
    };
  }, []);

  return { ref, inView };
}
