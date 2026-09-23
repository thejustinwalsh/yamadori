import { useCallback, useState } from 'react';

// Wrong: observes every node it is given and never unobserves or
// disconnects, so old and unmounted elements stay observed.
export function useInView<T extends Element = Element>(
  options?: IntersectionObserverInit,
): { ref: (node: T | null) => void; inView: boolean } {
  const [inView, setInView] = useState(false);
  const ref = useCallback(
    (node: T | null) => {
      if (!node) return;
      const io = new IntersectionObserver((entries) => {
        const last = entries[entries.length - 1];
        if (last) setInView(last.isIntersecting);
      }, options);
      io.observe(node);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
  return { ref, inView };
}
