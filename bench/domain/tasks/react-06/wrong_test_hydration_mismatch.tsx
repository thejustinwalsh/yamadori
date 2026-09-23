import { useEffect, useState } from 'react';

// Wrong: reads matchMedia while initialising state, so the client's
// hydration render disagrees with the server HTML (a hydration mismatch).
export function useMediaQuery(query: string, serverFallback = false): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window === 'undefined' ? serverFallback : window.matchMedia(query).matches,
  );
  useEffect(() => {
    const mql = window.matchMedia(query);
    setMatches(mql.matches);
    const onChange = (e: MediaQueryListEvent) => setMatches(e.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, [query]);
  return matches;
}
