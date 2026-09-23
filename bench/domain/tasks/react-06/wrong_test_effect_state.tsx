import { useEffect, useState } from 'react';

// Wrong: starts at false and only reads matchMedia in an effect, so the
// first render is wrong and serverFallback is ignored.
export function useMediaQuery(query: string, serverFallback = false): boolean {
  void serverFallback;
  const [matches, setMatches] = useState(false);
  useEffect(() => {
    const mql = window.matchMedia(query);
    setMatches(mql.matches);
    const onChange = (e: MediaQueryListEvent) => setMatches(e.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, [query]);
  return matches;
}
