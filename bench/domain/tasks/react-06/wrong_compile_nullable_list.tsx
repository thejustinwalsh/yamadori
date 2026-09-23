import { useEffect, useState } from 'react';

// Wrong (strict): the MediaQueryList is null on the server, and is used
// without narrowing.
export function useMediaQuery(query: string, serverFallback = false): boolean {
  const mql = typeof window !== 'undefined' ? window.matchMedia(query) : null;
  const [matches, setMatches] = useState<boolean>(mql ? mql.matches : serverFallback);
  useEffect(() => {
    const onChange = () => setMatches(mql.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, [mql]);
  return matches;
}
