// Screens live at the site root, matched and navigated by wouter. The server
// (mcp/dash_static.py) answers any unknown non-API path that asks for HTML
// with index.html, so a refresh on /data/<id> survives, and redirects old
// /dash/<route> bookmarks here.
import { useLocation, useRouter, matchRoute, type Parser } from 'wouter';
import { navigate as go } from 'wouter/use-browser-location';

export type Route =
  | { name: 'tokonoma' }
  | { name: 'nebari' }
  | { name: 'naedoko' }
  | { name: 'dataset'; id: string }
  | { name: 'sentei' }
  | { name: 'phase0' }
  | { name: 'notfound'; path: string };

/** wouter patterns, in match order. */
export const PATTERNS = {
  tokonoma: '/',
  nebari: '/nebari',
  naedoko: '/data',
  dataset: '/data/:id',
  sentei: '/results',
  phase0: '/phase0',
} as const;

export function resolve(parser: Parser, pathname: string): Route {
  for (const name of ['tokonoma', 'nebari', 'naedoko', 'sentei', 'phase0'] as const) {
    if (matchRoute(parser, PATTERNS[name], pathname)[0]) return { name };
  }
  const [hit, params] = matchRoute(parser, PATTERNS.dataset, pathname);
  if (hit && params.id) return { name: 'dataset', id: decodeURIComponent(params.id) };
  return { name: 'notfound', path: pathname };
}

export const href = {
  tokonoma: '/',
  nebari: '/nebari',
  naedoko: '/data',
  dataset: (id: string) => `/data/${encodeURIComponent(id)}`,
  sentei: '/results',
};

export function useRoute(): Route {
  const router = useRouter();
  const [path] = useLocation();
  return resolve(router.parser, path);
}

export function navigate(to: string) {
  if (to === window.location.pathname) return;
  go(to);
  window.scrollTo(0, 0);
}
