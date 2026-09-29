// Screens live at the site root, matched and navigated by wouter. The server
// (mcp/dash_static.py) answers any unknown non-API path that asks for HTML
// with index.html, so a refresh on /data/<id> survives, and redirects old
// /dash/<route> bookmarks here.
import { useLocation, useRouter, matchRoute, type Parser } from 'wouter';
import { navigate as go } from 'wouter/use-browser-location';

/** The skill factory's views under /skills (a skill's own page is /skills/<id>). */
export const SKILLS_VIEWS = ['library', 'create', 'selections', 'prompts'] as const;
export type SkillsView = (typeof SKILLS_VIEWS)[number];

export type Route =
  | { name: 'tokonoma' }
  | { name: 'nebari' }
  | { name: 'naedoko' }
  | { name: 'dataset'; id: string }
  | { name: 'sentei' }
  | { name: 'settings' }
  | { name: 'skills'; view: SkillsView }
  | { name: 'skill'; id: string }
  | { name: 'onboarding'; id: string }
  | { name: 'harness' }
  | { name: 'phase0' }
  | { name: 'notfound'; path: string };

/** wouter patterns, in match order. */
export const PATTERNS = {
  tokonoma: '/',
  nebari: '/nebari',
  naedoko: '/data',
  dataset: '/data/:id',
  sentei: '/results',
  settings: '/settings',
  skills: '/skills',
  skill: '/skills/:id',
  onboarding: '/skills/onboarding/:id',
  harness: '/harness',
  phase0: '/phase0',
} as const;

export function resolve(parser: Parser, pathname: string): Route {
  for (const name of ['tokonoma', 'nebari', 'naedoko', 'sentei', 'settings', 'harness', 'phase0'] as const) {
    if (matchRoute(parser, PATTERNS[name], pathname)[0]) return { name };
  }
  if (matchRoute(parser, PATTERNS.skills, pathname)[0]) return { name: 'skills', view: 'library' };
  const [ob, op] = matchRoute(parser, PATTERNS.onboarding, pathname);
  if (ob && op.id) return { name: 'onboarding', id: decodeURIComponent(op.id) };
  const [sk, sp] = matchRoute(parser, PATTERNS.skill, pathname);
  if (sk && sp.id) {
    const id = decodeURIComponent(sp.id);
    return (SKILLS_VIEWS as readonly string[]).includes(id) ? { name: 'skills', view: id as SkillsView } : { name: 'skill', id };
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
  settings: '/settings',
  skills: '/skills',
  skillsView: (v: SkillsView) => (v === 'library' ? '/skills' : `/skills/${v}`),
  skill: (id: string) => `/skills/${encodeURIComponent(id)}`,
  onboarding: (id: string) => `/skills/onboarding/${encodeURIComponent(id)}`,
  harness: '/harness',
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
