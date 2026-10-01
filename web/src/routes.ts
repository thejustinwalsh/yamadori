// What each screen needs before it can draw: its code chunk and its data.
// A nav link starts both on intent (hover, focus, press), so by the click the
// screen usually mounts with its snapshot instead of a loading state.
import { lazy, type ComponentType } from 'react';
import { peek, prefetch } from './api/cache';
import { PATHS } from './api/data';
import { LIBRARY_PATH } from './api/library';
import { IMAGE_SETTINGS_PATH } from './api/settings';
import { HARNESS_KIT_PATH } from './api/harness';
import { JJAVA_PATH, PERF_PATH } from './api/stats';
import { ONBOARDING_PATH, onboardingPath } from './api/onboarding';
import { PROMPTS_PATH, RECENT_PATH, SELECTIONS_PATH, SKILLS_PATH, skillPath } from './api/skills';
import type { Route } from './router';

type Chunk<P> = { Component: ComponentType<P>; preload: () => Promise<void>; loaded: () => boolean };

function chunk<P extends object>(load: () => Promise<ComponentType<P>>): Chunk<P> {
  let done: ComponentType<P> | null = null;
  let pending: Promise<void> | null = null;
  const preload = () =>
    (pending ??= load().then(
      (c) => {
        done = c;
      },
      () => {
        pending = null; // a failed chunk is retried on the next intent; the render surfaces the error
      },
    ));
  // Once preloaded, lazy() gets a thenable that resolves synchronously, so
  // React renders the screen at once instead of suspending for a microtask
  // and flashing the fallback.
  const Component = lazy(() =>
    done
      ? ({ then: (ok: (m: { default: ComponentType<P> }) => void) => ok({ default: done as ComponentType<P> }) } as unknown as Promise<{
          default: ComponentType<P>;
        }>)
      : load().then((c) => {
          done = c;
          return { default: c };
        }),
  );
  return { Component, preload, loaded: () => done !== null };
}

export const screens = {
  naedoko: chunk(() => import('./screens/Naedoko').then((m) => m.Naedoko)),
  dataset: chunk(() => import('./screens/DatasetDetail').then((m) => m.DatasetDetail)),
  perf: chunk(() => import('./screens/Performance').then((m) => m.PerformanceScreen)),
  jjava: chunk(() => import('./screens/Jjava').then((m) => m.JjavaScreen)),
  settings: chunk(() => import('./screens/Settings').then((m) => m.Settings)),
  skills: chunk(() => import('./screens/Skills').then((m) => m.Skills)),
  skill: chunk(() => import('./screens/SkillDetail').then((m) => m.SkillScreen)),
  onboarding: chunk(() => import('./screens/Onboarding').then((m) => m.OnboardingScreen)),
  harness: chunk(() => import('./screens/Harness').then((m) => m.Harness)),
  phase0: chunk(() => import('./phase0/Phase0').then((m) => m.Phase0)),
};

/**
 * The /dash/api paths a screen polls itself. TOKONOMA and NAEDOKO read only
 * the app-wide polls (vitals, datasets, tiers), which are always warm.
 */
export function dataFor(route: Route): string[] {
  switch (route.name) {
    case 'dataset':
      return [PATHS.dataset(route.id)];
    // the window the page last showed is the viewer's (StatsParts useWindow);
    // intent warms the default one
    case 'perf':
      return [PERF_PATH];
    case 'jjava':
      return [JJAVA_PATH];
    case 'settings':
      return [IMAGE_SETTINGS_PATH];
    case 'skills':
      return route.view === 'selections' ? [SKILLS_PATH, RECENT_PATH, SELECTIONS_PATH, JJAVA_PATH] : route.view === 'prompts' ? [PROMPTS_PATH] : route.view === 'create' ? [SKILLS_PATH, ONBOARDING_PATH] : [SKILLS_PATH, LIBRARY_PATH];
    case 'skill':
      return [SKILLS_PATH, skillPath(route.id)];
    case 'onboarding':
      return [onboardingPath(route.id)];
    case 'harness':
      return [HARNESS_KIT_PATH];
    default:
      return [];
  }
}

function chunkFor(route: Route): Pick<Chunk<object>, 'preload' | 'loaded'> | null {
  switch (route.name) {
    case 'naedoko':
    case 'dataset':
    case 'perf':
    case 'jjava':
    case 'settings':
    case 'skills':
    case 'skill':
    case 'onboarding':
    case 'harness':
    case 'phase0':
      return screens[route.name];
    default:
      return null;
  }
}

/**
 * Start loading `route`. `ready` says whether it can draw with data right
 * now; `done` settles when everything started here has (it never rejects).
 */
export function preloadRoute(route: Route): { ready: boolean; done: Promise<void> } {
  const c = chunkFor(route);
  const paths = dataFor(route);
  // A snapshot of any age counts: the screen shows it with its age while its
  // own poll refreshes it.
  const ready = (!c || c.loaded()) && paths.every((p) => peek(p) !== null);
  const done = Promise.all([c?.preload(), ...paths.map((p) => prefetch(p))]).then(
    () => undefined,
    () => undefined,
  );
  return { ready, done };
}
