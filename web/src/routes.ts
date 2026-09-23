// What each screen needs before it can draw: its code chunk and its data.
// A nav link starts both on intent (hover, focus, press), so by the click the
// screen usually mounts with its snapshot instead of a loading state.
import { lazy, type ComponentType } from 'react';
import { peek, prefetch } from './api/cache';
import { PATHS } from './api/data';
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
  nebari: chunk(() => import('./screens/Nebari').then((m) => m.Nebari)),
  naedoko: chunk(() => import('./screens/Naedoko').then((m) => m.Naedoko)),
  dataset: chunk(() => import('./screens/DatasetDetail').then((m) => m.DatasetDetail)),
  sentei: chunk(() => import('./screens/Sentei').then((m) => m.Sentei)),
  phase0: chunk(() => import('./phase0/Phase0').then((m) => m.Phase0)),
};

/**
 * The /dash/api paths a screen polls itself. TOKONOMA and NAEDOKO read only
 * the app-wide polls (vitals, datasets, tiers), which are always warm.
 */
export function dataFor(route: Route): string[] {
  switch (route.name) {
    case 'nebari':
      return [PATHS.stats, PATHS.results];
    case 'dataset':
      return [PATHS.dataset(route.id)];
    case 'sentei':
      return [PATHS.results];
    default:
      return [];
  }
}

function chunkFor(route: Route): Pick<Chunk<object>, 'preload' | 'loaded'> | null {
  switch (route.name) {
    case 'nebari':
    case 'naedoko':
    case 'dataset':
    case 'sentei':
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
