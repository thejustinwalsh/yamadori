// No `const` modifier: string arrays widen to string[], so tags lose their literal tuple type.
export type Method = 'GET' | 'POST' | 'PUT' | 'DELETE';
export type RouteDef = { path: `/${string}`; method: Method; tags?: readonly string[] };
export function defineRoutes<R extends Record<string, RouteDef>>(routes: R): R {
  return routes;
}
export function pathOf<R extends Record<string, RouteDef>, K extends keyof R>(routes: R, name: K): R[K]['path'] {
  return routes[name]!.path;
}
