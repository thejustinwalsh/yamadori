export type Method = 'GET' | 'POST' | 'PUT' | 'DELETE';
export type RouteDef = { path: `/${string}`; method: Method; tags?: readonly string[] };

export function defineRoutes<const R extends Record<string, RouteDef>>(routes: R): R {
  return routes;
}

export function pathOf<R extends Record<string, RouteDef>, K extends keyof R>(routes: R, name: K): R[K]['path'] {
  return routes[name]!.path;
}
