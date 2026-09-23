// Arrow-function exports, an interface for RouteDef, and a mapped-type constraint.
export type Method = 'GET' | 'POST' | 'PUT' | 'DELETE';
export interface RouteDef { path: `/${string}`; method: Method; tags?: readonly string[] }

export const defineRoutes = <const R extends { readonly [name: string]: RouteDef }>(routes: R): R => routes;

export const pathOf = <R extends { readonly [name: string]: RouteDef }, K extends keyof R & string>(
  routes: R,
  name: K,
): R[K]['path'] => routes[name]!.path;
