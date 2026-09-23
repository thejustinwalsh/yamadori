// Annotates the return as the wide Record type, throwing away every literal.
export type Method = 'GET' | 'POST' | 'PUT' | 'DELETE';
export type RouteDef = { path: `/${string}`; method: Method; tags?: readonly string[] };
export function defineRoutes(routes: Record<string, RouteDef>): Record<string, RouteDef> {
  return routes;
}
export function pathOf(routes: Record<string, RouteDef>, name: string): `/${string}` {
  return routes[name]!.path;
}
