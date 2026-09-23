// Only recognises a parameter that is followed by another slash: a trailing `:id` is missed.
export type PathParams<P extends string> =
  P extends `${string}:${infer Name}/${infer Rest}` ? { [K in Name]: string } & PathParams<Rest> : {};

export function buildPath<P extends string>(pattern: P, params: PathParams<P>): string {
  const values = params as Record<string, string>;
  return pattern.split('/').map((s) => (s.startsWith(':') ? encodeURIComponent(values[s.slice(1)]!) : s)).join('/');
}
