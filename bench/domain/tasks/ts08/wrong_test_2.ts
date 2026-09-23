// Substitutes values without URL-encoding them.
type ParamNames<P extends string> =
  P extends `${infer Head}/${infer Rest}` ? ParamNames<Head> | ParamNames<Rest>
  : P extends `:${infer Name}` ? Name : never;
export type PathParams<P extends string> = { [K in ParamNames<P>]: string };

export function buildPath<P extends string>(pattern: P, params: PathParams<P>): string {
  const values = params as Record<string, string>;
  return pattern.split('/').map((s) => (s.startsWith(':') ? values[s.slice(1)]! : s)).join('/');
}
