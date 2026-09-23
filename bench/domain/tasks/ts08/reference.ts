type ParamNames<P extends string> =
  P extends `${infer Head}/${infer Rest}`
    ? ParamNames<Head> | ParamNames<Rest>
    : P extends `:${infer Name}` ? Name : never;

export type PathParams<P extends string> = { [K in ParamNames<P>]: string };

export function buildPath<P extends string>(pattern: P, params: PathParams<P>): string {
  const values = params as Record<string, string>;
  return pattern
    .split('/')
    .map((seg) => (seg.startsWith(':') ? encodeURIComponent(values[seg.slice(1)]!) : seg))
    .join('/');
}
