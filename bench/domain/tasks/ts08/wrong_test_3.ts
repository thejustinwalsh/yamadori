// Loose typing: any record of strings is accepted, so a missing parameter compiles.
export type PathParams<P extends string> = Record<string, string>;

export function buildPath<P extends string>(pattern: P, params: PathParams<P>): string {
  return pattern.split('/').map((s) => (s.startsWith(':') ? encodeURIComponent(params[s.slice(1)]!) : s)).join('/');
}
