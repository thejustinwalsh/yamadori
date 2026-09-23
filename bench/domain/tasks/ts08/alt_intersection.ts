// Builds the params as an intersection of one-key records, and substitutes with a regex.
export type PathParams<P extends string> =
  P extends `${string}:${infer Name}/${infer Rest}`
    ? { [K in Name]: string } & PathParams<`/${Rest}`>
    : P extends `${string}:${infer Name}`
      ? { [K in Name]: string }
      : {};

export const buildPath = <P extends string>(pattern: P, params: PathParams<P>): string =>
  pattern.replace(/(?<=^|\/):([^/]+)/g, (_, name: string) =>
    encodeURIComponent((params as Record<string, string>)[name]!));
