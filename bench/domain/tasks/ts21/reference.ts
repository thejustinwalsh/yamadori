export type Split<S extends string, D extends string> =
  string extends S ? string[]
  : string extends D ? string[]
  : S extends `${infer Head}${D}${infer Tail}` ? [Head, ...Split<Tail, D>]
  : [S];

export type Join<T extends readonly string[], D extends string> =
  number extends T['length'] ? string
  : string extends D ? string
  : T extends readonly [] ? ''
  : T extends readonly [infer Only extends string] ? Only
  : T extends readonly [infer First extends string, ...infer Rest extends string[]] ? `${First}${D}${Join<Rest, D>}`
  : string;

export function split<S extends string, D extends string>(s: S, delimiter: D): Split<S, D> {
  return s.split(delimiter) as Split<S, D>;
}

export function join<const T extends readonly string[], D extends string>(parts: T, delimiter: D): Join<T, D> {
  return parts.join(delimiter) as Join<T, D>;
}
