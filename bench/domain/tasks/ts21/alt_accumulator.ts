// A tail-recursive accumulator Split, a [...T] variadic parameter instead of `const`,
// and hand-written loops at runtime.
type IsWide<X extends string> = string extends X ? true : false;

type SplitAcc<S extends string, D extends string, Acc extends string[] = []> =
  S extends `${infer Head}${D}${infer Tail}` ? SplitAcc<Tail, D, [...Acc, Head]> : [...Acc, S];

export type Split<S extends string, D extends string> =
  IsWide<S> extends true ? string[] : IsWide<D> extends true ? string[] : SplitAcc<S, D>;

type JoinFixed<T extends readonly string[], D extends string> =
  T extends readonly [infer H extends string, ...infer R extends readonly string[]]
    ? R extends readonly [] ? H : `${H}${D}${JoinFixed<R, D>}`
    : '';

export type Join<T extends readonly string[], D extends string> =
  number extends T['length'] ? string : IsWide<D> extends true ? string : JoinFixed<T, D>;

export function split<S extends string, D extends string>(s: S, d: D): Split<S, D> {
  const out: string[] = [];
  let from = 0;
  for (let at = s.indexOf(d, from); at !== -1; at = s.indexOf(d, from)) {
    out.push(s.slice(from, at));
    from = at + d.length;
  }
  out.push(s.slice(from));
  return out as Split<S, D>;
}

export function join<T extends string[], D extends string>(parts: [...T], d: D): Join<T, D> {
  let s = '';
  parts.forEach((p, i) => { s += (i ? d : '') + p; });
  return s as Join<T, D>;
}
