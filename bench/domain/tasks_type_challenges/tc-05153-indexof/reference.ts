type IndexOfStrictEqual<X, Y> =
  (<G>() => G extends X ? 1 : 2) extends (<G>() => G extends Y ? 1 : 2) ? true : false

type IndexOf<T extends readonly unknown[], U, Seen extends unknown[] = []> =
  T extends readonly [infer F, ...infer R]
    ? IndexOfStrictEqual<F, U> extends true
      ? Seen['length']
      : IndexOf<R, U, [...Seen, F]>
    : -1
