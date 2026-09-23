type LastIndexOfStrictEqual<X, Y> =
  (<G>() => G extends X ? 1 : 2) extends (<G>() => G extends Y ? 1 : 2) ? true : false

type LastIndexOf<T extends readonly unknown[], U> =
  T extends readonly [...infer R, infer L]
    ? LastIndexOfStrictEqual<L, U> extends true
      ? R['length']
      : LastIndexOf<R, U>
    : -1
