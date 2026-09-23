type AllStrictEqual<X, Y> =
  (<G>() => G extends X ? 1 : 2) extends (<G>() => G extends Y ? 1 : 2) ? true : false

type All<T extends readonly unknown[], U> =
  T extends readonly [infer F, ...infer R]
    ? AllStrictEqual<F, U> extends true ? All<R, U> : false
    : true
