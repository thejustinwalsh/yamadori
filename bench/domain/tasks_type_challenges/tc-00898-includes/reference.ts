type IncludesIsEqual<X, Y> =
  (<G>() => G extends X ? 1 : 2) extends (<G>() => G extends Y ? 1 : 2) ? true : false

type Includes<T extends readonly any[], U> =
  T extends readonly [infer F, ...infer R]
    ? IncludesIsEqual<F, U> extends true
      ? true
      : Includes<R, U>
    : false
