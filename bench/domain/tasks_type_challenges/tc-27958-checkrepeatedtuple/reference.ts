type CheckRepeatedStrictEqual<X, Y> =
  (<G>() => G extends X ? 1 : 2) extends (<G>() => G extends Y ? 1 : 2) ? true : false

type CheckRepeatedIncludes<T extends unknown[], U> =
  T extends [infer F, ...infer R]
    ? CheckRepeatedStrictEqual<F, U> extends true ? true : CheckRepeatedIncludes<R, U>
    : false

type CheckRepeatedTuple<T extends unknown[]> =
  T extends [infer F, ...infer R]
    ? CheckRepeatedIncludes<R, F> extends true ? true : CheckRepeatedTuple<R>
    : false
