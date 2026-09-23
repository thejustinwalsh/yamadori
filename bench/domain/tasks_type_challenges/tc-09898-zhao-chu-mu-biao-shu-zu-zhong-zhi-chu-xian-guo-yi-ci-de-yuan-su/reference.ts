type FindElesStrictEqual<X, Y> =
  (<G>() => G extends X ? 1 : 2) extends (<G>() => G extends Y ? 1 : 2) ? true : false

type FindElesCount<T extends unknown[], U, N extends unknown[] = []> =
  T extends [infer F, ...infer R]
    ? FindElesCount<R, U, FindElesStrictEqual<F, U> extends true ? [...N, unknown] : N>
    : N['length']

type FindEles<T extends any[], S extends any[] = T, Acc extends any[] = []> =
  S extends [infer F, ...infer R]
    ? FindEles<T, R, FindElesCount<T, F> extends 1 ? [...Acc, F] : Acc>
    : Acc
