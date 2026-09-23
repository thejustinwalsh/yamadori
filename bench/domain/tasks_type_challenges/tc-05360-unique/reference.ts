type UniqueStrictEqual<X, Y> =
  (<G>() => G extends X ? 1 : 2) extends (<G>() => G extends Y ? 1 : 2) ? true : false

type UniqueIncludes<T extends readonly unknown[], U> =
  T extends readonly [infer F, ...infer R]
    ? UniqueStrictEqual<F, U> extends true ? true : UniqueIncludes<R, U>
    : false

type Unique<T extends readonly unknown[], Acc extends unknown[] = []> =
  T extends readonly [infer F, ...infer R]
    ? UniqueIncludes<Acc, F> extends true
      ? Unique<R, Acc>
      : Unique<R, [...Acc, F]>
    : Acc
