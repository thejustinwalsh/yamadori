type FlattenDepth<T extends readonly unknown[], D extends number = 1, C extends unknown[] = []> =
  C['length'] extends D
    ? T
    : T extends readonly [infer F, ...infer R]
      ? F extends readonly unknown[]
        ? [...FlattenDepth<F, D, [...C, unknown]>, ...FlattenDepth<R, D, C>]
        : [F, ...FlattenDepth<R, D, C>]
      : T
