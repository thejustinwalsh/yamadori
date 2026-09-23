type Chunk<T extends readonly unknown[], N extends number, Cur extends unknown[] = []> =
  T extends readonly [infer F, ...infer R]
    ? Cur['length'] extends N
      ? [Cur, ...Chunk<T, N>]
      : Chunk<R, N, [...Cur, F]>
    : Cur extends [] ? [] : [Cur]
