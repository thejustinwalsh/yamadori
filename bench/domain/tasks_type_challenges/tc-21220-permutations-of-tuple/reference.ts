type PermutationsOfTuple<T extends unknown[], Prev extends unknown[] = []> =
  T extends [infer F, ...infer R]
    ? [F, ...PermutationsOfTuple<[...Prev, ...R]>] | PermutationsOfTuple<R, [...Prev, F]>
    : Prev extends [] ? [] : never
