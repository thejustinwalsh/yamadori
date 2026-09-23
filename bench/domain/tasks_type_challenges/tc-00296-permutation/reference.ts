type Permutation<T, All = T> =
  [T] extends [never]
    ? []
    : T extends unknown
      ? [T, ...Permutation<Exclude<All, T>>]
      : never
