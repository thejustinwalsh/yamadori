type CountElementFlatten<T extends unknown[]> =
  T extends [infer F, ...infer R]
    ? [F] extends [never]
      ? CountElementFlatten<R>
      : F extends unknown[]
        ? [...CountElementFlatten<F>, ...CountElementFlatten<R>]
        : [F, ...CountElementFlatten<R>]
    : []

type CountElementOccurrences<T extends unknown[], K, N extends unknown[] = []> =
  T extends [infer F, ...infer R]
    ? CountElementOccurrences<R, K, [F] extends [K] ? [...N, unknown] : N>
    : N['length']

type CountElementNumberToObject<T extends unknown[], L extends unknown[] = CountElementFlatten<T>> = {
  [K in L[number] & PropertyKey]: CountElementOccurrences<L, K>
}
