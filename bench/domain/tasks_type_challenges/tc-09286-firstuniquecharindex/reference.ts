type FirstUniqueCharCount<S extends string, C extends string, N extends unknown[] = []> =
  S extends `${string}${C}${infer R}`
    ? FirstUniqueCharCount<R, C, [...N, unknown]>
    : N['length']

type FirstUniqueCharIndex<T extends string, S extends string = T, I extends unknown[] = []> =
  S extends `${infer F}${infer R}`
    ? FirstUniqueCharCount<T, F> extends 1
      ? I['length']
      : FirstUniqueCharIndex<T, R, [...I, unknown]>
    : -1
