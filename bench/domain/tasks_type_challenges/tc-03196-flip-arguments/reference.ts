type FlipArgumentsReverse<T extends readonly unknown[]> = T extends readonly [infer F, ...infer R]
  ? [...FlipArgumentsReverse<R>, F]
  : []

type FlipArguments<T extends (...args: any[]) => unknown> = T extends (...args: infer A) => infer R
  ? (...args: FlipArgumentsReverse<A>) => R
  : never
