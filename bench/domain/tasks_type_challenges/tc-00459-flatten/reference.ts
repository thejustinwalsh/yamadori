type Flatten<T extends readonly unknown[]> =
  T extends readonly [infer F, ...infer R]
    ? F extends readonly unknown[]
      ? [...Flatten<F>, ...Flatten<R>]
      : [F, ...Flatten<R>]
    : []
