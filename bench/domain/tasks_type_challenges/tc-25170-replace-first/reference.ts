type ReplaceFirst<T extends readonly unknown[], S, R> =
  T extends readonly [infer F, ...infer Rest]
    ? [F] extends [S]
      ? [R, ...Rest]
      : [F, ...ReplaceFirst<Rest, S, R>]
    : []
