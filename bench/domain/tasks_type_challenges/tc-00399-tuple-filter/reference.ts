type FilterOut<T extends any[], F> =
  T extends [infer Head, ...infer Rest]
    ? [Head] extends [F]
      ? FilterOut<Rest, F>
      : [Head, ...FilterOut<Rest, F>]
    : []
