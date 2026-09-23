type ParseUrlParams<T extends string> =
  T extends `${string}:${infer P}`
    ? P extends `${infer Name}/${infer Rest}`
      ? Name | ParseUrlParams<Rest>
      : P
    : never
