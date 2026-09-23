type Integer<T extends number> =
  number extends T
    ? never
    : `${T}` extends `${string}.${string}` | `${string}e-${string}`
      ? never
      : T
