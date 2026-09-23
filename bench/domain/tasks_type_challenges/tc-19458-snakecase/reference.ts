type SnakeCase<T> =
  T extends `${infer F}${infer R}`
    ? `${F extends Lowercase<F> ? F : `_${Lowercase<F>}`}${SnakeCase<R>}`
    : T
