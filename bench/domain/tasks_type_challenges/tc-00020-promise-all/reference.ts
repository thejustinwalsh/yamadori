type PromiseAllUnwrap<X> = X extends PromiseLike<infer V> ? PromiseAllUnwrap<V> : X

declare function PromiseAll<T extends readonly unknown[] | []>(
  values: T,
): Promise<{ -readonly [K in keyof T]: PromiseAllUnwrap<T[K]> }>
