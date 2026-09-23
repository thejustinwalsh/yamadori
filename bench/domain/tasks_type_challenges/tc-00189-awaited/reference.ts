type MyAwaited<T extends PromiseLike<any>> =
  T extends PromiseLike<infer V>
    ? V extends PromiseLike<any>
      ? MyAwaited<V>
      : V
    : never
