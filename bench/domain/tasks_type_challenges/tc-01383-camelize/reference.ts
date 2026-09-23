type CamelizeKey<S> =
  S extends `${infer Head}_${infer Rest}` ? `${Head}${CamelizeKey<Capitalize<Rest>>}` : S

type Camelize<T> =
  T extends readonly unknown[]
    ? { [I in keyof T]: Camelize<T[I]> }
    : T extends object
      ? { [K in keyof T as CamelizeKey<K>]: Camelize<T[K]> }
      : T
