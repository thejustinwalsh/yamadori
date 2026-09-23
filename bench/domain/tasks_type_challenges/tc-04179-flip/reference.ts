type Flip<T> = {
  [K in keyof T as T[K] extends PropertyKey
    ? T[K]
    : T[K] extends boolean | bigint | null | undefined
      ? `${T[K]}`
      : never]: K
}
