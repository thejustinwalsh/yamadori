type IsRequiredKey<T, K extends keyof T> =
  Pick<T, K> extends Required<Pick<T, K>> ? true : false
