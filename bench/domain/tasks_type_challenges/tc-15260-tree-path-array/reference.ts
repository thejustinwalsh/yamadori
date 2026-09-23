type Path<T> =
  | []
  | (T extends object ? { [K in keyof T]-?: [K, ...Path<T[K]>] }[keyof T] : never)
