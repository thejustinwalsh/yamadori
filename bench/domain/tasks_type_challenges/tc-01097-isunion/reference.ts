type IsUnion<T, C = T> = [T] extends [never]
  ? false
  : T extends unknown
    ? [C] extends [T] ? false : true
    : never
