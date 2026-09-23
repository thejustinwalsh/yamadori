type INNIsUnion<T, U = T> = T extends unknown ? ([U] extends [T] ? false : true) : never

type IsNegativeNumber<T extends number> =
  number extends T
    ? never
    : INNIsUnion<T> extends false
      ? `${T}` extends `-${string}` ? true : false
      : never
