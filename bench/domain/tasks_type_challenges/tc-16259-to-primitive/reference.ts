type ToPrimitive<T> =
  T extends string ? string
  : T extends number ? number
  : T extends boolean ? boolean
  : T extends bigint ? bigint
  : T extends symbol ? symbol
  : T extends (...args: any[]) => unknown ? Function
  : T extends object ? { [K in keyof T]: ToPrimitive<T[K]> }
  : T
