type CapitalizeNestObjectKeys<T> =
  T extends (...args: never[]) => unknown
    ? T
    : T extends readonly unknown[]
      ? { [K in keyof T]: CapitalizeNestObjectKeys<T[K]> }
      : T extends object
        ? { [K in keyof T as K extends string ? Capitalize<K> : K]: CapitalizeNestObjectKeys<T[K]> }
        : T
