type DUToIntersection<U> =
  (U extends unknown ? (x: U) => void : never) extends (x: infer I) => void ? I : never
type DULast<U> = DUToIntersection<U extends unknown ? () => U : never> extends () => infer L ? L : never
type DUMerge<O> = { [P in keyof O]: O[P] }

type DUTuple<T extends readonly unknown[]> =
  T extends readonly [infer H, ...infer R]
    ? DistributeUnions<H> extends infer DH
      ? DH extends unknown
        ? DUTuple<R> extends infer DR extends unknown[]
          ? DR extends unknown ? [DH, ...DR] : never
          : never
        : never
      : never
    : T extends readonly [] ? [] : T

// Cartesian product over the properties, one key at a time.
type DUObject<T, Keys extends keyof T = keyof T> =
  [Keys] extends [never] ? {}
  : DULast<Keys> extends infer K extends keyof T
    ? DistributeUnions<T[K]> extends infer V
      ? V extends unknown
        ? DUObject<T, Exclude<Keys, K>> extends infer Rest
          ? Rest extends unknown ? DUMerge<{ [P in keyof Pick<T, K>]: V } & Rest> : never
          : never
        : never
      : never
    : never

type DistributeUnions<T> =
  T extends readonly unknown[] ? DUTuple<T>
  : T extends (...args: never[]) => unknown ? T
  : T extends object ? DUObject<T>
  : T
