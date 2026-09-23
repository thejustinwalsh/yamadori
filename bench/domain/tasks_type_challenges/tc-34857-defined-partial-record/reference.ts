type DefinedPartialFlatten<O> = { [K in keyof O]: O[K] }

type DefinedPartial<T, K extends keyof T = keyof T> =
  K extends K
    ? DefinedPartialFlatten<T> | DefinedPartial<DefinedPartialFlatten<{ [P in keyof T as P extends K ? never : P]: T[P] }>>
    : never
