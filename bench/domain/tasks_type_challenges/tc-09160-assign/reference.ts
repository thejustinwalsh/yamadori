type AssignMerge<A, B> = {
  [K in keyof A | keyof B]: K extends keyof B ? B[K] : K extends keyof A ? A[K] : never
}

type AssignAll<T, U> =
  U extends [infer F, ...infer R]
    ? AssignAll<F extends object ? AssignMerge<T, F> : T, R>
    : T

type Assign<T extends Record<string, unknown>, U> = AssignAll<T, U>
