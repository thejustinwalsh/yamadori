type MergeAllTwo<A, B> = {
  [K in keyof A | keyof B]:
    K extends keyof A
      ? K extends keyof B ? A[K] | B[K] : A[K]
      : K extends keyof B ? B[K] : never
}

type MergeAll<XS extends readonly object[], Acc = {}> =
  XS extends readonly [infer F, ...infer R extends object[]]
    ? MergeAll<R, MergeAllTwo<Acc, F>>
    : Acc
