type PartialByKeysMerge<O> = { [P in keyof O]: O[P] }

type PartialByKeys<T, K extends keyof T = keyof T> = PartialByKeysMerge<
  Omit<T, K> & Partial<Pick<T, K>>
>
