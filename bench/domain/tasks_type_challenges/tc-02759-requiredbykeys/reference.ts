type RequiredByKeysMerge<O> = { [P in keyof O]: O[P] }

type RequiredByKeys<T, K extends keyof T = keyof T> = RequiredByKeysMerge<
  Omit<T, K> & Required<Pick<T, K>>
>
