type ExtractToObjectFlatten<O> = { [K in keyof O]: O[K] }

type ExtractToObject<T, U extends keyof T> =
  ExtractToObjectFlatten<{ [K in keyof T as K extends U ? never : K]: T[K] } & T[U]>
