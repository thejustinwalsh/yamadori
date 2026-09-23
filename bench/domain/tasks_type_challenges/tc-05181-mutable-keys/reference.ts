type MKIsEqual<X, Y> =
  (<G>() => G extends X ? 1 : 2) extends (<G>() => G extends Y ? 1 : 2) ? true : false

type MutableKeys<T> = keyof {
  [K in keyof T as MKIsEqual<Pick<T, K>, Readonly<Pick<T, K>>> extends true ? never : K]: T[K]
}
