type UBFunction = (...args: never[]) => unknown

type UBIsBoxed<T> =
  T extends UBFunction ? true : T extends readonly unknown[] ? true : T extends Promise<unknown> ? true : false

type UBOnce<T> =
  T extends (...args: never[]) => infer R
    ? R
    : T extends readonly (infer E)[]
      ? E
      : T extends Promise<infer P>
        ? P
        : T

// N = 0 unboxes fully; otherwise at most N levels.
type Unbox<T, N extends number = 0, C extends unknown[] = []> =
  true extends UBIsBoxed<T>
    ? [N] extends [0]
      ? Unbox<UBOnce<T>, N, C>
      : C['length'] extends N
        ? T
        : Unbox<UBOnce<T>, N, [...C, unknown]>
    : T
