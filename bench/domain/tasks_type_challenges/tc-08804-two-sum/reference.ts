type TSTuple<N extends number, R extends unknown[] = []> =
  R['length'] extends N ? R : TSTuple<N, [...R, unknown]>

// A + each member of B, as a union of sums
type TSSums<A extends number, B extends number> =
  B extends number ? [...TSTuple<A>, ...TSTuple<B>]['length'] : never

type TwoSum<T extends number[], U extends number> =
  T extends [infer F extends number, ...infer R extends number[]]
    ? U extends TSSums<F, R[number]>
      ? true
      : TwoSum<R, U>
    : false
