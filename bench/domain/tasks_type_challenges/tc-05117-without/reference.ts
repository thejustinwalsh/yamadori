type WithoutToUnion<U> = U extends readonly unknown[] ? U[number] : U

type Without<T, U> = T extends [infer F, ...infer R]
  ? F extends WithoutToUnion<U>
    ? Without<R, U>
    : [F, ...Without<R, U>]
  : []
