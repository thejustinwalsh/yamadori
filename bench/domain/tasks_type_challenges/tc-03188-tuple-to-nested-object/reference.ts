type TupleToNestedObject<T extends readonly unknown[], U> = T extends readonly [infer F extends PropertyKey, ...infer R]
  ? { [K in F]: TupleToNestedObject<R, U> }
  : U
