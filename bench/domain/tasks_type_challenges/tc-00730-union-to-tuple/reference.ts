type UnionToTupleIntersection<U> =
  (U extends unknown ? (arg: U) => void : never) extends (arg: infer I) => void ? I : never

// The last member of a union: an intersection of functions resolves overloads to its last signature.
type UnionToTupleLast<U> =
  UnionToTupleIntersection<U extends unknown ? () => U : never> extends () => infer L ? L : never

type UnionToTuple<T, Last = UnionToTupleLast<T>> =
  [T] extends [never] ? [] : [...UnionToTuple<Exclude<T, Last>>, Last]
