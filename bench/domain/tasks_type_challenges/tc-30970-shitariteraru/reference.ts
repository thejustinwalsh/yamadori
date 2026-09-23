type FixedLiteralIsUnion<T, Whole = T> =
  T extends T ? ([Whole] extends [T] ? false : true) : never

type IsFixedStringLiteralType<S extends string> =
  [S] extends [never]
    ? false
    : FixedLiteralIsUnion<S> extends true
      ? false
      : {} extends { [K in S]: 1 }
        ? false
        : true
