type GreaterThanDigits = '0123456789'

// 'gt' | 'eq' | 'lt' comparing the lengths of two digit strings.
type GreaterThanCompareLength<A extends string, B extends string> = A extends `${string}${infer RA}`
  ? B extends `${string}${infer RB}`
    ? GreaterThanCompareLength<RA, RB>
    : 'gt'
  : B extends '' ? 'eq' : 'lt'

type GreaterThanDigit<A extends string, B extends string> = A extends B
  ? false
  : GreaterThanDigits extends `${string}${B}${string}${A}${string}` ? true : false

// Same-length digit strings, compared most significant digit first.
type GreaterThanSameLength<A extends string, B extends string> = A extends `${infer FA}${infer RA}`
  ? B extends `${infer FB}${infer RB}`
    ? FA extends FB
      ? GreaterThanSameLength<RA, RB>
      : GreaterThanDigit<FA, FB>
    : false
  : false

type GreaterThanCompare<A extends string, B extends string> =
  GreaterThanCompareLength<A, B> extends infer L
    ? L extends 'gt' ? true : L extends 'lt' ? false : GreaterThanSameLength<A, B>
    : never

type GreaterThan<T extends number, U extends number> = GreaterThanCompare<`${T}`, `${U}`>
