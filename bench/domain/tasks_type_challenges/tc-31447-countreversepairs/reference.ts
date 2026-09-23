// Integer comparison (negative numbers and any number of digits).
type CRPDigitLess<A extends string, B extends string> =
  A extends B ? false : '0123456789' extends `${string}${A}${string}${B}${string}` ? true : false

type CRPLengthCmp<A extends string, B extends string> =
  A extends `${string}${infer AR}`
    ? B extends `${string}${infer BR}` ? CRPLengthCmp<AR, BR> : 'gt'
    : B extends '' ? 'eq' : 'lt'

type CRPDigitsCmp<A extends string, B extends string> =
  A extends `${infer a}${infer AR}`
    ? B extends `${infer b}${infer BR}`
      ? a extends b ? CRPDigitsCmp<AR, BR> : CRPDigitLess<a, b> extends true ? 'lt' : 'gt'
      : 'eq'
    : 'eq'

type CRPUnsignedCmp<A extends string, B extends string> =
  CRPLengthCmp<A, B> extends 'eq' ? CRPDigitsCmp<A, B> : CRPLengthCmp<A, B>

type CRPGreater<A extends number, B extends number> =
  `${A}` extends `-${infer a}`
    ? `${B}` extends `-${infer b}` ? (CRPUnsignedCmp<b, a> extends 'gt' ? true : false) : false
    : `${B}` extends `-${string}` ? true : (CRPUnsignedCmp<`${A}`, `${B}`> extends 'gt' ? true : false)

// Adds one element to Acc for every later element smaller than X.
type CRPCountFor<X extends number, Rest extends number[], Acc extends unknown[]> =
  Rest extends [infer H extends number, ...infer R extends number[]]
    ? CRPCountFor<X, R, CRPGreater<X, H> extends true ? [...Acc, unknown] : Acc>
    : Acc

type CRPGo<T extends number[], Acc extends unknown[] = []> =
  T extends [infer H extends number, ...infer R extends number[]]
    ? CRPGo<R, CRPCountFor<H, R, Acc>>
    : Acc['length']

type CountReversePairs<T extends number[]> = CRPGo<T>
