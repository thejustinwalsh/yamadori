// Comparison of non-negative decimal numbers (any number of digits, optional fraction).
type SortDigitLess<A extends string, B extends string> =
  A extends B ? false : '0123456789' extends `${string}${A}${string}${B}${string}` ? true : false

type SortLengthCmp<A extends string, B extends string> =
  A extends `${string}${infer AR}`
    ? B extends `${string}${infer BR}` ? SortLengthCmp<AR, BR> : 'gt'
    : B extends '' ? 'eq' : 'lt'

// Digit-by-digit comparison; a missing digit counts as '0' (for fractions).
type SortDigitsCmp<A extends string, B extends string> =
  A extends '' ? (B extends '' ? 'eq' : SortDigitsCmp<'0', B>)
  : B extends '' ? SortDigitsCmp<A, '0'>
  : A extends `${infer a}${infer AR}`
    ? B extends `${infer b}${infer BR}`
      ? a extends b
        ? (AR extends '' ? (BR extends '' ? 'eq' : SortDigitsCmp<'', BR>) : SortDigitsCmp<AR, BR>)
        : SortDigitLess<a, b> extends true ? 'lt' : 'gt'
      : never
    : never

type SortSplit<S extends string> = S extends `${infer I}.${infer F}` ? [I, F] : [S, '']

type SortCmp<A extends number, B extends number> =
  [SortSplit<`${A}`>, SortSplit<`${B}`>] extends [[infer AI extends string, infer AF extends string], [infer BI extends string, infer BF extends string]]
    ? SortLengthCmp<AI, BI> extends 'eq'
      ? SortDigitsCmp<AI, BI> extends 'eq' ? SortDigitsCmp<AF, BF> : SortDigitsCmp<AI, BI>
      : SortLengthCmp<AI, BI>
    : never

// true when X must be placed before Y
type SortBefore<X extends number, Y extends number, Desc extends boolean> =
  Desc extends true
    ? (SortCmp<X, Y> extends 'gt' ? true : false)
    : (SortCmp<X, Y> extends 'lt' ? true : false)

type SortInsert<S extends number[], X extends number, Desc extends boolean, Done extends number[] = []> =
  S extends [infer H extends number, ...infer R extends number[]]
    ? SortBefore<X, H, Desc> extends true ? [...Done, X, ...S] : SortInsert<R, X, Desc, [...Done, H]>
    : [...Done, X]

type Sort<T extends number[], Desc extends boolean = false, Acc extends number[] = []> =
  T extends [infer H extends number, ...infer R extends number[]]
    ? Sort<R, Desc, SortInsert<Acc, H, Desc>>
    : Acc
