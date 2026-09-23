type SumDigitTuple = {
  '0': [], '1': [0], '2': [0, 0], '3': [0, 0, 0], '4': [0, 0, 0, 0],
  '5': [0, 0, 0, 0, 0], '6': [0, 0, 0, 0, 0, 0], '7': [0, 0, 0, 0, 0, 0, 0],
  '8': [0, 0, 0, 0, 0, 0, 0, 0], '9': [0, 0, 0, 0, 0, 0, 0, 0, 0],
}
type SumDigit = keyof SumDigitTuple
// digit of (a + b + carry), indexed by the raw sum 0..19
type SumOnes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

type SumReverse<S extends string> = S extends `${infer H}${infer R}` ? `${SumReverse<R>}${H}` : ''

type SumAddDigits<A extends SumDigit, B extends SumDigit, C extends 0 | 1> =
  [...SumDigitTuple[A], ...SumDigitTuple[B], ...(C extends 1 ? [0] : [])]['length'] extends infer N extends number
    ? [SumOnes[N], N extends 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 ? 1 : 0]
    : never

// A and B are reversed (least significant digit first); so is the result.
type SumReversed<A extends string, B extends string, C extends 0 | 1 = 0> =
  A extends '' ? (B extends '' ? (C extends 1 ? '1' : '') : SumReversed<'0', B, C>)
  : B extends '' ? SumReversed<A, '0', C>
  : A extends `${infer a extends SumDigit}${infer AR}`
    ? B extends `${infer b extends SumDigit}${infer BR}`
      ? SumAddDigits<a, b, C> extends [infer D extends string, infer NC extends 0 | 1]
        ? `${D}${SumReversed<AR, BR, NC>}`
        : never
      : never
    : never

type Sum<A extends string | number | bigint, B extends string | number | bigint> =
  SumReverse<SumReversed<SumReverse<`${A}`>, SumReverse<`${B}`>>>
