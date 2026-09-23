type MulDigitTuple = {
  '0': [], '1': [0], '2': [0, 0], '3': [0, 0, 0], '4': [0, 0, 0, 0],
  '5': [0, 0, 0, 0, 0], '6': [0, 0, 0, 0, 0, 0], '7': [0, 0, 0, 0, 0, 0, 0],
  '8': [0, 0, 0, 0, 0, 0, 0, 0], '9': [0, 0, 0, 0, 0, 0, 0, 0, 0],
}
type MulDigit = keyof MulDigitTuple
type MulOnes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

type MulReverse<S extends string> = S extends `${infer H}${infer R}` ? `${MulReverse<R>}${H}` : ''

type MulAddDigits<A extends MulDigit, B extends MulDigit, C extends 0 | 1> =
  [...MulDigitTuple[A], ...MulDigitTuple[B], ...(C extends 1 ? [0] : [])]['length'] extends infer N extends number
    ? [MulOnes[N], N extends 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 ? 1 : 0]
    : never

// Addition of reversed digit strings (least significant digit first).
type MulAddRev<A extends string, B extends string, C extends 0 | 1 = 0> =
  A extends '' ? (B extends '' ? (C extends 1 ? '1' : '') : MulAddRev<'0', B, C>)
  : B extends '' ? MulAddRev<A, '0', C>
  : A extends `${infer a extends MulDigit}${infer AR}`
    ? B extends `${infer b extends MulDigit}${infer BR}`
      ? MulAddDigits<a, b, C> extends [infer D extends string, infer NC extends 0 | 1]
        ? `${D}${MulAddRev<AR, BR, NC>}`
        : never
      : never
    : never

// Reversed A times a single digit, by repeated addition.
type MulByDigit<A extends string, D extends MulDigit, C extends unknown[] = [], Acc extends string = '0'> =
  C['length'] extends MulDigitTuple[D]['length'] ? Acc : MulByDigit<A, D, [...C, 0], MulAddRev<Acc, A>>

// Long multiplication on reversed strings; Shift holds the zeros for the current place.
type MulRev<A extends string, B extends string, Shift extends string = '', Acc extends string = '0'> =
  B extends `${infer b extends MulDigit}${infer BR}`
    ? MulRev<A, BR, `${Shift}0`, MulAddRev<Acc, `${Shift}${MulByDigit<A, b>}`>>
    : Acc

type MulStripZeros<S extends string> =
  S extends `0${infer R}` ? (R extends '' ? '0' : MulStripZeros<R>) : S

type Multiply<A extends string | number | bigint, B extends string | number | bigint> =
  MulStripZeros<MulReverse<MulRev<MulReverse<`${A}`>, MulReverse<`${B}`>>>>
