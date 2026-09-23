// Decimal arithmetic on digit strings stored least-significant digit first.
type SquareTuple<D extends string, A extends unknown[] = []> =
  `${A['length']}` extends D ? A : SquareTuple<D, [...A, unknown]>

type SquareDigitSum<A extends string, B extends string, C extends 0 | 1> =
  `${[...SquareTuple<A>, ...SquareTuple<B>, ...(C extends 1 ? [unknown] : [])]['length'] & number}`

type SquareStep<A extends string, B extends string, C extends 0 | 1, AR extends string, BR extends string> =
  SquareDigitSum<A, B, C> extends `1${infer D extends string}`
    ? D extends ''
      ? `1${SquareAddReversed<AR, BR, 0>}`
      : `${D}${SquareAddReversed<AR, BR, 1>}`
    : `${SquareDigitSum<A, B, C>}${SquareAddReversed<AR, BR, 0>}`

type SquareAddReversed<A extends string, B extends string, C extends 0 | 1 = 0> =
  A extends `${infer HA}${infer AR}`
    ? B extends `${infer HB}${infer BR}`
      ? SquareStep<HA, HB, C, AR, BR>
      : SquareStep<HA, '0', C, AR, ''>
    : B extends `${infer HB}${infer BR}`
      ? SquareStep<'0', HB, C, '', BR>
      : C extends 1 ? '1' : ''

type SquareTimesDigit<A extends string, D extends string, Count extends unknown[] = [], Acc extends string = '0'> =
  `${Count['length']}` extends D
    ? Acc
    : SquareTimesDigit<A, D, [...Count, unknown], SquareAddReversed<Acc, A>>

type SquareMultiplyReversed<A extends string, B extends string, Shift extends string = '', Acc extends string = '0'> =
  B extends `${infer D}${infer BR}`
    ? SquareMultiplyReversed<A, BR, `0${Shift}`, SquareAddReversed<Acc, `${Shift}${SquareTimesDigit<A, D>}`>>
    : Acc

type SquareReverse<S extends string, Acc extends string = ''> =
  S extends `${infer F}${infer R}` ? SquareReverse<R, `${F}${Acc}`> : Acc

type SquareTrimZeros<S extends string> =
  S extends `0${infer R}` ? (R extends '' ? '0' : SquareTrimZeros<R>) : S

type SquareAbs<N extends number> = `${N}` extends `-${infer A}` ? A : `${N}`

type Square<N extends number> =
  SquareTrimZeros<
    SquareReverse<SquareMultiplyReversed<SquareReverse<SquareAbs<N>>, SquareReverse<SquareAbs<N>>>>
  > extends `${infer R extends number}` ? R : never
