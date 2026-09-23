type MinusOnePrevDigit = { '1': '0', '2': '1', '3': '2', '4': '3', '5': '4', '6': '5', '7': '6', '8': '7', '9': '8' }

type MinusOneReverse<S extends string> = S extends `${infer F}${infer R}` ? `${MinusOneReverse<R>}${F}` : ''

// Decrement a digit string written least-significant digit first.
type MinusOneDecReversed<S extends string> = S extends `${infer F}${infer R}`
  ? F extends '0'
    ? `9${MinusOneDecReversed<R>}`
    : `${MinusOnePrevDigit[F & keyof MinusOnePrevDigit]}${R}`
  : ''

type MinusOneTrimZeros<S extends string> = S extends `0${infer R}`
  ? R extends '' ? '0' : MinusOneTrimZeros<R>
  : S

type MinusOneToNumber<S extends string> = S extends `${infer N extends number}` ? N : never

type MinusOne<T extends number> = T extends 0
  ? -1
  : MinusOneToNumber<MinusOneTrimZeros<MinusOneReverse<MinusOneDecReversed<MinusOneReverse<`${T}`>>>>>
