enum Comparison {
  Greater,
  Equal,
  Lower,
}

type CmpDigit<A extends string, B extends string> =
  A extends B ? Comparison.Equal
  : '0123456789' extends `${string}${A}${string}${B}${string}` ? Comparison.Lower
  : Comparison.Greater

// Compare by length only.
type CmpLength<A extends string, B extends string> =
  A extends `${string}${infer AR}`
    ? B extends `${string}${infer BR}` ? CmpLength<AR, BR> : Comparison.Greater
    : B extends '' ? Comparison.Equal : Comparison.Lower

// Lexicographic digit comparison; a missing digit counts as '0' (fractions).
type CmpDigits<A extends string, B extends string> =
  A extends '' ? (B extends '' ? Comparison.Equal : CmpDigits<'0', B>)
  : B extends '' ? CmpDigits<A, '0'>
  : A extends `${infer a}${infer AR}`
    ? B extends `${infer b}${infer BR}`
      ? CmpDigit<a, b> extends Comparison.Equal
        ? (AR extends '' ? (BR extends '' ? Comparison.Equal : CmpDigits<'', BR>) : CmpDigits<AR, BR>)
        : CmpDigit<a, b>
      : never
    : never

type CmpSplit<S extends string> = S extends `${infer I}.${infer F}` ? [I, F] : [S, '']

type CmpUnsigned<A extends string, B extends string> =
  [CmpSplit<A>, CmpSplit<B>] extends [[infer AI extends string, infer AF extends string], [infer BI extends string, infer BF extends string]]
    ? CmpLength<AI, BI> extends Comparison.Equal
      ? CmpDigits<AI, BI> extends Comparison.Equal ? CmpDigits<AF, BF> : CmpDigits<AI, BI>
      : CmpLength<AI, BI>
    : never

type Comparator<A extends number, B extends number> =
  `${A}` extends `-${infer a}`
    ? `${B}` extends `-${infer b}` ? CmpUnsigned<b, a> : Comparison.Lower
    : `${B}` extends `-${string}` ? Comparison.Greater : CmpUnsigned<`${A}`, `${B}`>
