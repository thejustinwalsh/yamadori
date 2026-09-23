// `string & {}` is a template-literal placeholder that matches exactly one
// character when followed by another placeholder, and -- unlike `${string}` --
// consecutive copies are not collapsed into `string`. So LSChunk<k> is a
// pattern that consumes exactly 10^k characters in one match, which lets a
// length of millions be counted in about a hundred steps instead of millions.
type LSAny = string & {}
type LSTimes10<P extends string> = `${P}${P}${P}${P}${P}${P}${P}${P}${P}${P}`
type LSC0 = `${LSAny}`
type LSC1 = LSTimes10<LSC0>
type LSC2 = LSTimes10<LSC1>
type LSC3 = LSTimes10<LSC2>
type LSC4 = LSTimes10<LSC3>
type LSChunks = [LSC0, LSC1, LSC2, LSC3, LSC4]
type LSPrev = [never, 0, 1, 2, 3]

// Below 10^5: strip as many 10^K chunks as fit (at most 9), then move to the
// next lower power, appending one decimal digit per power.
type LSCount<S extends string, K extends number, N extends unknown[], Digits extends string> =
  S extends `${LSChunks[K]}${infer Rest}`
    ? LSCount<Rest, K, [...N, unknown], Digits>
    : K extends 0
      ? `${Digits}${N['length']}`
      : LSCount<S, LSPrev[K], [], `${Digits}${N['length']}`>

// Peel 10^5 characters per step first (a larger single pattern measured slower).
type LSHundredThousands<S extends string, M extends unknown[] = []> =
  S extends `${LSC4}${LSC4}${LSC4}${LSC4}${LSC4}${LSC4}${LSC4}${LSC4}${LSC4}${LSC4}${infer Rest}`
    ? LSHundredThousands<Rest, [...M, unknown]>
    : [S, M['length']]

type LSStripZeros<D extends string> = D extends `0${infer R}` ? (R extends '' ? D : LSStripZeros<R>) : D
type LSToNumber<D extends string> = LSStripZeros<D> extends `${infer N extends number}` ? N : never

type LengthOfString<S extends string> =
  LSHundredThousands<S> extends [infer Rest extends string, infer M extends number]
    ? LSToNumber<`${M extends 0 ? '' : M}${LSCount<Rest, 4, [], ''>}`>
    : never
