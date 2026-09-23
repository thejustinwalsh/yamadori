type JoinParts<P extends string[], D extends string> =
  P extends [] ? ''
    : P extends [infer Only extends string] ? Only
      : P extends [infer Head extends string, ...infer Rest extends string[]]
        ? `${Head}${D}${JoinParts<Rest, D>}`
        : string

declare function join<D extends string>(delimiter: D): <P extends string[]>(...parts: P) => JoinParts<P, D>
