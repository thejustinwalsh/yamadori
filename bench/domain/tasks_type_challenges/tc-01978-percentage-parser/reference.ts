type PercentageParserUnit<A extends string> = A extends `${infer N}%` ? [N, '%'] : [A, '']

type PercentageParser<A extends string> = A extends `${infer S extends '+' | '-'}${infer R}`
  ? [S, ...PercentageParserUnit<R>]
  : ['', ...PercentageParserUnit<A>]
