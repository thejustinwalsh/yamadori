type LongestCommonPrefix<T extends string[], P extends string = ''> =
  T extends [infer F extends string, ...string[]]
    ? F extends `${P}${infer C}${string}`
      ? T extends `${P}${C}${string}`[]
        ? LongestCommonPrefix<T, `${P}${C}`>
        : P
      : P
    : P
