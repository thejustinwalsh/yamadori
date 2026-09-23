type TrimWhitespace = ' ' | '\n' | '\t'
type Trim<S extends string> =
  S extends `${TrimWhitespace}${infer R}`
    ? Trim<R>
    : S extends `${infer L}${TrimWhitespace}`
      ? Trim<L>
      : S
