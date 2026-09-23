type Pure<T> = {
  [P in keyof T]: T[P] extends object ? Pure<T[P]> : T[P]
}

type SetProperty<T, K extends PropertyKey, V> = {
  [P in (keyof T) | K]: P extends K ? V : P extends keyof T ? T[P] : never
}

// Punctuation, the three keywords, and string tokens (a string is carried as [value]).
type Token = '{' | '}' | '[' | ']' | ':' | ',' | true | false | null | [string]
type ParseResult<T, K extends Token[]> = [T, K]

type JsonWhitespace = ' ' | '\n' | '\r' | '\t'
type JsonPunctuation = '{' | '}' | '[' | ']' | ':' | ','
type JsonEscapes = {
  '"': '"', '\\': '\\', '/': '/', b: '\b', f: '\f', n: '\n', r: '\r', t: '\t'
}

// Body of a string literal after the opening quote: [value, rest of input] or never.
type JsonString<T extends string, Acc extends string = ''> =
  T extends `"${infer Rest}` ? [Acc, Rest]
  : T extends `\\${infer E}${infer Rest}`
    ? E extends keyof JsonEscapes ? JsonString<Rest, `${Acc}${JsonEscapes[E]}`> : never
  : T extends `${infer C}${infer Rest}`
    ? C extends '\n' | '\r' ? never : JsonString<Rest, `${Acc}${C}`>
  : never

type Tokenize<T extends string, S extends Token[] = []> =
  T extends '' ? S
  : T extends `${JsonWhitespace}${infer Rest}` ? Tokenize<Rest, S>
  : T extends `${infer C extends JsonPunctuation}${infer Rest}` ? Tokenize<Rest, [...S, C]>
  : T extends `true${infer Rest}` ? Tokenize<Rest, [...S, true]>
  : T extends `false${infer Rest}` ? Tokenize<Rest, [...S, false]>
  : T extends `null${infer Rest}` ? Tokenize<Rest, [...S, null]>
  : T extends `"${infer Rest}`
    ? JsonString<Rest> extends [infer V extends string, infer R extends string] ? Tokenize<R, [...S, [V]]> : never
  : never

type JsonValue<T extends Token[]> =
  T extends [infer H, ...infer R extends Token[]]
    ? H extends '{' ? (R extends ['}', ...infer R2 extends Token[]] ? ParseResult<{}, R2> : JsonMembers<R, {}>)
    : H extends '[' ? (R extends [']', ...infer R2 extends Token[]] ? ParseResult<[], R2> : JsonElements<R, []>)
    : H extends [infer S extends string] ? ParseResult<S, R>
    : H extends true | false | null ? ParseResult<H, R>
    : never
  : never

type JsonMembers<T extends Token[], Acc> =
  T extends [[infer K extends string], ':', ...infer R extends Token[]]
    ? JsonValue<R> extends infer P
      ? [P] extends [never] ? never
      : P extends [infer V, infer R2 extends Token[]]
        ? R2 extends [',', ...infer R3 extends Token[]] ? JsonMembers<R3, SetProperty<Acc, K, V>>
        : R2 extends ['}', ...infer R3 extends Token[]] ? ParseResult<SetProperty<Acc, K, V>, R3>
        : never
      : never
    : never
  : never

type JsonElements<T extends Token[], Acc extends unknown[]> =
  JsonValue<T> extends infer P
    ? [P] extends [never] ? never
    : P extends [infer V, infer R extends Token[]]
      ? R extends [',', ...infer R2 extends Token[]] ? JsonElements<R2, [...Acc, V]>
      : R extends [']', ...infer R2 extends Token[]] ? ParseResult<[...Acc, V], R2>
      : never
    : never
  : never

// A whole document: one value and no tokens left over.
type ParseLiteral<T extends Token[]> =
  JsonValue<T> extends infer P
    ? [P] extends [never] ? never
    : P extends [infer V, []] ? ParseResult<V, []> : never
  : never

type Parse<T extends string> = Pure<ParseLiteral<Tokenize<T>>[0]>
