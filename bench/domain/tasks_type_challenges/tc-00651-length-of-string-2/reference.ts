// Consumes ten characters per step so strings far longer than the recursion limit still count.
type LengthOfStringCount<S extends string, Acc extends unknown[] = []> =
  S extends `${infer _0}${infer _1}${infer _2}${infer _3}${infer _4}${infer _5}${infer _6}${infer _7}${infer _8}${infer _9}${infer Rest}`
    ? LengthOfStringCount<Rest, [...Acc, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
    : S extends `${infer _}${infer Rest}`
      ? LengthOfStringCount<Rest, [...Acc, 0]>
      : Acc['length']

type LengthOfString<S extends string> = LengthOfStringCount<S>
