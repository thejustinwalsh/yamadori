type BXChars<S extends string> = S extends `${infer F}${infer R}` ? [F, ...BXChars<R>] : []
type BXLast<T extends unknown[]> = T extends [...unknown[], infer L extends string] ? L : '0'
type BXInit<T extends unknown[]> = T extends [...infer I, unknown] ? I : []

// XOR from the least significant bit; the shorter operand is padded with 0.
type BXXor<A extends unknown[], B extends unknown[], Out extends string = ''> =
  [A, B] extends [[], []]
    ? Out
    : BXXor<BXInit<A>, BXInit<B>, `${BXLast<A> extends BXLast<B> ? '0' : '1'}${Out}`>

type BitwiseXOR<S1 extends string, S2 extends string> = BXXor<BXChars<S1>, BXChars<S2>>
