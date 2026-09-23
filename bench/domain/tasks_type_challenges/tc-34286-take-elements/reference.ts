type TakeFirst<N extends number, Arr extends readonly unknown[], Acc extends unknown[] = []> =
  Acc['length'] extends N
    ? Acc
    : Arr extends readonly [infer F, ...infer R] ? TakeFirst<N, R, [...Acc, F]> : Acc

type TakeLast<N extends number, Arr extends readonly unknown[], Acc extends unknown[] = []> =
  Acc['length'] extends N
    ? Acc
    : Arr extends readonly [...infer I, infer L] ? TakeLast<N, I, [L, ...Acc]> : Acc

type Take<N extends number, Arr extends readonly unknown[]> =
  `${N}` extends `-${infer M extends number}` ? TakeLast<M, Arr> : TakeFirst<N, Arr>
