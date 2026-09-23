type BinaryToDecimal<S extends string, Acc extends unknown[] = []> =
  S extends `${infer B}${infer R}`
    ? BinaryToDecimal<R, B extends '1' ? [...Acc, ...Acc, unknown] : [...Acc, ...Acc]>
    : Acc['length']
