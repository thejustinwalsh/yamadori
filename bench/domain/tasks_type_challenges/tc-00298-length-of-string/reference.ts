type LengthOfString<S extends string, Acc extends unknown[] = []> =
  S extends `${infer _F}${infer R}` ? LengthOfString<R, [...Acc, unknown]> : Acc['length']
