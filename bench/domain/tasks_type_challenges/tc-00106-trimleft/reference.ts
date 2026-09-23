type TrimLeftWhitespace = ' ' | '\n' | '\t'
type TrimLeft<S extends string> = S extends `${TrimLeftWhitespace}${infer R}` ? TrimLeft<R> : S
