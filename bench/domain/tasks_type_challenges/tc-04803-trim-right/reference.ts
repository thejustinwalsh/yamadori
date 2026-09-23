type TrimRightSpace = ' ' | '\n' | '\t'

type TrimRight<S extends string> = S extends `${infer R}${TrimRightSpace}` ? TrimRight<R> : S
