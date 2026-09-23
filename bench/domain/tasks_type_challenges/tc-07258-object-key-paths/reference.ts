type OKPJoin<P extends string, K extends string | number> = P extends '' ? `${K}` : `${P}.${K}`

// index keys of an array: `number` for arrays, the literal indices for tuples
type OKPIndex<T extends readonly unknown[]> =
  number extends T['length'] ? number : { [I in keyof T]: I }[number]

type OKPArray<T extends readonly unknown[], P extends string, I> =
  I extends string | number
    ? | OKPJoin<P, I>
      | `${P}[${I}]`
      | OKPJoin<P, `[${I}]`>
      | OKPaths<T[I & keyof T], OKPJoin<P, I>>
      | OKPaths<T[I & keyof T], `${P}[${I}]`>
    : never

type OKPaths<T, P extends string> =
  T extends readonly unknown[]
    ? OKPArray<T, P, OKPIndex<T>>
    : T extends object
      ? {
          [K in keyof T & (string | number)]-?: OKPJoin<P, K> | OKPaths<T[K], OKPJoin<P, K>>
        }[keyof T & (string | number)]
      : never

type ObjectKeyPaths<T extends object> = OKPaths<T, ''>
