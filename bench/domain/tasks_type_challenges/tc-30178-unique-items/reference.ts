type UIDuplicateError<V> = { error: 'duplicate item in uniqueItems'; item: V }

type UISame<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false

// true if some member of the union Seen is the same type as V
type UIIn<V, Seen> = true extends (Seen extends unknown ? UISame<V, Seen> : never) ? true : false

// union of the elements of T before index I
type UIBefore<T extends readonly unknown[], I, Acc extends unknown[] = []> =
  `${Acc['length']}` extends I
    ? Acc[number]
    : T extends readonly [infer F, ...infer R] ? UIBefore<R, I, [...Acc, F]> : Acc[number]

// each element that repeats an earlier one is replaced by an error type
type UIChecked<T extends readonly unknown[]> = {
  [I in keyof T]: UIIn<T[I], UIBefore<T, I>> extends true ? UIDuplicateError<T[I]> : T[I]
}

function uniqueItems<const T extends readonly unknown[]>(items: UIChecked<T>) {
  return items
}
