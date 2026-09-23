type DropStringChars<R extends string> =
  R extends `${infer C}${infer Rest}` ? C | DropStringChars<Rest> : never

type DropStringFrom<S extends string, Drop extends string, Acc extends string = ''> =
  S extends `${infer C}${infer Rest}`
    ? DropStringFrom<Rest, Drop, [C] extends [Drop] ? Acc : `${Acc}${C}`>
    : Acc

type DropString<S extends string, R extends string> = DropStringFrom<S, DropStringChars<R>>
