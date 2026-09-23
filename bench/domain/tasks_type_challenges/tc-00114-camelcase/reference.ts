type IsCasedLetter<C extends string> = Uppercase<C> extends Lowercase<C> ? false : true

type CamelCaseLower<S extends string, Acc extends string = ''> =
  S extends `_${infer C}${infer Rest}`
    ? IsCasedLetter<C> extends true
      ? CamelCaseLower<Rest, `${Acc}${Uppercase<C>}`>
      : CamelCaseLower<`${C}${Rest}`, `${Acc}_`>
    : S extends `${infer C}${infer Rest}`
      ? CamelCaseLower<Rest, `${Acc}${C}`>
      : Acc

type CamelCase<S extends string> = CamelCaseLower<Lowercase<S>>
