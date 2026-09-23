// A character is a word separator when it has no case (upper and lower forms agree).
type IsWordSeparator<C extends string> = Uppercase<C> extends Lowercase<C> ? true : false

type CapitalizeWordsFrom<S extends string, Prev extends string, Acc extends string = ''> =
  S extends `${infer C}${infer Rest}`
    ? CapitalizeWordsFrom<Rest, C, `${Acc}${IsWordSeparator<Prev> extends true ? Uppercase<C> : C}`>
    : Acc

type CapitalizeWords<S extends string> = CapitalizeWordsFrom<S, ' '>
