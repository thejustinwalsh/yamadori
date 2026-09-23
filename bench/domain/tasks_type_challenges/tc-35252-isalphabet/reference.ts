type IsAlphabetChar<C extends string> = Uppercase<C> extends Lowercase<C> ? false : true

type IsAlphabet<S extends string> =
  S extends `${infer F}${infer R}`
    ? IsAlphabetChar<F> extends true
      ? R extends '' ? true : IsAlphabet<R>
      : false
    : false
