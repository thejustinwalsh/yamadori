type AllCombinationsChars<S extends string> = S extends `${infer F}${infer R}`
  ? F | AllCombinationsChars<R>
  : never

type AllCombinationsOf<U extends string, A extends string = U> = [U] extends [never]
  ? ''
  : '' | (A extends A ? `${A}${AllCombinationsOf<Exclude<U, A>>}` : never)

type AllCombinations<S extends string> = AllCombinationsOf<AllCombinationsChars<S>>
