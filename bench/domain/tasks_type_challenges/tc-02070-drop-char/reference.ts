type DropChar<S extends string, C extends string> = C extends ''
  ? S
  : S extends `${infer L}${C}${infer R}`
    ? DropChar<`${L}${R}`, C>
    : S
