type Enum<T extends readonly string[], N extends boolean = false> = {
  readonly [K in keyof T as K extends `${number}` ? Capitalize<T[K] & string> : never]:
    N extends true
      ? K extends `${infer I extends number}` ? I : never
      : T[K]
}
