type Format<T extends string> =
  T extends `${string}%${infer C}${infer Rest}`
    ? C extends 's'
      ? (s1: string) => Format<Rest>
      : C extends 'd'
        ? (d1: number) => Format<Rest>
        : Format<Rest>
    : string
