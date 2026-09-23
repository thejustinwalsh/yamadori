// Mirrors String.prototype.split: no separator gives [S], '' splits into characters.
type Split<S extends string, SEP extends string = never> =
  string extends S ? string[]
    : [SEP] extends [never] ? [S]
      : string extends SEP ? string[]
        : SEP extends ''
          ? S extends `${infer C}${infer Rest}` ? [C, ...Split<Rest, SEP>] : []
          : S extends `${infer Head}${SEP}${infer Rest}` ? [Head, ...Split<Rest, SEP>] : [S]
