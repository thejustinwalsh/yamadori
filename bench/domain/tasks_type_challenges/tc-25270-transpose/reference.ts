type Transpose<M extends number[][], Row extends number[] = M['length'] extends 0 ? [] : M[0]> = {
  [X in keyof Row]: {
    [Y in keyof M]: X extends keyof M[Y] ? M[Y][X] : never
  }
}
