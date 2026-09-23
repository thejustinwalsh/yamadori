type PascalTuple<N extends number, A extends unknown[] = []> =
  A['length'] extends N ? A : PascalTuple<N, [...A, unknown]>

type PascalAdd<A extends number, B extends number> =
  [...PascalTuple<A>, ...PascalTuple<B>]['length'] & number

type PascalNextRow<Row extends number[], Acc extends number[] = [1]> =
  Row extends [infer A extends number, infer B extends number, ...infer R extends number[]]
    ? PascalNextRow<[B, ...R], [...Acc, PascalAdd<A, B>]>
    : [...Acc, 1]

type Pascal<N extends number, Rows extends number[][] = [[1]]> =
  N extends 0
    ? []
    : Rows['length'] extends N
      ? Rows
      : Rows extends [...unknown[], infer Last extends number[]]
        ? Pascal<N, [...Rows, PascalNextRow<Last>]>
        : never
