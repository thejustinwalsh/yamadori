type Triangular<N extends number, C extends unknown[] = [], Acc extends unknown[] = []> =
  C['length'] extends N
    ? Acc['length']
    : Triangular<N, [...C, unknown], [...Acc, ...C, unknown]>
