type Hanoi<N extends number, From = 'A', To = 'B', Intermediate = 'C', Depth extends unknown[] = []> =
  Depth['length'] extends N
    ? []
    : [
        ...Hanoi<N, From, Intermediate, To, [...Depth, unknown]>,
        [From, To],
        ...Hanoi<N, Intermediate, To, From, [...Depth, unknown]>,
      ]
