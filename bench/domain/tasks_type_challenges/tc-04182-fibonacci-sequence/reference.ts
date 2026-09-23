type Fibonacci<
  T extends number,
  I extends unknown[] = [unknown],
  Prev extends unknown[] = [],
  Cur extends unknown[] = [unknown],
> = I['length'] extends T
  ? Cur['length']
  : Fibonacci<T, [...I, unknown], Cur, [...Prev, ...Cur]>
