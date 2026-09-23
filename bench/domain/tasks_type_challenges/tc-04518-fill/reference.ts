type Fill<
  T extends unknown[],
  N,
  Start extends number = 0,
  End extends number = T['length'],
  Acc extends unknown[] = [],
  Filling extends boolean = false,
> = T extends [infer F, ...infer R]
  ? Acc['length'] extends End
    ? [...Acc, ...T]
    : (Acc['length'] extends Start ? true : Filling) extends true
      ? Fill<R, N, Start, End, [...Acc, N], true>
      : Fill<R, N, Start, End, [...Acc, F], false>
  : Acc
