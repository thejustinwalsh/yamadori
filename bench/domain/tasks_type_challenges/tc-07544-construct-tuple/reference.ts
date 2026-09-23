type ConstructTuple<L extends number, Acc extends unknown[] = []> =
  Acc['length'] extends L ? Acc : ConstructTuple<L, [...Acc, unknown]>
