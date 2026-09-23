type NumberRange<
  L extends number,
  H extends number,
  C extends unknown[] = [],
  Acc = never,
  On extends boolean = false,
> =
  C['length'] extends H
    ? Acc | H
    : C['length'] extends L
      ? NumberRange<L, H, [...C, unknown], Acc | L, true>
      : NumberRange<L, H, [...C, unknown], On extends true ? Acc | C['length'] : Acc, On>
