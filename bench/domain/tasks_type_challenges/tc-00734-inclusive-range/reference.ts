type IRGo<L extends number, H extends number, C extends unknown[], R extends number[], On extends boolean> =
  (On extends true ? true : C['length'] extends L ? true : false) extends infer O extends boolean
    ? C['length'] extends H
      ? (O extends true ? [...R, H] : R)
      : IRGo<L, H, [...C, 0], O extends true ? [...R, C['length']] : R, O>
    : never

type InclusiveRange<Lower extends number, Higher extends number> = IRGo<Lower, Higher, [], [], false>
