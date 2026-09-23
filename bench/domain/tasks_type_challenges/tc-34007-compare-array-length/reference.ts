type CompareArrayLength<T extends any[], U extends any[]> =
  T extends [unknown, ...infer TR]
    ? U extends [unknown, ...infer UR]
      ? CompareArrayLength<TR, UR>
      : 1
    : U extends [unknown, ...unknown[]]
      ? -1
      : 0
