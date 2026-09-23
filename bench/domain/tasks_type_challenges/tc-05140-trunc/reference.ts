type Trunc<T extends number | string> =
  `${T}` extends `${infer I}.${string}`
    ? I extends '' ? '0' : I extends '-' ? '-0' : I
    : `${T}`
