type Join<T extends readonly string[], U extends string | number = ','> =
  T extends readonly [infer F extends string, ...infer R extends string[]]
    ? R extends []
      ? F
      : `${F}${U}${Join<R, U>}`
    : ''
