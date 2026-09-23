type IsOdd<T extends number> =
  number extends T
    ? false
    : `${T}` extends `${string}.${string}` | `${string}e${string}`
      ? false
      : `${T}` extends `${string}${1 | 3 | 5 | 7 | 9}`
        ? true
        : false
