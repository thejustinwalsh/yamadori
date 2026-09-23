type CheckRepeatedChars<T extends string> =
  T extends `${infer F}${infer R}`
    ? R extends `${string}${F}${string}`
      ? true
      : CheckRepeatedChars<R>
    : false
