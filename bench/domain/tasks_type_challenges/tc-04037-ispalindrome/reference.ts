type IsPalindrome<T extends string | number> =
  `${T}` extends `${infer F}${infer R}`
    ? R extends ''
      ? true
      : `${T}` extends `${F}${infer M}${F}`
        ? IsPalindrome<M>
        : false
    : true
