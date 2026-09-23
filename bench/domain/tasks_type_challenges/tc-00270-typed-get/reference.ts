type Get<T, K> =
  K extends keyof T
    ? T[K]
    : K extends `${infer Head}.${infer Rest}`
      ? Head extends keyof T
        ? Get<T[Head], Rest>
        : never
      : never
