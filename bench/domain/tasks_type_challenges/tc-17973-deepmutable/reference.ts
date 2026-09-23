type DeepMutable<T extends object> = {
  -readonly [K in keyof T]: T[K] extends (...args: any[]) => unknown
    ? T[K]
    : T[K] extends object
      ? DeepMutable<T[K]>
      : T[K]
}
