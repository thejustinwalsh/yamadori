// Iterates the string keys directly instead of filtering inside the `as` clause.
export type Getters<T> = {
  [K in Extract<keyof T, string> as `get${Capitalize<K>}`]: () => T[K];
};
