// Uppercases the whole key (`getNAME`) instead of only its first letter.
export type Getters<T> = {
  [K in keyof T as `get${Uppercase<K & string>}`]: () => T[K];
};
