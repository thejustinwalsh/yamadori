// Forgets to capitalize: produces `getname` instead of `getName`.
export type Getters<T> = {
  [K in keyof T as `get${K & string}`]: () => T[K];
};
