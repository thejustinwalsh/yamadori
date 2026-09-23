// Stringifies numeric keys instead of dropping them, so `0` becomes a `get0` getter.
export type Getters<T> = {
  [K in keyof T as K extends string | number ? `get${Capitalize<`${K}`>}` : never]: () => T[K];
};
