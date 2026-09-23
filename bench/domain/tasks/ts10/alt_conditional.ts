// Descends through a helper that decides per property whether to recurse; reduce-based get.
type IsLeaf<V> = V extends object
  ? V extends Date | Function | readonly unknown[] ? true : false
  : true;

type PathsOf<T> = {
  [K in Extract<keyof T, string>]: IsLeaf<T[K]> extends true ? K : K | `${K}.${PathsOf<T[K]>}`;
}[Extract<keyof T, string>];

export type Paths<T> = PathsOf<T>;

export type PathValue<T, P extends string> =
  P extends keyof T ? T[P]
  : P extends `${infer Head extends keyof T & string}.${infer Tail}` ? PathValue<T[Head], Tail>
  : never;

export const get = <T, P extends Paths<T>>(obj: T, path: P): PathValue<T, P> =>
  path.split('.').reduce<any>((o, k) => o[k], obj);
