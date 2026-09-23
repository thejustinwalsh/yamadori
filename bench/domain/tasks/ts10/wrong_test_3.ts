// Right paths, but `get` returns unknown instead of the value type at the path.
type Leaf = string | number | boolean | bigint | symbol | null | undefined | Date | readonly unknown[] | ((...args: never[]) => unknown);
export type Paths<T> = T extends Leaf
  ? never
  : { [K in keyof T & string]: K | `${K}.${Paths<T[K]>}` }[keyof T & string];
export type PathValue<T, P extends string> = unknown;
export function get<T, P extends Paths<T>>(obj: T, path: P): unknown {
  let cur: any = obj;
  for (const key of path.split('.')) cur = cur[key];
  return cur;
}
