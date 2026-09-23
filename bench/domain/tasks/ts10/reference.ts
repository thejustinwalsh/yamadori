type Leaf = string | number | boolean | bigint | symbol | null | undefined | Date | readonly unknown[] | ((...args: never[]) => unknown);

export type Paths<T> = T extends Leaf
  ? never
  : { [K in keyof T & string]: K | `${K}.${Paths<T[K]>}` }[keyof T & string];

export type PathValue<T, P extends string> = P extends `${infer K}.${infer Rest}`
  ? K extends keyof T ? PathValue<T[K], Rest> : never
  : P extends keyof T ? T[P] : never;

export function get<T, P extends Paths<T>>(obj: T, path: P): PathValue<T, P> {
  let cur: unknown = obj;
  for (const key of path.split('.')) cur = (cur as Record<string, unknown>)[key];
  return cur as PathValue<T, P>;
}
