// Only leaf paths: intermediate objects such as `server` and `server.tls` are missing.
type Leaf = string | number | boolean | bigint | symbol | null | undefined | Date | readonly unknown[] | ((...args: never[]) => unknown);
export type Paths<T> = {
  [K in keyof T & string]: T[K] extends Leaf ? K : `${K}.${Paths<T[K]>}`;
}[keyof T & string];
export type PathValue<T, P extends string> = P extends `${infer K}.${infer Rest}`
  ? K extends keyof T ? PathValue<T[K], Rest> : never
  : P extends keyof T ? T[P] : never;
export function get<T, P extends Paths<T>>(obj: T, path: P): PathValue<T, P> {
  let cur: any = obj;
  for (const key of path.split('.')) cur = cur[key];
  return cur;
}
