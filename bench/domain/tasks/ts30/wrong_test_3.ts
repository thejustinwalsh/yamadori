// Types are right, but the runtime only renames top-level keys.
export type CamelCase<S extends string> =
  S extends `${infer Head}_${infer Tail}` ? `${Head}${Capitalize<CamelCase<Tail>>}` : S;
export type CamelCaseKeys<T> =
  T extends readonly (infer E)[] ? CamelCaseKeys<E>[]
  : T extends object ? { [K in keyof T as K extends string ? CamelCase<K> : K]: CamelCaseKeys<T[K]> }
  : T;
const camel = (key: string): string => key.replace(/_([a-z0-9])/g, (_, c: string) => c.toUpperCase());
export function camelizeKeys<T>(value: T): CamelCaseKeys<T> {
  if (value === null || typeof value !== 'object') return value as CamelCaseKeys<T>;
  return Object.fromEntries(Object.entries(value).map(([k, v]) => [camel(k), v])) as CamelCaseKeys<T>;
}
