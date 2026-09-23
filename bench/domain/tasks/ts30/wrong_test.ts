// Recurses into arrays with the key-remapping object branch, which turns arrays into objects with `length`, `map`, ...
export type CamelCase<S extends string> =
  S extends `${infer Head}_${infer Tail}` ? `${Head}${Capitalize<CamelCase<Tail>>}` : S;
export type CamelCaseKeys<T> =
  T extends object ? { [K in keyof T as K extends string ? CamelCase<K> : K]: CamelCaseKeys<T[K]> } : T;
const camel = (key: string): string => key.replace(/_([a-z0-9])/g, (_, c: string) => c.toUpperCase());
export function camelizeKeys<T>(value: T): CamelCaseKeys<T> {
  if (Array.isArray(value)) return value.map((v) => camelizeKeys(v)) as CamelCaseKeys<T>;
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [camel(k), camelizeKeys(v)])) as CamelCaseKeys<T>;
  }
  return value as CamelCaseKeys<T>;
}
