// Handles arrays/tuples and ReadonlyMap/ReadonlySet inputs explicitly, and freezes children first
// with a recursive function and an explicit visited set.
type Primitive = string | number | boolean | bigint | symbol | null | undefined;

export type DeepReadonly<T> =
  T extends Primitive | Function ? T
  : T extends ReadonlyMap<infer K, infer V> ? ReadonlyMap<DeepReadonly<K>, DeepReadonly<V>>
  : T extends ReadonlySet<infer U> ? ReadonlySet<DeepReadonly<U>>
  : T extends readonly unknown[] ? { readonly [I in keyof T]: DeepReadonly<T[I]> }
  : { readonly [P in keyof T]: DeepReadonly<T[P]> };

function freezeAll(v: unknown, visited: Set<unknown>): void {
  if (v === null || typeof v !== 'object' || visited.has(v)) return;
  visited.add(v);
  if (v instanceof Map) v.forEach((val, key) => { freezeAll(key, visited); freezeAll(val, visited); });
  if (v instanceof Set) v.forEach((val) => freezeAll(val, visited));
  for (const k of Object.keys(v)) freezeAll((v as { [k: string]: unknown })[k], visited);
  Object.freeze(v);
}

export const deepFreeze = <T,>(value: T): DeepReadonly<T> => {
  freezeAll(value, new Set());
  return value as DeepReadonly<T>;
};
