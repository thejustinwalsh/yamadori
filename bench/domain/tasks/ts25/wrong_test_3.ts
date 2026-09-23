// Types are right, but the runtime freeze never looks inside Map and Set entries.
export type DeepReadonly<T> =
  T extends (...args: any[]) => unknown ? T
  : T extends Map<infer K, infer V> ? ReadonlyMap<DeepReadonly<K>, DeepReadonly<V>>
  : T extends Set<infer U> ? ReadonlySet<DeepReadonly<U>>
  : T extends object ? { readonly [P in keyof T]: DeepReadonly<T[P]> }
  : T;
export function deepFreeze<T>(value: T): DeepReadonly<T> {
  const seen = new WeakSet<object>();
  const visit = (v: unknown): void => {
    if (typeof v !== 'object' || v === null || seen.has(v)) return;
    seen.add(v);
    for (const key of Object.keys(v)) visit((v as Record<string, unknown>)[key]);
    Object.freeze(v);
  };
  visit(value);
  return value as DeepReadonly<T>;
}
