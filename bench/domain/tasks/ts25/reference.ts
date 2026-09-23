export type DeepReadonly<T> =
  T extends (...args: any[]) => unknown ? T
  : T extends Map<infer K, infer V> ? ReadonlyMap<DeepReadonly<K>, DeepReadonly<V>>
  : T extends Set<infer U> ? ReadonlySet<DeepReadonly<U>>
  : T extends object ? { readonly [P in keyof T]: DeepReadonly<T[P]> }
  : T;

export function deepFreeze<T>(value: T): DeepReadonly<T> {
  const seen = new WeakSet<object>();
  const visit = (v: unknown): void => {
    if ((typeof v !== 'object' && typeof v !== 'function') || v === null || seen.has(v)) return;
    seen.add(v);
    if (v instanceof Map) {
      for (const [k, x] of v) { visit(k); visit(x); }
    } else if (v instanceof Set) {
      for (const x of v) visit(x);
    }
    if (typeof v === 'object') {
      for (const key of Reflect.ownKeys(v)) visit((v as Record<PropertyKey, unknown>)[key]);
    }
    Object.freeze(v);
  };
  visit(value);
  return value as DeepReadonly<T>;
}
