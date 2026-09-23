// No Map/Set cases: the mapped type over Map leaves `set` and `add` callable.
export type DeepReadonly<T> =
  T extends (...args: any[]) => unknown ? T
  : T extends object ? { readonly [P in keyof T]: DeepReadonly<T[P]> }
  : T;
export function deepFreeze<T>(value: T): DeepReadonly<T> {
  const seen = new WeakSet<object>();
  const visit = (v: unknown): void => {
    if (typeof v !== 'object' || v === null || seen.has(v)) return;
    seen.add(v);
    if (v instanceof Map) for (const [k, x] of v) { visit(k); visit(x); }
    if (v instanceof Set) for (const x of v) visit(x);
    for (const key of Object.keys(v)) visit((v as Record<string, unknown>)[key]);
    Object.freeze(v);
  };
  visit(value);
  return value as DeepReadonly<T>;
}
