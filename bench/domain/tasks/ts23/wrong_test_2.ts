// A recursive copy with no memo: cycles overflow the stack and shared references are duplicated.
export function snapshot<T>(value: T): T {
  const copy = (v: unknown): unknown => {
    if (typeof v === 'function') throw new TypeError('cannot clone a function');
    if (v === null || typeof v !== 'object') return v;
    if (v instanceof Date) return new Date(v.getTime());
    if (v instanceof RegExp) return new RegExp(v.source, v.flags);
    if (v instanceof Map) return new Map([...v].map(([k, x]) => [copy(k), copy(x)]));
    if (v instanceof Set) return new Set([...v].map(copy));
    if (ArrayBuffer.isView(v)) return (v as unknown as { slice(): object }).slice();
    if (Array.isArray(v)) return v.map(copy);
    return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, copy(x)]));
  };
  return copy(value) as T;
}
