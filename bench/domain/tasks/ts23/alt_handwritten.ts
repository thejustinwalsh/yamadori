// A hand-written deep copy with a memo table for cycles and shared references.
export function snapshot<T>(value: T): T {
  const seen = new Map<object, unknown>();

  const copy = (v: unknown): unknown => {
    if (typeof v === 'function' || typeof v === 'symbol') throw new TypeError(`cannot clone ${typeof v}`);
    if (v === null || typeof v !== 'object') return v;
    if (seen.has(v)) return seen.get(v);

    if (v instanceof Date) { const d = new Date(v.getTime()); seen.set(v, d); return d; }
    if (v instanceof RegExp) { const r = new RegExp(v.source, v.flags); seen.set(v, r); return r; }
    if (ArrayBuffer.isView(v) && !(v instanceof DataView)) {
      const ta = v as unknown as { slice(): object };
      const c = ta.slice();
      seen.set(v, c);
      return c;
    }
    if (v instanceof Map) {
      const m = new Map();
      seen.set(v, m);
      for (const [k, val] of v) m.set(copy(k), copy(val));
      return m;
    }
    if (v instanceof Set) {
      const s = new Set();
      seen.set(v, s);
      for (const item of v) s.add(copy(item));
      return s;
    }
    if (Array.isArray(v)) {
      const a: unknown[] = [];
      seen.set(v, a);
      v.forEach((item, i) => { a[i] = copy(item); });
      return a;
    }
    const o: Record<string, unknown> = {};
    seen.set(v, o);
    for (const k of Object.keys(v)) o[k] = copy((v as Record<string, unknown>)[k]);
    return o;
  };

  return copy(value) as T;
}
