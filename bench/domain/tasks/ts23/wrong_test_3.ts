// A shallow copy: nested objects are still shared with the original.
export function snapshot<T>(value: T): T {
  if (value instanceof Map) return new Map(value) as T;
  if (value instanceof Set) return new Set(value) as T;
  if (Array.isArray(value)) return [...value] as T;
  if (value !== null && typeof value === 'object') return { ...value };
  return value;
}
