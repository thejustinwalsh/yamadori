export function snapshot<T>(value: T): T {
  return structuredClone(value);
}
