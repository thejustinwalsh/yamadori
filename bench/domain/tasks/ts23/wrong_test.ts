// The JSON round-trip: Dates become strings, Maps and Sets become {}, cycles throw.
export function snapshot<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}
