export function assertIsDefined<T>(value: T, message?: string): asserts value is NonNullable<T> {
  if (value === null || value === undefined) {
    throw new TypeError(message ?? `expected a defined value, got ${value}`);
  }
}
