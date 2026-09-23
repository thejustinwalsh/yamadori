// A type predicate instead of an assertion signature: nothing is narrowed after a bare call.
export function assertIsDefined<T>(value: T, message?: string): value is NonNullable<T> {
  if (value === null || value === undefined) throw new TypeError(message ?? 'value is not defined');
  return true;
}
