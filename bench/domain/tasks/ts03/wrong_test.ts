// Truthiness check: rejects the legitimate defined values 0, '' and false.
export function assertIsDefined<T>(value: T, message?: string): asserts value is NonNullable<T> {
  if (!value) throw new TypeError(message ?? 'value is not defined');
}
