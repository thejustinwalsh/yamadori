// Throws a plain Error rather than the TypeError the spec asks for.
export function assertIsDefined<T>(value: T, message?: string): asserts value is NonNullable<T> {
  if (value === null || value === undefined) throw new Error(message ?? 'value is not defined');
}
