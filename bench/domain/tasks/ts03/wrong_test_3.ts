// Unannotated arrow function: TS2775, it cannot be used as an assertion at the call site.
export const assertIsDefined = <T,>(value: T, message?: string): asserts value is NonNullable<T> => {
  if (value === null || value === undefined) throw new TypeError(message ?? 'value is not defined');
};
