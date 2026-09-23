// An arrow function works as an assertion function only through an explicit type annotation.
type AssertDefined = <T>(value: T, message?: string) => asserts value is T & {};
export const assertIsDefined: AssertDefined = (value, message) => {
  if (value == null) throw new TypeError(message ?? 'value is null or undefined');
};
