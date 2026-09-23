// Shallow: only the top level is readonly and frozen.
export type DeepReadonly<T> = Readonly<T>;
export function deepFreeze<T>(value: T): DeepReadonly<T> {
  return Object.freeze(value);
}
