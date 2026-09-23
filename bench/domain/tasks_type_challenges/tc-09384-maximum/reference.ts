// Counts up from 0, removing each value from the remaining set; the value
// that empties the set is the maximum. Negative numbers are out of scope.
type Maximum<T extends any[], U = T[number], C extends unknown[] = []> =
  [U] extends [never]
    ? never
    : [Exclude<U, C['length']>] extends [never]
      ? C['length']
      : Maximum<T, Exclude<U, C['length']>, [...C, unknown]>
