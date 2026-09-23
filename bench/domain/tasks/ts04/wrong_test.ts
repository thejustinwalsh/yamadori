// Only the boolean form: a type-guard predicate does not narrow either half.
export function partition<T>(items: readonly T[], predicate: (item: T, index: number) => boolean): [T[], T[]] {
  const yes: T[] = [];
  const no: T[] = [];
  items.forEach((item, i) => (predicate(item, i) ? yes : no).push(item));
  return [yes, no];
}
