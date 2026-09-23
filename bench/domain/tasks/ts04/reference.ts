export function partition<T, U extends T>(
  items: readonly T[],
  predicate: (item: T, index: number) => item is U,
): [U[], Exclude<T, U>[]];
export function partition<T>(
  items: readonly T[],
  predicate: (item: T, index: number) => boolean,
): [T[], T[]];
export function partition<T>(
  items: readonly T[],
  predicate: (item: T, index: number) => boolean,
): [T[], T[]] {
  const yes: T[] = [];
  const no: T[] = [];
  items.forEach((item, i) => (predicate(item, i) ? yes : no).push(item));
  return [yes, no];
}
