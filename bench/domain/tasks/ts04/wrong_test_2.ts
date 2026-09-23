// The guard overload narrows the first half but leaves the second as the full T.
export function partition<T, U extends T>(items: readonly T[], predicate: (item: T, index: number) => item is U): [U[], T[]];
export function partition<T>(items: readonly T[], predicate: (item: T, index: number) => boolean): [T[], T[]];
export function partition<T>(items: readonly T[], predicate: (item: T, index: number) => boolean): [T[], T[]] {
  const yes: T[] = [];
  const no: T[] = [];
  items.forEach((item, i) => (predicate(item, i) ? yes : no).push(item));
  return [yes, no];
}
