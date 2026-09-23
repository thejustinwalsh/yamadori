// Calls the predicate without the index argument.
export function partition<T, U extends T>(items: readonly T[], predicate: (item: T, index: number) => item is U): [U[], Exclude<T, U>[]];
export function partition<T>(items: readonly T[], predicate: (item: T, index: number) => boolean): [T[], T[]];
export function partition<T>(items: readonly T[], predicate: (item: T, index: number) => boolean): [T[], T[]] {
  const yes: T[] = [];
  const no: T[] = [];
  for (const item of items) ((predicate as (item: T) => boolean)(item) ? yes : no).push(item);
  return [yes, no];
}
