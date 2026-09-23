// The overloads expressed as an overloaded function type on a const.
type Partition = {
  <T, U extends T>(items: readonly T[], predicate: (item: T, index: number) => item is U): [U[], Exclude<T, U>[]];
  <T>(items: readonly T[], predicate: (item: T, index: number) => boolean): [T[], T[]];
};

export const partition = (<T,>(items: readonly T[], predicate: (item: T, index: number) => boolean): [T[], T[]] => {
  const pass: T[] = [];
  const fail: T[] = [];
  for (let i = 0; i < items.length; i++) {
    if (predicate(items[i]!, i)) pass.push(items[i]!);
    else fail.push(items[i]!);
  }
  return [pass, fail];
}) as Partition;
