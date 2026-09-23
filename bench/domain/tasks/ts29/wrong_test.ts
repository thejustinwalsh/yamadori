// jaccard of two empty sets is 0/0 = NaN instead of 1.
export type TagDiff = { added: Set<string>; removed: Set<string>; kept: Set<string> };
export function diffTags(before: ReadonlySet<string>, after: ReadonlySet<string>): TagDiff {
  return { added: after.difference(before), removed: before.difference(after), kept: before.intersection(after) };
}
export function jaccard(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  return a.intersection(b).size / a.union(b).size;
}
